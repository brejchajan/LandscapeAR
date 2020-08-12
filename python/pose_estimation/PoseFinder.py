# @Author: Jan Brejcha <ibrejcha>
# @Date:   2020-08-11T17:47:03+02:00
# @Email:  ibrejcha@fit.vutbr.cz; brejchaja@gmail.com
# @Project: LandscapeAR
# @Last modified by:   ibrejcha
# @Last modified time: 2020-08-12T11:43:51+02:00
# @License: Copyright 2020 CPhoto@FIT, Brno University of Technology,
# Faculty of Information Technology,
# Božetěchova 2, 612 00, Brno, Czech Republic
#
# Redistribution and use in source code form, with or without modification,
# are permitted provided that the following conditions are met:
#
# 1. Redistributions must retain the above copyright notice, this list of
#    conditions and the following disclaimer.
#
# 2. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# 3. Redistributions must be pursued only for non-commercial research
#    collaboration and demonstration purposes.
#
# 4. Where separate files retain their original licence terms
#    (e.g. MPL 2.0, Apache licence), these licence terms are announced, prevail
#    these terms and must be complied.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF  FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
# OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
# OF SUCH DAMAGE.

import torch
import torchvision
import cv2
import os
import exiftool
import glob
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from pyquaternion import Quaternion
import pickle
import sys
from torch.autograd import Variable

# our code
from sfm.Matcher import Matcher
from trainPatchesDescriptors import getCoordsAtSiftKeypoints, getPatchesFromImg
from trainPatchesDescriptors import findLatestModelPath, show, plotMatches
from pose_estimation.patchSamplingDepth import loadDepth, unproject, projectWithIntrinsics
from pose_estimation.patchSamplingDepth import projectWithIntrinsicsAndZ, savePointCloudToPly
from pose_estimation.patchSamplingDepth import getSizeFOV
from pose_estimation import FUtil
from pose_estimation import BundleAdjustment
from pose_estimation.EstimatePose import poseFrom2D3DWithFOV, poseEPNPBAIterative
from pose_estimation.EstimatePose import poseFrom2D3DWithFOVEPNPOurRansac, poseFrom2D3DP4Pf
from training.MultimodalPatchesDataset import MultimodalPatchesDataset
from render_panorama import checkImagesAlreadyRendered
from pose_estimation.KeypointDetector import KeypointDetector
from training.Architectures import MultimodalKeypointPatchNet5lShared2lFCN
from training.Architectures import MultimodalKeypointPatchNet5lShared2l
import training.Architectures as Architectures

# NCNet
from thirdparty.ncnet.lib.model import ImMatchNet
from thirdparty.ncnet.lib.torch_util import BatchTensorToVars
from thirdparty.ncnet.lib.point_tnf import corr_to_matches

# D2net
from thirdparty.d2net.lib.model_test import D2Net

# AffNet
from thirdparty.affnet.architectures import AffNetFast
from thirdparty.affnet.SparseImgRepresenter import ScaleSpaceAffinePatchExtractor
from PIL import Image
from thirdparty.affnet.HardNet import HardNet

# saddle keypoint detector
try:
    import pysaddlepts
    pysaddlepts_available = True
except Exception:
    pysaddlepts_available = False


class PoseFinder(object):

    def __init__(self, args):
        self.gps = None
        if args.gps:
            self.gps = args.gps
        self.fov = None
        if args.fov:
            self.fov = args.fov

        self.best_buddy_refine = args.best_buddy_refine
        self.query_img_path = args.query_image
        self.grid_radius = args.grid_options[0]
        self.grid_offset = args.grid_options[1]
        self.do_voting = not args.no_voting
        self.refine = False
        self.use_affnet = args.use_hardnet
        self.patchsize = 64
        if self.use_affnet:
            self.patchsize = 32
        self.cuda = args.cuda
        print("Cuda: ", self.cuda)
        self.voting_cnt = args.voting_cnt
        self.matching_dir = args.matching_dir
        self.maxres = args.maxres
        self.fcn_keypoints = args.fcn_keypoints
        self.fcn_keypoints_multiscale = args.fcn_keypoints_multiscale
        self.use_d2net = args.d2net
        self.use_ncnet = args.ncnet
        self.sift_descriptors = args.sift_descriptors
        self.dense_uniform_keypoints = args.dense_uniform_keypoints
        self.dense_halton_keypoints = args.dense_halton_keypoints
        self.saddle_keypoints = args.saddle_keypoints
        if (self.saddle_keypoints and not pysaddlepts_available):
            raise Exception("Module pysaddlepts not available. \
            Please, install pysaddlepts to your PYTHONPATH from: \
            https://github.com/brejchajan/saddle_detector.git")
        if self.saddle_keypoints:
            self.sorb = pysaddlepts.cmp_SORB(nfeatures=10000)
        self.needles = False
        self.stride = args.stride

        self.use_depth = args.use_depth
        self.use_normals = args.use_normals
        self.use_silhouettes = args.use_silhouettes

        if not os.path.isfile(self.query_img_path):
            raise RuntimeError("Unable to find query image file.", self.query_img_path)

        self.working_dir = args.working_directory
        self.earth_file = args.earth_file
        self.p4pf = args.p4pf
        self.p4pf_epnp_iterative = args.p4pf_epnp_iterative
        self.epnpor = args.epnpor
        self.device = torch.device("cuda:0" if args.cuda else "cpu")

        log_dir = os.path.join(args.log_dir, args.snapshot)

        models_path = os.path.join(log_dir, "models")
        if args.model_name:
            model_path = os.path.join(models_path, args.model_name)
            model_name = os.path.basename(model_path).split("_epoch_")[0]
        else:
            model_path = findLatestModelPath(models_path)
            model_name = os.path.basename(model_path).split("_epoch_")[0]

        if self.use_affnet:
            # HardNet
            self.hn_descriptor = HardNet()
            model_weights = 'pretrained/hardnet/HardNet++.pth'
            if args.cuda:
                hncheckpoint = torch.load(model_weights)
            else:
                hncheckpoint = torch.load(model_weights, map_location='cpu')
            # model_state_dict for our net, state_dict for orig
            self.hn_descriptor.load_state_dict(hncheckpoint['state_dict'])
            self.hn_descriptor.eval()

            # AffNet
            self.PS = 32
            self.AffNetPix = AffNetFast(PS = self.PS)
            weightd_fname = 'pretrained/affnet/AffNet.pth'
            if args.cuda:
                checkpoint = torch.load(weightd_fname)
            else:
                checkpoint = torch.load(weightd_fname, map_location='cpu')
            self.AffNetPix.load_state_dict(checkpoint['state_dict'])
            self.AffNetPix.eval()
            self.affnet_detector = ScaleSpaceAffinePatchExtractor( mrSize = 5.192, num_features = 3000,
                                              border = 5, num_Baum_iters = 1,
                                              AffNet = self.AffNetPix)

        # Our detector
        module = __import__("training").Architectures
        net_class = getattr(module, model_name)
        if self.fcn_keypoints or self.fcn_keypoints_multiscale:
            if  net_class.__name__ != "MultimodalKeypointPatchNet5lShared2lFCN":
                print("loading FCN variant")
                self.net = Architectures.MultimodalPatchNet5lShared2lFCN(args.normalize_output)
                self.net_keypoints = Architectures.MultimodalPatchNet5lShared2lFCN(False,
                strides=[1], strides_branch=[1, 1, 1, 1],
                dilations=[1], dilations_branch=[1, 1, 1, 1])
            else:
                self.net = MultimodalKeypointPatchNet5lShared2lFCN(
                    args.normalize_output, pretrained=False
                )
                self.net_keypoints = MultimodalKeypointPatchNet5lShared2lFCN(
                    args.normalize_output,
                    strides=[1], strides_branch=[2, 1, 1, 1],
                    dilations=[2], dilations_branch=[1, 1, 2, 2],
                    pretrained=False
                )
        else:
            self.net = net_class(args.normalize_output)
        self.loadModel(model_path)
        if args.cuda:
            self.net = self.net.to(self.device)
            if self.fcn_keypoints or self.fcn_keypoints_multiscale:
                self.net_keypoints = self.net_keypoints.to(self.device)
            if self.use_affnet:
                self.affnet_detector = self.affnet_detector.to(self.device)
                self.hn_descriptor = self.hn_descriptor.to(self.device)
        self.net.eval()
        if self.fcn_keypoints or self.fcn_keypoints_multiscale:
            self.net_keypoints.eval()

        self.has_keypoints = False
        if (isinstance(self.net, MultimodalKeypointPatchNet5lShared2l)
            or isinstance(self.net, MultimodalKeypointPatchNet5lShared2lFCN)):
            self.has_keypoints = True

        #D2net
        if self.use_d2net:
            pathname = os.path.dirname(sys.argv[0])
            model_file='pretrained/d2_net/d2_tf.pth'

            d2_model_path = os.path.join(pathname, model_file)
            self.d2net = D2Net(
                model_file=d2_model_path,
                use_relu=True,
                use_cuda=args.cuda
            )

        if self.use_ncnet:
            self.ncnet = ImMatchNet(use_cuda=True, checkpoint='pretrained/ncnet/ncnet_pfpascal.pth.tar')

    def loadModel(self, model_path):
        print("Loading model: ", model_path)
        checkpoint = torch.load(model_path, map_location=self.device)
        if self.fcn_keypoints or self.fcn_keypoints_multiscale:
            x = checkpoint['model_state_dict']['conv2.weight']
            checkpoint['model_state_dict']['conv2.weight'] = x.reshape(x.shape[0], x.shape[1], 1, 1)
            print(checkpoint['model_state_dict'].keys())
            self.net_keypoints.load_state_dict(checkpoint['model_state_dict'])
        self.net.load_state_dict(checkpoint['model_state_dict'])

    def getGPS(self):
        with exiftool.ExifTool() as et:
            if self.gps:
                if self.fov is None:
                    res = et.execute_json("-n", "-FOV", self.query_img_path)
                lat = self.gps[0]
                lon = self.gps[1]
            else:
                if self.fov is None:
                    res = et.execute_json("-n", "-GPSLatitude", "-GPSLongitude",
                                      "-FOV", self.query_img_path)
                else:
                    res = et.execute_json("-n", "-GPSLatitude", "-GPSLongitude",
                                          self.query_img_path)
                print(res[0])
                lat = res[0]['EXIF:GPSLatitude']
                lon = res[0]['EXIF:GPSLongitude']
            if self.fov:
                print("Using fov from command line: ", self.fov)
                fov = self.fov
            elif 'Composite:FOV' in res[0]:
                fov = res[0]['Composite:FOV']
            else:
                print("WARNING: FieldOfView not found in image EXIF, fallback to 60degrees.")
                fov = 60
            fov = (fov / 180.0) * np.pi
            return lat, lon, fov
        raise RuntimeError("Unable to get GPS position from the photo.")

    def loadImage(self, img_name, fov, scale=False):
        img = cv2.imread(img_name)

        if scale:
            wp, hp, scale_p = getSizeFOV(img.shape[1], img.shape[0], fov, maxres=self.maxres)
            img = cv2.resize(img, (wp, hp))
        else:
            scale = 1024.0 / max(img.shape[1], img.shape[0])
            wp = int(np.round(img.shape[1] * scale))
            hp = int(np.round(img.shape[0] * scale))
            img = cv2.resize(img, (wp, hp))
            print("loaded image size", img.shape)
        img = np.flip(img, 2)
        return img

    def loadImageAndSift(self, img_name, fov, contrastThreshold=0.04, edgeThreshold=10,
                         sigma=1.6, nfeatures=5000, scale=False):
        img = self.loadImage(img_name, fov, scale)
        img = np.flip(img, 2)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        sift = cv2.xfeatures2d.SIFT_create(nfeatures=nfeatures,
                                           contrastThreshold=contrastThreshold,
                                           edgeThreshold=edgeThreshold,
                                           sigma=sigma)
        kp, ds_sift = sift.detectAndCompute(gray, None)
        if len(kp) == 0:
            return img, kp, ds_sift
        coords = getCoordsAtSiftKeypoints(kp)
        sel = np.logical_and(coords > self.patchsize, coords < (np.array(img.shape[:2]).reshape(1, 2) - self.patchsize))
        sel = np.logical_and(sel[:, 0], sel[:, 1])
        nkp = []
        for idx in range(0, sel.shape[0]):
            if sel[idx]:
                nkp.append(kp[idx])
        kp = nkp
        ds_sift = ds_sift[sel]

        img = np.flip(img, 2)  # flip channels as opencv treats them as BGR
        return img, kp, ds_sift

    def describePhoto(self, patches1, batchsize=50):
        p1 = []
        if self.has_keypoints:
            p1_scores = []
        else:
            p1_scores = np.ones(patches1.shape[0])
        if self.use_affnet:
            patches1 = (0.299 * patches1[:, 0, :, :]) + (0.587 * patches1[:, 1, :, :]) + (0.114 * patches1[:, 2, :, :])
            patches1 = patches1[:, None, :, :]
        for idx in range(0, patches1.shape[0], batchsize):
            batch = patches1[idx:idx + batchsize].to(self.device)
            if self.use_affnet:
                p1_fea = self.hn_descriptor(batch).detach().cpu().numpy()
            else:
                p1_fea = self.net.forward_photo(batch)
                if self.has_keypoints:
                    p1_score = p1_fea[1].detach().cpu().numpy()
                    p1_scores.append(p1_score)
                    p1_fea = p1_fea[0]
                p1_fea = p1_fea.detach().cpu().numpy()
            p1.append(p1_fea)
        p1 = np.concatenate(p1)
        p1 = p1.reshape(p1.shape[0], p1.shape[1])
        if self.has_keypoints:
            p1_scores = np.concatenate(p1_scores)
        return p1, p1_scores

    def describeRender(self, patches1, batchsize=50):
        p1 = []
        if self.use_affnet:
            patches1 = (0.299 * patches1[:, 0, :, :]) + (0.587 * patches1[:, 1, :, :]) + (0.114 * patches1[:, 2, :, :])
            patches1 = patches1[:, None, :, :]
        if self.has_keypoints:
            p1_scores = []
        else:
            p1_scores = np.ones(patches1.shape[0])
        for idx in range(0, patches1.shape[0], batchsize):
            batch = patches1[idx:idx + batchsize].to(self.device)
            if self.use_affnet:
                p1_fea = self.hn_descriptor(batch).detach().cpu().numpy()
            else:
                p1_fea = self.net.forward_render(batch)
                if self.has_keypoints:
                    p1_score = p1_fea[1].detach().cpu().numpy()
                    p1_scores.append(p1_score)
                    p1_fea = p1_fea[0]
                p1_fea = p1_fea.detach().cpu().numpy()
            p1.append(p1_fea)
        p1 = np.concatenate(p1)
        p1 = p1.reshape(p1.shape[0], p1.shape[1])
        if self.has_keypoints:
            p1_scores = np.concatenate(p1_scores)
        return p1, p1_scores

    def loadAdditionalChannels(self, img1, depth):
        if len(depth.shape) == 2:
            depth = depth[:, :, None]
        if self.use_depth:
            img1 = np.concatenate([img1, depth / 500000.0], axis=2)

        if self.use_normals:
            normals = MultimodalPatchesDataset.normalsFromDepth(depth)
            img1 = np.concatenate([img1, normals], axis=2)

        if self.use_silhouettes:
            silhouettes = MultimodalPatchesDataset.silhouettesFromDepth(depth)
            img1 = np.concatenate([img1, silhouettes], axis=2)
        return img1

    def detectSiftAndDescribePhoto(self, photo_name, fov,
                                   contrastThr=0.01, edgeThr=10, sigma=1.0,
                                   scale=False, nfeatures=5000):
        print("Detecting sift photo, scale:", scale)
        img1, kp1, ds1_sift = self.loadImageAndSift(photo_name, fov, contrastThr,
                                               edgeThr, sigma, nfeatures=nfeatures, scale=scale)
        print("Scaled photo width: ", img1.shape)
        if len(kp1) == 0:
            kp1 = []
            ds1_sift = []
            p1 = []
            patches1 = []
            return img1, kp1, ds1_sift, p1, patches1

        coords1 = getCoordsAtSiftKeypoints(kp1)
        patches1 = getPatchesFromImg(img1, coords1, fov, patchsize=self.patchsize, maxres=self.maxres)

        if self.sift_descriptors:
            p1 = ds1_sift
        else:
            p1, p1_scores = self.describePhoto(patches1)
        print("desc photo shape", p1.shape)
        return img1, kp1, ds1_sift, p1, patches1

    def detectSiftAndDescribeRender(self, render_name, fov,
                                   contrastThr=0.02, edgeThr=15, sigma=1.0,
                                   scale=False, nfeatures=5000):
        print("Detecting sift render, scale:", scale)
        start = timer()
        img1, kp1, ds1_sift = self.loadImageAndSift(render_name, fov, contrastThr,
                                               edgeThr, sigma, nfeatures=nfeatures, scale=scale)
        depth_path = os.path.splitext(render_name)[0] + "_depth.txt.gz"
        depth = loadDepth(depth_path)
        depth = cv2.resize(depth, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_AREA)
        img1 = self.loadAdditionalChannels(img1, depth)
        nkp = []
        sel = []
        for kp in kp1:
            depth_at_kp = depth[np.round(kp.pt[1]).astype(np.int), np.round(kp.pt[0]).astype(np.int)]
            if depth_at_kp < 400000:
                nkp.append(kp)
                sel.append(True)
            else:
                sel.append(False)
        kp1 = nkp
        print("image load and sift: ", timer() - start)

        coords1 = getCoordsAtSiftKeypoints(kp1)
        start = timer()
        if len(coords1) > 0:
            patches1 = getPatchesFromImg(img1, coords1, fov, patchsize=self.patchsize, maxres=self.maxres)
            print("get patches: ", timer() - start)

            if self.sift_descriptors:
                p1 = ds1_sift[sel]
            else:
                start = timer()
                p1, p1_scores = self.describeRender(patches1)
                print("NET FORWARD: ", timer() - start)
        else:
            kp1 = []
            ds1_sift = []
            p1 = []
            patches1 = []

        return img1, kp1, ds1_sift, p1, patches1

    def load_grayscale_var(self, fname):
        img = Image.open(fname).convert('RGB')
        img = np.mean(np.array(img), axis = 2)
        var_image = torch.autograd.Variable(torch.from_numpy(img.astype(np.float32)), volatile = True)
        var_image_reshape = var_image.view(1, 1, var_image.size(0),var_image.size(1))
        if self.cuda:
            var_image_reshape = var_image_reshape.cuda()
        return var_image_reshape

    def get_geometry_affnet(self, img_name):
        img = self.load_grayscale_var(img_name)
        with torch.no_grad():
            LAFs, resp = self.affnet_detector(img, do_ori = True)
            patches = self.affnet_detector.extract_patches_from_pyr(LAFs, PS = self.PS)
            kp = []
            for idx in range(0, LAFs.shape[0]):
                kp.append(cv2.KeyPoint(LAFs[idx, 0, 2], LAFs[idx, 1, 2], 32))
            plt.show()
        return patches, kp

    @staticmethod
    def exportPoseToNVM(query_img_path, scene_center, img, intr, pose, pt2D, pt3D, output_dir):
        pose = np.copy(pose)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        output_name = os.path.join(output_dir, "sfm_data.nvm")
        output_img_name = "image_" + os.path.basename(query_img_path)
        img = np.flip(img, 2)
        cv2.imwrite(os.path.join(output_dir, output_img_name), img)

        R = pose[:3, :3]
        t = pose[:3, 3]
        R[1:3, :] = -R[1:3, :]
        t[1:3] = -t[1:3]

        cam_center = scene_center + np.dot(-R.transpose(), t)

        focal = intr[0, 0]
        quat = Quaternion(matrix=R)

        with open(output_name, 'w') as f:
            f.write("NVM_V3\n")
            f.write("\n")
            f.write("1\n") #number of cameras
            f.write(output_img_name + " " + str(focal) + " " + str(quat.w) + " " + str(quat.x) + " " + str(quat.y) + " " + str(quat.z) + " " + str(cam_center[0]) + " " + str(cam_center[1]) + " " + str(cam_center[2]) + " 0 0\n")

            # write points
            pt3D = pt3D + scene_center
            f.write(str(pt3D.shape[0]) + "\n")
            for idx in range(0, pt3D.shape[0]):
                f.write(str(pt3D[idx, 0]) + " " + str(pt3D[idx, 1]) + " " + str(pt3D[idx, 2]) + " " + "255 0 0 1 0 " + str(idx) + " " + str(pt2D[idx, 0]) + " " + str(pt2D[idx, 1]) + "\n")

    def renderImagesAtGPS(self, lat, lon, output_dir):
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        if (checkImagesAlreadyRendered(output_dir)):
            print("All image are already rendered, skipping...")
            return

        # render the images
        r = str(self.grid_radius)
        o = str(self.grid_offset)
        if self.grid_offset > 0:
            count_expected = (self.grid_radius * 2) / self.grid_offset
        else:
            count_expected = 1
        # it is a squared grid; faces per pano is 12
        count_expected = count_expected * count_expected * 12

        cmd = "source ~/.bashrc; itr --egl 0 --render-grid " + str(lat) + " " + str(lon) + " " + r + " " + r + " " + o + " " + o + " 1200 " + output_dir + " --render-grid-mode perspective " + self.earth_file
        start = timer()
        print(cmd)
        os.system(cmd)
        print("Rendering took: ", timer() - start)

        if (not checkImagesAlreadyRendered(output_dir)):
            raise RuntimeError("The images were not rendered. Please check \n\
            whether itr is installed and works properly.")

    def getRenderedImageNames(self, image_base):
        output_dir = os.path.join(self.working_dir, image_base)
        images = glob.glob(os.path.join(output_dir, "*.png"))
        return images

    def matching(self, img1, img2, desc1, desc2, kp1, kp2, patches1, patches2, fov, P2, RT2, img2_depth, patches_outfile, matches_outfile, lowe_ratio=2.0, save_patches=False, fundam_filter=False, best_buddy=False, pnp_filter=True):

        if self.use_ncnet:
            kp1, kp2, good = PoseFinder.runNCNet(img1, img2, self.ncnet)
        else:
            good = Matcher.matchDescriptors(desc1, desc2, self.cuda)
        print("descriptors matched, estimating pose. Number of matches: ", len(good))
        query_idxs = []
        train_idxs = []
        for m in good:
            query_idxs.append(m.queryIdx)
            train_idxs.append(m.trainIdx)

        if self.refine:
            fov_render, fovy_render = FUtil.projectiveToFOV(P2)
            coords1_ref, coords2_ref, new_good, src_desc, dst_desc = PoseFinder.refineSparseAtRes(2*self.maxres, img1, img2, fov, fov_render, kp1, kp2, good, self, radius=10, patchsize=self.patchsize, cuda=self.cuda)
            good = new_good
        else:
            src_desc = desc1
            dst_desc = desc2

        if best_buddy:
            # filter ambiguous
            closest = {}
            for m in good:
                if m.trainIdx in closest:
                    closest_match = closest[m.trainIdx]
                    if m.distance < closest_match.distance:
                        closest_match = m
                    closest[m.trainIdx] = closest_match
                else:
                    closest.update({m.trainIdx:m})
            good = []
            for trainIdx, match in closest.items():
                good.append(match)

            count_matches_good = len(good)
            print("num matches after best buddies: ", count_matches_good)

        if save_patches:
            fig1 = plt.figure(dpi=600)
            plt.subplot(1, 2, 1)
            number_rows = int(np.floor(np.sqrt(query_idxs.shape[0])))
            if number_rows > 0:
                show(torchvision.utils.make_grid(patches1[query_idxs], nrow=number_rows))
                plt.subplot(1, 2, 2)
                show(torchvision.utils.make_grid(patches2[train_idxs], nrow=number_rows))
                plt.savefig(patches_outfile)
            plt.close(fig1)

        mparts = os.path.splitext(matches_outfile)
        matches_outfile_good = mparts[0] + "_good" + mparts[1]
        matches_outfile_reproj = mparts[0] + "_reproj" + mparts[1]
        matches_outfile_pose = mparts[0] + "_pose.txt"
        matches_outfile_info = mparts[0] + "_info.txt"
        plotMatches(img1, kp1, img2[:, :, :3], kp2, good, matches_outfile_good)

        R = -1
        t = -1
        res = False
        count_matches_pose = 0
        if len(good) > 10:
            src_pts = np.float64([kp1[m.queryIdx].pt for m in good])
            dst_pts = np.float64([kp2[m.trainIdx].pt for m in good])

            if fundam_filter:
                M, mask = cv2.findFundamentalMat(src_pts, dst_pts,
                                                 cv2.FM_RANSAC, 3.0)

                src_pts = src_pts[mask[:, -1].astype(np.bool), :2]
                dst_pts = dst_pts[mask[:, -1].astype(np.bool), :2]

                ngood = []
                for i in range(0, mask.shape[0]):
                    if mask[i] != 0:
                        g = good[i]
                        ngood.append(g)
                good = ngood

            if pnp_filter:
                coords3d = unproject(dst_pts, img2_depth, RT2, P2)[:, :3]
                res = False
                if self.p4pf or self.p4pf_epnp_iterative:
                    res, R, t, f_, mask = poseFrom2D3DP4Pf(src_pts, coords3d)
                    if res:
                        # change FOV to the estimated one
                        fov = 2 * np.arctan2(img1.shape[1] / 2.0, f_)
                        if self.p4pf_epnp_iterative:
                            R, t, fov, mask = poseEPNPBAIterative(
                                R, t, fov, mask, img1.shape, src_pts, coords3d
                            )

                elif self.epnpor:
                    res, R, t, mask = poseFrom2D3DWithFOVEPNPOurRansac(src_pts, fov, img1.shape, coords3d)
                else:
                    res, R, t, mask = poseFrom2D3DWithFOV(src_pts, fov, img1.shape, coords3d)
                if not res:
                    return None, None, src_desc, dst_desc, good

                print("Pose result: ")
                print(R)
                print(t)
                num_inliers = np.sum(mask)
                print("num inliers: ", num_inliers)

                plotMatches(img1, kp1, img2[:, :, :3], kp2, good, matches_outfile, mask)

                orig_src_pts = np.copy(src_pts)
                orig_dst_pts = np.copy(dst_pts)
                src_pts = src_pts[mask[:, -1].astype(np.bool), :2]
                dst_pts = dst_pts[mask[:, -1].astype(np.bool), :2]
                coords3d = coords3d[mask[:, -1].astype(np.bool), :3]


                p4d = np.concatenate((coords3d, np.ones((coords3d.shape[0], 1))), axis=1)
                photo_estimated_MV = np.ones((4,4))
                photo_estimated_MV[:3, :3] = R
                photo_estimated_MV[:3, 3] = t
                intr = FUtil.fovToIntrinsics(fov, img1.shape[1], img1.shape[0])
                pt_2d = projectWithIntrinsics(p4d, img1.shape[1], img1.shape[0], photo_estimated_MV, intr)
                mean_repr_err = np.mean(np.linalg.norm(src_pts - pt_2d, axis=1))
                print("Mean reprojection error pair: ", mean_repr_err)

                radial = np.array([0, 0])
                # run bundle adjustment
                left_cam_idx = np.zeros(src_pts.shape[0]).astype(np.int)
                left_cam_pt_idx = np.arange(0, src_pts.shape[0]).astype(np.int)
                points2d = src_pts - np.array([[img1.shape[1] / 2.0, img1.shape[0] / 2.0]])
                R, t, res_intr, success = BundleAdjustment.bundleAdjustment([photo_estimated_MV], [intr], coords3d, points2d, left_cam_idx, left_cam_pt_idx)

                if not success:
                    return None, None, src_desc, dst_desc, good
                print("Pose result after Bundle Adjustment: ")
                print(R)
                print(t)
                print(res_intr)

                fov_bundle = 2 * np.arctan2(img1.shape[1] / 2.0, res_intr[0])
                # update intrisics with adjusted focal
                intr[0, 0] = res_intr[0]
                intr[1, 1] = -res_intr[0]
                photo_estimated_MV[:3, :3] = R
                photo_estimated_MV[:3, 3] = t
                radial = res_intr[1:3]

                np.savetxt(matches_outfile_pose, photo_estimated_MV)
                pt_2d = projectWithIntrinsics(p4d, img1.shape[1], img1.shape[0], photo_estimated_MV, intr, radial)
                pt_2d_noradial = projectWithIntrinsics(p4d, img1.shape[1], img1.shape[0], photo_estimated_MV, intr)

                mean_repr_err = np.mean(np.linalg.norm(src_pts - pt_2d, axis=1))
                print("Mean reprojection error pair after BA: ", mean_repr_err)
                with open(matches_outfile_info, 'w') as info:
                    info.write("Mean reprojection error: " + str(mean_repr_err) + "\n")
                    info.write("Num inliers: " + str(np.sum(num_inliers)) + "\n")
                    info.write("FOV: " + str(fov) + "\n")
                    info.write("FOV BA: " + str(fov_bundle) + "\n")

                sel1 = np.logical_and(pt_2d[:, 0] > 0, pt_2d[:, 0] <= img1.shape[1])
                sel2 = np.logical_and(pt_2d[:, 1] > 0, pt_2d[:, 1] <= img1.shape[0])
                sel = np.logical_and(sel1, sel2)

                fig = plt.figure(dpi=600)
                plt.imshow(img1)
                plt.scatter(src_pts[sel, 0], src_pts[sel, 1], c='green', s=1.0)
                plt.scatter(pt_2d_noradial[sel, 0], pt_2d_noradial[sel, 1], c='blue', s=1.0)
                plt.savefig(matches_outfile_reproj)
                plt.close(fig)

                if num_inliers >= 0:
                    return orig_src_pts, orig_dst_pts, src_desc, dst_desc, good
                else:
                    return None, None, src_desc, dst_desc, good
            return orig_src_pts, orig_dst_pts, src_desc, dst_desc, good

        return None, None, src_desc, dst_desc, good

    @staticmethod
    def parseInfoFile(info_file_name):
        with open(info_file_name, 'r') as f:
            lines = [l.strip() for l in f.readlines()]
            repr_err = float(lines[0].split(":")[1].strip())
            num_inliers = float(lines[1].split(":")[1].strip())
            fov = None
            fov_bundle = None
            if len(lines) > 2:
                fov = float(lines[2].split(":")[1].strip())
                fov_bundle = float(lines[3].split(":")[1].strip())
            return repr_err, num_inliers, fov, fov_bundle
        return None, None, None

    @staticmethod
    def findBestPose(output_dir_matching):
        # parse info files
        info_path = os.path.join(output_dir_matching, "*_matches_info.txt")
        info_files = glob.glob(info_path)
        repr_errors = []
        inliers = []
        fovs = []
        fovs_bundle = []
        for info_file in info_files:
            repr_err, num_inliers, fov, fov_bundle = PoseFinder.parseInfoFile(info_file)
            repr_errors.append(repr_err)
            inliers.append(num_inliers)
            fovs.append(fov)
            fovs_bundle.append(fov_bundle)

        if len(inliers) == 0 or len(repr_errors) == 0:
            return None, None, None, None, None, None
        repr_errors = np.array(repr_errors)
        inliers = np.array(inliers)
        fovs = np.array(fovs)
        fovs_bundle = np.array(fovs_bundle)

        if len(inliers) < 5:
            idx = np.arange(0, repr_errors.shape[0])
            thr = 60
            sel_inliers = inliers >= thr
            while (np.sum(sel_inliers) == 0 and (thr >= 11)):
                thr = thr - 10
                sel_inliers = inliers >= thr
            if np.sum(sel_inliers) > 0:
                sel_idx = idx[sel_inliers]
                err_idx = np.argmin(repr_errors[sel_inliers])
                res_idx = sel_idx[err_idx]

                res_pose_file = info_files[res_idx].replace("_matches_info.txt", "_matches_pose.txt")
                pose = np.loadtxt(res_pose_file)
                return pose, repr_errors[res_idx], inliers[res_idx], res_pose_file, fovs[res_idx], fovs_bundle[res_idx]
            return None, None, None, None, None, None

        mean_err = np.mean(repr_errors)
        mean_inliers = np.mean(inliers)
        std_err = np.std(repr_errors)
        std_inliers = np.std(inliers)

        idx = np.arange(0, repr_errors.shape[0])

        thr = 3.0
        while thr > 0:
            sel_err = repr_errors < (mean_err - thr * std_err)
            sel_inliers = inliers > (mean_inliers + thr * std_inliers)
            sel = np.logical_and(sel_err, sel_inliers)
            num_selected = np.sum(sel)
            if num_selected > 0:
                sel_err = repr_errors[sel]
                sel_inliers = inliers[sel]
                sel_idx = idx[sel]
                min_idx = np.argmin(sel_err)
                min_err = sel_err[min_idx]
                res_inliers = sel_inliers[min_idx]
                res_idx = sel_idx[min_idx]

                res_info_file = info_files[res_idx]
                res_pose_file = res_info_file.replace("_matches_info.txt", "_matches_pose.txt")
                pose = np.loadtxt(res_pose_file)
                return pose, min_err, res_inliers, res_pose_file, fovs[res_idx], fovs_bundle[res_idx]

            thr -= 0.001
        return None, None, None, None, None, None

    @staticmethod
    def refineMatchDescriptors(good, all_src_pts, all_dst_pts, desc_photo_ref, all_dst_desc_ref, radius=10, cuda=True):

        right_idx = np.arange(0, all_dst_pts.shape[0])
        new_good = []
        new_query_idxs = []
        new_train_idxs = []
        fidx = 0
        for m in good:
            idx = m.queryIdx
            lc = all_src_pts[fidx]
            rc = all_dst_pts[m.trainIdx]
            sel = np.logical_and(all_dst_pts >= (rc - radius), all_dst_pts <= (rc + radius))
            sel = np.logical_and(sel[:, 0], sel[:, 1])
            all_rc = all_dst_pts[sel]

            if np.sum(sel) > 1:
                p2_input = all_dst_desc_ref[sel]
                good_ref = Matcher.matchDescriptors(desc_photo_ref[idx].reshape(1, -1), p2_input, cuda)
                query_idxs_ref = []
                train_idxs_ref = []
                for m in good_ref:
                    query_idxs_ref.append(m.queryIdx)
                    train_idxs_ref.append(m.trainIdx)
                d1 = desc_photo_ref[idx]
                d2 = p2_input[train_idxs_ref]
                match_dist = np.linalg.norm(d1 - d2)
                match = cv2.DMatch(_distance = match_dist, _imgIdx = 0, _queryIdx = idx, _trainIdx = right_idx[sel][train_idxs_ref])
                new_good.append(match)
                new_query_idxs.append(match.queryIdx)
                new_train_idxs.append(match.trainIdx)
            elif np.sum(sel) == 1:
                # the matching was not ambiguous, no need to refine
                p2_input = all_dst_desc_ref[sel]
                match_dist = np.linalg.norm(desc_photo_ref[idx] - p2_input)
                match = cv2.DMatch(_distance = match_dist, _imgIdx = 0, _queryIdx = idx, _trainIdx = right_idx[sel])
                new_good.append(match)
                new_query_idxs.append(match.queryIdx)
                new_train_idxs.append(match.trainIdx)
            fidx += 1
        return new_good, new_query_idxs, new_train_idxs

    @staticmethod
    def refineSparseAtRes(maxres, img1, img2, img1_fov, img2_fov, kp1, kp2, good, describer, radius=10, patchsize=64, cuda=True):
        # refine matched
        coords1 = getCoordsAtSiftKeypoints(kp1)
        all_right_coords = getCoordsAtSiftKeypoints(kp2)

        print("Refining at resolution", maxres)
        patches1 = getPatchesFromImg(img1, coords1, img1_fov, maxres=maxres, patchsize=patchsize)
        patches2 = getPatchesFromImg(img2, all_right_coords, img2_fov, maxres=maxres, patchsize=patchsize)
        p1, p1_score = describer.describePhoto(patches1)
        p2, p2_score = describer.describeRender(patches2)

        left_pts_c = []
        right_pts_c = []
        new_good = []
        key_idx = 0
        right_idx = np.arange(0, len(kp2))
        for m in good:
            idx = m.queryIdx
            lc = np.flip(np.float64(kp2[m.trainIdx].pt).reshape(1, -1), 1)
            rc = np.float64(kp2[m.trainIdx].pt)
            rc = np.flip(rc, 0)
            sel = np.logical_and(all_right_coords >= (rc - radius), all_right_coords <= (rc + radius))
            sel = np.logical_and(sel[:, 0], sel[:, 1])
            all_rc = all_right_coords[sel]

            if np.sum(sel) > 1:
                p2_input = p2[sel]
                good_ref = Matcher.matchDescriptors(p1[idx].reshape(1, -1), p2_input, cuda)
                query_idxs_ref = []
                train_idxs_ref = []
                for m in good:
                    query_idxs_ref.append(m.queryIdx)
                    train_idxs_ref.append(m.trainIdx)
                left_pts_c.append(lc)
                right_pts_c.append(all_rc[train_idxs_ref])
                match_dist = np.linalg.norm(p1[idx] - p2_input[train_idxs_ref])
                match = cv2.DMatch(_distance = match_dist, _imgIdx = 0, _queryIdx = idx, _trainIdx = right_idx[sel][train_idxs_ref])
                new_good.append(match)
                key_idx += 1
            elif np.sum(sel) == 1:
                # the matching was not ambiguous, no need to refine
                p2_input = p2[sel]
                left_pts_c.append(lc)
                right_pts_c.append(rc.reshape(1, -1))
                match_dist = np.linalg.norm(p1[idx] - p2_input)
                match = cv2.DMatch(_distance = match_dist, _imgIdx = 0, _queryIdx = idx, _trainIdx = right_idx[sel])
                new_good.append(match)
                key_idx += 1


        left_pts_c = np.concatenate(left_pts_c)
        right_pts_c = np.concatenate(right_pts_c)
        print("left_pts_c", left_pts_c.shape)
        print("right_pts_c", right_pts_c.shape)
        return left_pts_c, right_pts_c, new_good, p1, p2

    @staticmethod
    def pickleKeypoints(kp, output_fname):
        point_list = []
        for point in kp:
            temp = (point.pt, point.size, point.angle, point.response,
                    point.octave, point.class_id)
            point_list.append(temp)
        with open(output_fname, 'wb') as f:
            pickle.dump(point_list, f)

    def loadKeypoints(input_fname):
        kp = []
        with open(input_fname, 'rb') as f:
            point_list = pickle.load(f)
            for temp in point_list:
                point = cv2.KeyPoint(x=temp[0][0],y=temp[0][1],_size=temp[1], _angle=temp[2],
                            _response=temp[3], _octave=temp[4], _class_id=temp[5])
                kp.append(point)
        return kp

    def getRepresentations(self, render_name, output_dir, output_dir_matching, image_base,
                           photo_img, desc_photo, kp_photo, patches_photo, fov_photo, match=False):
        print("render name", render_name)
        render_path = os.path.join(output_dir, render_name)
        base = os.path.splitext(render_name)[0]
        render_depth_path = os.path.join(output_dir, base + "_depth.txt.gz")
        MV_path = os.path.join(output_dir, base + "_pose.txt")
        P_path = os.path.join(output_dir, base + "_projection.txt")

        P = FUtil.loadMatrixFromFile(P_path)
        MV = FUtil.loadMatrixFromFile(MV_path)
        fov_render, fovy_render = FUtil.projectiveToFOV(P)

        matches_outfile = os.path.join(output_dir_matching, image_base + "-" + base + "_matches.jpg")
        patches_outfile = os.path.join(output_dir_matching, image_base + "-" + base + "_patches.jpg")

        if match:
            if not os.path.isdir(os.path.join(output_dir, "highres")):
                os.makedirs(os.path.join(output_dir, "highres"))
            dst_pts_outfile = os.path.join(output_dir, "highres", base + "_dst_pts.npy")
            dst_kp_outfile = os.path.join(output_dir, "highres", base + "_dst_kp.pickle")
            dst_patches_outfile = os.path.join(output_dir, "highres", base + "_dst_patches.npy")
            coords3d_outfile = os.path.join(output_dir, "highres", base + "_coords3d.npy")
            dst_desc_outfile = os.path.join(output_dir, "highres", base + "_dst_desc.npy")
        else:
            dst_pts_outfile = os.path.join(output_dir, base + "_dst_pts.npy")
            dst_kp_outfile = os.path.join(output_dir, base + "_dst_kp.pickle")
            dst_patches_outfile = os.path.join(output_dir, base + "_dst_patches.npy")
            coords3d_outfile = os.path.join(output_dir, base + "_coords3d.npy")
            dst_desc_outfile = os.path.join(output_dir, base + "_dst_desc.npy")

        selected = False
        if False and os.path.isfile(dst_pts_outfile) and os.path.isfile(coords3d_outfile): #not match and
            print("Loading features for render ", render_name)
            selected = True
            # we found cached version, load it
            dst_pts = np.load(dst_pts_outfile)
            kp_ren = PoseFinder.loadKeypoints(dst_kp_outfile)
            coords3d = np.load(coords3d_outfile)
            desc_ren = np.load(dst_desc_outfile)

            ren_img = cv2.imread(render_path)
            if not match:
                wp, hp, scale_p = getSizeFOV(ren_img.shape[1], ren_img.shape[0], fov_render, maxres=self.maxres)
                ren_img = cv2.resize(ren_img, (wp, hp))
            ren_img = np.flip(ren_img, 2) #flip channels as opencv treats them as BGR
            render_depth = loadDepth(render_depth_path)
            render_depth = cv2.resize(render_depth, (ren_img.shape[1], ren_img.shape[0]), interpolation=cv2.INTER_AREA)
            patches_ren = None

        else:
            print("Detecting features in render ", render_name)
            selected = True
            # we don't have a cached version, do feature detection,
            # description, and matching
            if self.use_d2net:
                render_img = self.loadImage(render_path, fov_render, scale=False)
                feat_r = KeypointDetector.describeD2Net(render_img, self, photo=False)
            elif self.fcn_keypoints:
                render_img = self.loadImage(render_path, fov_render, scale=False)
                print("render img shape", render_img.shape, match)
                feat_r = KeypointDetector.getOurKeypoints(
                    render_img, 1, self, maxres=self.maxres, photo=False,
                    fov=fov_render
                )
            elif self.fcn_keypoints_multiscale:
                render_img = self.loadImage(render_path, fov_render, scale=False)
                print("render img shape", render_img.shape, match)
                feat_r = KeypointDetector.getOurKeypoints(
                    render_img, 1, self, maxres=self.maxres, photo=False
                )
            elif self.dense_uniform_keypoints:
                render_img = self.loadImage(render_path, fov_render, scale=False)
                feat_r = KeypointDetector.getDenseRepresentationsWithKp(
                    render_img, fov_render, self.stride, self,
                    maxres=self.maxres, photo=False
                )
            elif self.dense_halton_keypoints:
                render_img = self.loadImage(render_path, fov_render, scale=False)
                feat_r = KeypointDetector.getHaltonDenseRepresentationsWithKp(
                    render_img, fov_render, self.stride, self,
                    maxres=self.maxres, photo=False
                )
            elif self.saddle_keypoints:
                render_img = self.loadImage(render_path, fov_render, scale=False)
                feat_r = KeypointDetector.detectSaddleKeypointsAndDescribe(
                    render_img, fov_render, self, photo=False
                )
            else:
                nfeatures = 5000
                if match:
                    nfeatures = 10000
                feat_r = self.detectSiftAndDescribeRender(render_path, fov_render, scale=(not match), nfeatures=nfeatures) #(not match)
            if len(feat_r[1]) == 0:
                # no features in the image
                return None, None, None, None, None
            ren_img = feat_r[0]
            kp_ren = feat_r[1]
            desc_sift_ren = feat_r[2]
            desc_ren = feat_r[3]
            patches_ren = feat_r[4]
            print("Num detected features: ", desc_ren.shape[0])
            render_depth = loadDepth(render_depth_path)
            render_depth = cv2.resize(render_depth, (ren_img.shape[1], ren_img.shape[0]), interpolation=cv2.INTER_AREA)

            dst_pts = np.float64([kp_ren[idx].pt for idx in range(0, len(kp_ren))])
            coords3d = unproject(dst_pts, render_depth, MV, P)[:, :3]
            print("desc ren shape", desc_ren.shape)

        # just to know whether the method works pairwise
        desc_ren_ref = None
        desc_photo_ref = None
        if match:
            src_pts_tmp, dst_pts_tmp, desc_photo_ref, desc_ren_ref, good = self.matching(photo_img, ren_img, desc_photo, desc_ren, kp_photo, kp_ren, patches_photo, patches_ren, fov_photo, P, MV, render_depth, patches_outfile, matches_outfile, lowe_ratio=1.1, fundam_filter=False, save_patches=False, pnp_filter=True, best_buddy=False)

            if src_pts_tmp is None:
                # pose is not valid, no need to further process these points
                return None, None, None, None, None

        return dst_pts, coords3d, desc_ren, desc_ren_ref, desc_photo_ref

    @staticmethod
    def runNCNet(img1i, img2i, model):

        # for security to fit it on GPU RAM
        maxw = 800
        if np.max(img1i.shape) > maxw:
            factor = maxw / np.max(img1i.shape)
            nw = np.floor(img1i.shape[1] * factor).astype(np.int)
            nh = np.floor(img1i.shape[0] * factor).astype(np.int)
            img1i = cv2.resize(img1i, (nw, nh))

        if np.max(img2i.shape) > maxw:
            factor = maxw / np.max(img2i.shape)
            nw = np.floor(img2i.shape[1] * factor).astype(np.int)
            nh = np.floor(img2i.shape[0] * factor).astype(np.int)
            img2i = cv2.resize(img2i, (nw, nh))

        img1i = np.expand_dims(img1i.transpose((2,0,1)),0) / 255.0
        img1i = torch.Tensor(img1i.astype(np.float32))
        img1i_var = Variable((img1i - 0.5)*2.0,requires_grad=False)

        img2i = np.expand_dims(img2i.transpose((2,0,1)),0) / 255.0
        img2i = torch.Tensor(img2i.astype(np.float32))
        img2i_var = Variable((img2i - 0.5)*2.0,requires_grad=False)

        sample = {'source_image': img1i_var, 'target_image': img2i_var, 'source_im_size': torch.tensor(np.asarray(img1i.shape).astype(np.float32)), 'target_im_size':  torch.tensor(np.asarray(img2i.shape).astype(np.float32))}
        batch_tnf = BatchTensorToVars(use_cuda=True)
        batch = batch_tnf(sample)

        corr4d = model(batch)
        xA,yA,xB,yB,sB=corr_to_matches(corr4d,do_softmax=True)

        kp1 = []
        kp2 = []
        good = []
        w1 = img1i.shape[3]
        w2 = img2i.shape[3]
        h1 = img1i.shape[2]
        h2 = img2i.shape[2]
        jidx = 0
        for idx in range(0, xA.shape[1]):
            x1 = torch.floor(((xA[0, idx] + 1.0) / 2.0) * w1)
            y1 = torch.floor(((yA[0, idx] + 1.0) / 2.0) * h1)
            x2 = torch.floor(((xB[0, idx] + 1.0) / 2.0) * w2)
            y2 = torch.floor(((yB[0, idx] + 1.0) / 2.0) * h2)

            if sB[0, idx] >= 0.5 and x1 >= 0 and x1 < w1 and x2 >= 0 and x2 < w2 and y1 >=0 and y1 <= h1 and y2 >=0 and y2 < h2:
                kp1.append(cv2.KeyPoint(x1, y1, 1))
                kp2.append(cv2.KeyPoint(x2, y2, 1))
                good.append(cv2.DMatch(jidx, jidx, sB[0, idx]))
                jidx += 1
        return kp1, kp2, good

    def findPose(self):
        start_pose = timer()
        image_basename = os.path.basename(self.query_img_path)
        image_base = os.path.splitext(image_basename)[0]
        output_dir = os.path.join(self.working_dir, image_base)
        output_dir_matching = os.path.join(self.working_dir, self.matching_dir, image_base)

        final_reproj_outfile = os.path.join(output_dir_matching, image_base + "_final_reproj.jpg")
        final_reproj_model_outfile = os.path.join(output_dir_matching, image_base + "_final_model_reproj.jpg")
        bestpose_reproj_outfile = os.path.join(output_dir_matching, image_base + "_bestpose_reproj.jpg")
        bestpose_reproj_matched_outfile = os.path.join(output_dir_matching, image_base + "_bestpose_reproj_matched.jpg")
        bestpose_reproj_model_outfile = os.path.join(output_dir_matching, image_base + "_bestpose_model_reproj.jpg")
        estimated_pose_outfile = os.path.join(output_dir_matching, image_base + "_final_pose.txt")
        estimated_bestpose_outfile = os.path.join(output_dir_matching, image_base + "_bestpose_pose.txt")
        estimated_bestpose_radial = os.path.join(output_dir_matching, image_base + "_bestpose_radial.txt")
        estimated_bestpose_kalman_ouftfile = os.path.join(output_dir_matching, image_base + "_bestpose_kalman_pose.txt")
        info_outfile = os.path.join(output_dir_matching, image_base + "_info.txt")
        bestpose_info_outfile = os.path.join(output_dir_matching, image_base + "_bestpose_info.txt")
        matched_pt2D = os.path.join(output_dir_matching, image_base + "_matched_pt2D.npy")
        reproj_pt2D = os.path.join(output_dir_matching, image_base + "_reproj_pt2D.npy")
        reproj_pt3D = os.path.join(output_dir_matching, image_base + "_reproj_pt3D.npy")
        matched_pt3D = os.path.join(output_dir_matching, image_base + "_matched_pt3D.npy")
        scene_info_file = os.path.join(output_dir, "scene_info.txt")
        output_dir_nvm = os.path.join(output_dir_matching, "nvm_export")

        if not os.path.isdir(output_dir_matching):
            os.makedirs(output_dir_matching)

        lat, lon, fov = self.getGPS()
        print("found FOV", fov)
        if not self.fov:
            photo_proj_name = os.path.splitext(self.query_img_path)[0] + "_projection.txt"
            if os.path.isfile(photo_proj_name):
                photo_P = FUtil.loadMatrixFromFile(photo_proj_name)
                fov, _ = FUtil.projectiveToFOV(photo_P)
                print("Used fov for query from projection matrix: ", fov)

        self.renderImagesAtGPS(lat, lon, output_dir)

        # detect features from photograph
        print("Detecting features in photograph")
        if self.use_d2net:
            query_img = self.loadImage(self.query_img_path, fov, scale=False)
            feat_p = KeypointDetector.describeD2Net(query_img, self, photo=True)
        elif self.fcn_keypoints:
            query_img = self.loadImage(self.query_img_path, fov, scale=False)
            feat_p = KeypointDetector.getOurKeypoints(query_img, 1, self, maxres=self.maxres, photo=True, fov=fov)
        elif self.fcn_keypoints_multiscale:
            query_img = self.loadImage(self.query_img_path, fov, scale=False)
            feat_p = KeypointDetector.getOurKeypoints(query_img, 1, self, maxres=self.maxres, photo=True)
        elif self.dense_uniform_keypoints:
            query_img = self.loadImage(self.query_img_path, fov, scale=False)
            feat_p = KeypointDetector.getDenseRepresentationsWithKp(query_img, fov, self.stride, self, maxres=self.maxres, photo=True)
        elif self.dense_halton_keypoints:
            query_img = self.loadImage(self.query_img_path, fov, scale=False)
            feat_p = KeypointDetector.getHaltonDenseRepresentationsWithKp(query_img, fov, self.stride, self, maxres=self.maxres, photo=True)
        else:
            feat_p = self.detectSiftAndDescribePhoto(self.query_img_path, fov, scale=False) #self.do_voting #scale true because of the voting
        if len(feat_p[1]) == 0:
            print("Photograph seems to be empty - no keypoints detected.")
            with open(os.path.join(output_dir_matching, "done.txt"), 'w') as f:
                f.write("Done, unable to find pose, no keypoints in the photo detected.")
            return False, None, None
        photo_img = feat_p[0]
        kp_photo = feat_p[1]
        desc_photo = feat_p[3]
        patches_photo = feat_p[4]

        images = self.getRenderedImageNames(image_base)

        if os.path.isfile(matched_pt3D) and os.path.isfile(matched_pt2D) and os.path.isfile(estimated_bestpose_outfile):
            scene_center = MultimodalPatchesDataset.getSceneCenter(scene_info_file)
            intr = FUtil.fovToIntrinsics(fov, photo_img.shape[1], photo_img.shape[0])
            if os.path.isfile(estimated_bestpose_kalman_ouftfile):
                print("exporting kalman pose to nvm")
                photo_estimated_MV = np.loadtxt(estimated_bestpose_kalman_ouftfile)
            else:
                photo_estimated_MV = np.loadtxt(estimated_bestpose_outfile)

            all_src_pts_sel = np.load(matched_pt2D)
            all_coords3d_sel = np.load(matched_pt3D)

            if (os.path.isfile(estimated_bestpose_radial)):
                radial = np.loadtxt(estimated_bestpose_radial)
                dist_coeffs = np.array([radial[0], radial[1], 0, 0, 0])
                map1, map2 = cv2.initUndistortRectifyMap(np.abs(intr), dist_coeffs, None, np.abs(intr), (photo_img.shape[1], photo_img.shape[0]), cv2.CV_32FC1)
                print("maps ", map1, map2)
                photo_img_undistort = cv2.remap(photo_img, map1, map2, cv2.INTER_AREA)
                photo_img = photo_img_undistort

            PoseFinder.exportPoseToNVM(self.query_img_path, scene_center, photo_img, intr, photo_estimated_MV, all_src_pts_sel, all_coords3d_sel, output_dir_nvm)
            photo_estimated_MV = np.loadtxt(estimated_bestpose_outfile)

            print("Export done.")
            return True, photo_estimated_MV, scene_center

        coords3d = glob.glob(os.path.join(output_dir_matching, "*_coords3d.npy"))

        all_coords3d = []
        all_dst_pts = []
        all_dst_desc = []
        all_dst_desc_ref = []  # refined descriptors
        all_image_idxs = []

        orig_refine = self.refine
        # we don't need to refine in case we don't use voting
        self.refine = False

        if self.saddle_keypoints:
            if not self.do_voting:
                # for matching we need 10k keypoints
                self.sorb = pysaddlepts.cmp_SORB(nfeatures=10000)
            else:
                # for voting 5k keypoints is enough
                self.sorb = pysaddlepts.cmp_SORB(nfeatures=5000)

        for i in range(0, len(images)):
            render_name = os.path.basename(images[i])
            rep = self.getRepresentations(render_name, output_dir,
                                          output_dir_matching, image_base,
                                          photo_img, desc_photo, kp_photo,
                                          patches_photo, fov,
                                          match=(not self.do_voting))
            if rep[0] is not None:
                all_dst_pts.append(rep[0])
                all_coords3d.append(rep[1])
                all_dst_desc.append(rep[2])
                if rep[3] is not None:
                    all_dst_desc_ref.append(rep[3])
                if not self.do_voting:
                    desc_photo_ref = rep[4]
                print("all coords and desc shape", rep[1].shape, rep[2].shape)
                image_idxs = (np.ones(rep[0].shape[0]) * i).astype(int)
                all_image_idxs.append(image_idxs)
        self.refine = orig_refine

        all_coords3d = np.concatenate(all_coords3d)
        all_dst_pts = np.concatenate(all_dst_pts)
        all_dst_desc = np.concatenate(all_dst_desc)
        all_image_idxs = np.concatenate(all_image_idxs)
        if len(all_dst_desc_ref) > 0:
            all_dst_desc_ref = np.concatenate(all_dst_desc_ref)
        else:
            all_dst_desc_ref = np.array([])

        # initial matching to guess which image to use for initial
        # pose estimation
        if self.do_voting:
            start = timer()
            print("Voting descriptor size: ", desc_photo.shape, all_dst_desc.shape)
            good_init = Matcher.matchDescriptors(desc_photo, all_dst_desc, self.cuda)
            query_idxs_init = []
            train_idxs_init = []
            for m in good_init:
                query_idxs_init.append(m.queryIdx)
                train_idxs_init.append(m.trainIdx)

            print("NN matching took: ", timer() - start)
            start = timer()
            voting = np.zeros(len(images))
            for image_idx in all_image_idxs[train_idxs_init]:
                voting[image_idx] += 1
            print("voting took: ", timer() - start)
            sel_image_idxs = np.flip(np.argsort(voting))[:self.voting_cnt]
            print("voting vec:", voting, "sel", sel_image_idxs)
            for sel_image_idx in sel_image_idxs:
                print("selected image", images[sel_image_idx])

            # reload photo at full scale
            if self.use_d2net:
                query_img = self.loadImage(self.query_img_path, fov, scale=False)
                feat_p = KeypointDetector.describeD2Net(query_img, self, photo=True)
            elif self.fcn_keypoints:
                query_img = self.loadImage(self.query_img_path, fov, scale=False)
                feat_p = KeypointDetector.getOurKeypoints(query_img, 1, self, maxres=self.maxres, photo=True, fov=fov)
            elif self.fcn_keypoints_multiscale:
                query_img = self.loadImage(self.query_img_path, fov, scale=False)
                feat_p = KeypointDetector.getOurKeypoints(query_img, 1, self, maxres=self.maxres, photo=True)
            elif self.dense_uniform_keypoints:
                query_img = self.loadImage(self.query_img_path, fov, scale=False)
                feat_p = KeypointDetector.getDenseRepresentationsWithKp(query_img, fov, self.stride, self, maxres=self.maxres, photo=True)
            elif self.dense_halton_keypoints:
                query_img = self.loadImage(self.query_img_path, fov, scale=False)
                feat_p = KeypointDetector.getHaltonDenseRepresentationsWithKp(query_img, fov, self.stride, self, maxres=self.maxres, photo=True)
            else:
                feat_p = self.detectSiftAndDescribePhoto(self.query_img_path, fov, scale=False, nfeatures=10000) #True
            if len(feat_p[1]) == 0:
                print("Photograph seems to be empty - no keypoints detected.")
                with open(os.path.join(output_dir_matching, "done.txt"), 'w') as f:
                    f.write("Done, unable to find pose, no keypoints in photo detected.")
                return False, None, None
            photo_img = feat_p[0]
            kp_photo = feat_p[1]
            desc_photo = feat_p[3]
            patches_photo = feat_p[4]

            all_coords3d = []
            all_dst_pts = []
            all_dst_desc = []
            all_dst_desc_ref = []  # refined descriptors
            desc_photo_ref = []    # refined photo descriptors
            all_image_idxs = []

            if self.saddle_keypoints:
                # set the pysaddledetector to get 10k keypoints
                self.sorb = pysaddlepts.cmp_SORB(nfeatures=10000)
            for sel_image_idx in sel_image_idxs:
                sel = all_image_idxs == sel_image_idx
                # match the winning renders to get initial pose
                render_name = os.path.basename(images[sel_image_idx])
                rep = self.getRepresentations(render_name, output_dir, output_dir_matching, image_base,
                                       photo_img, desc_photo, kp_photo, patches_photo, fov, match=True)
                if rep[0] is not None:
                    all_dst_pts.append(rep[0])
                    all_coords3d.append(rep[1])
                    all_dst_desc.append(rep[2])
                    if rep[3] is not None:
                        all_dst_desc_ref.append(rep[3])
                    desc_photo_ref = rep[4]
                    print("all coords and desc shape", rep[1].shape, rep[2].shape)
                    image_idxs = (np.ones(rep[0].shape[0]) * i).astype(int)
                    all_image_idxs.append(image_idxs)
            if len(all_coords3d) == 0:
                print("Unable to find valid pose.")
                with open(os.path.join(output_dir_matching, "done.txt"), 'w') as f:
                    f.write("Done, unable to valid find pose.")
                return False, None, None
            all_coords3d = np.concatenate(all_coords3d)
            all_dst_pts = np.concatenate(all_dst_pts)
            all_dst_desc = np.concatenate(all_dst_desc)
            all_image_idxs = np.concatenate(all_image_idxs)
            all_dst_desc_ref = np.concatenate(all_dst_desc_ref)

            colors = np.random.uniform(0, 255, (all_coords3d.shape[0], 3)).astype(np.int)
            pointcloud_outfile = os.path.join(output_dir_matching, image_base + "_pointcloud.ply")
            savePointCloudToPly(all_coords3d, colors, pointcloud_outfile)

        num_inliers = 0

        photo_estimated_MV, err, num_inliers, bp_filename, fov, fov_bundle = PoseFinder.findBestPose(output_dir_matching)
        if num_inliers is None:# or num_inliers < 30:
            print("Unable to find valid pose.")
            with open(os.path.join(output_dir_matching, "done.txt"), 'w') as f:
                f.write("Done, unable to valid find pose.")
            return False, None, None
        print("Number of inliers of best pose: ", num_inliers, " reproj error: ", err)
        print("Pose:")
        print(photo_estimated_MV)
        # we have valid pose, lets refine
        intr = FUtil.fovToIntrinsics(fov_bundle, photo_img.shape[1], photo_img.shape[0])
        p4d = np.concatenate((all_coords3d, np.ones((all_coords3d.shape[0], 1))), axis=1)

        pt_2d, z = projectWithIntrinsicsAndZ(p4d, photo_img.shape[1], photo_img.shape[0], photo_estimated_MV, intr)

        sel1 = np.logical_and(pt_2d[:, 0] > 0, pt_2d[:, 0] <= photo_img.shape[1])
        sel2 = np.logical_and(pt_2d[:, 1] > 0, pt_2d[:, 1] <= photo_img.shape[0])
        sel = np.logical_and(sel1, sel2)
        all_dst_desc = all_dst_desc[sel]
        all_dst_desc_ref = all_dst_desc_ref[sel]
        all_coords3d_nosel = all_coords3d.copy()
        all_coords3d = all_coords3d[sel]
        all_dst_pts = all_dst_pts[sel]
        print("Number of reprojected 3D points: ", np.sum(sel))

        fig = plt.figure(dpi=600)
        plt.imshow(photo_img)
        plt.scatter(pt_2d[sel, 0], pt_2d[sel, 1], c='blue', s=1.0)
        plt.savefig(bestpose_reproj_outfile)
        plt.close(fig)

        scene_center = MultimodalPatchesDataset.getSceneCenter(scene_info_file)

        if self.use_ncnet:
            # use the best pose as refined pose (no actual refinement applied)
            # since ncnet is not compatible with refinement using all keypoints
            # and descriptors.
            np.savetxt(estimated_bestpose_outfile, photo_estimated_MV)
            return True, photo_estimated_MV, scene_center


        refineBestPose = desc_photo.shape[0] > 0 and all_dst_desc.shape[0] > 0
        if (refineBestPose):
            start = timer()
            print("desc photo shape", desc_photo.shape)
            print("all dst desc shape", all_dst_desc.shape)
            good = Matcher.matchDescriptors(desc_photo, all_dst_desc, self.cuda)
            # refine matching using zoomed descriptors
            all_src_pts = np.float64([kp_photo[m.queryIdx].pt for m in good])
            good, query_idxs, train_idxs = PoseFinder.refineMatchDescriptors(good, all_src_pts, all_dst_pts, desc_photo_ref, all_dst_desc_ref, cuda=self.cuda)
            # apply best buddy matching
            if self.best_buddy_refine:
                good, train_idxs = Matcher.bestBuddy(good)

            print("NN matching took: ", timer() - start, "len good", len(good))
            all_src_pts = np.float64([kp_photo[m.queryIdx].pt for m in good])
            all_coords3d = all_coords3d[train_idxs]

            colors = np.random.uniform(0, 255, (all_coords3d.shape[0], 3)).astype(np.int)
            pointcloud_outfile = os.path.join(output_dir_matching, image_base + "_matched_pointcloud.ply")
            savePointCloudToPly(all_coords3d, colors, pointcloud_outfile)

            start = timer()
            res = False
            if self.p4pf or self.p4pf_epnp_iterative:
                res, R, t, f_, mask = poseFrom2D3DP4Pf(all_src_pts, all_coords3d, reprojection_error=8.0)
                if res:
                    # change FOV to the estimated one
                    fov = 2 * np.arctan2(photo_img.shape[1] / 2.0, f_)
                    if self.p4pf_epnp_iterative:
                        R, t, fov, mask = poseEPNPBAIterative(
                            R, t, fov, mask, photo_img.shape, all_src_pts, all_coords3d
                        )
            elif self.epnpor:
                res, R, t, mask = poseFrom2D3DWithFOVEPNPOurRansac(all_src_pts, fov, photo_img.shape, all_coords3d, reprojection_error=8.0)
            else:
                res, R, t, mask = poseFrom2D3DWithFOV(all_src_pts, fov, photo_img.shape, all_coords3d, reprojection_error=8.0, iterationsCount=100000)
            if not res:
                return False, photo_estimated_MV, scene_center
            print("POSE ESTIMATION TOOK: ", timer() - start)
            print("Final pose result: ")
            print(R)
            print(t)
            final_inliers = np.sum(mask)
            print("num inliers: ", final_inliers)
            all_coords3d = all_coords3d[mask[:, -1].astype(np.bool), :3]
            all_src_pts = all_src_pts[mask[:, -1].astype(np.bool), :2]
            photo_estimated_MV = np.zeros((4,4))
            photo_estimated_MV[3, 3] = 1
            photo_estimated_MV[:3, :3] = R
            photo_estimated_MV[:3, 3] = t

            intr = FUtil.fovToIntrinsics(fov, photo_img.shape[1], photo_img.shape[0])
            left_cam_idx = np.zeros(all_src_pts.shape[0]).astype(np.int)
            left_cam_pt_idx = np.arange(0, all_src_pts.shape[0]).astype(np.int)
            points2d = all_src_pts - np.array([[photo_img.shape[1] / 2.0, photo_img.shape[0] / 2.0]])
            R_ba, t_ba, res_intr, success = BundleAdjustment.bundleAdjustment([photo_estimated_MV], [intr], all_coords3d, points2d, left_cam_idx, left_cam_pt_idx)

            radial = np.array([0, 0])
            fov_bundle = fov
            if success:
                fov_bundle = 2 * np.arctan2(photo_img.shape[1] / 2.0, res_intr[0])
                print("old focal: ", intr[0, 0], "new focal:", res_intr[0])
                intr[0, 0] = res_intr[0] # update intrisics with adjusted focal
                intr[1, 1] = -res_intr[0]
                photo_estimated_MV[:3, :3] = R_ba
                photo_estimated_MV[:3, 3] = t_ba
                radial = res_intr[1:3]
                np.savetxt(estimated_bestpose_radial, radial)

            np.savetxt(estimated_bestpose_outfile, photo_estimated_MV)
            p4d = np.concatenate((all_coords3d, np.ones((all_coords3d.shape[0], 1))), axis=1)

            pt_2d = projectWithIntrinsics(p4d, photo_img.shape[1], photo_img.shape[0], photo_estimated_MV, intr, radial)

            sel1 = np.logical_and(pt_2d[:, 0] > 0, pt_2d[:, 0] <= photo_img.shape[1])
            sel2 = np.logical_and(pt_2d[:, 1] > 0, pt_2d[:, 1] <= photo_img.shape[0])
            sel = np.logical_and(sel1, sel2)

            # undistort the photo
            dist_coeffs = np.array([radial[0], radial[1], 0, 0, 0])
            map1, map2 = cv2.initUndistortRectifyMap(np.abs(intr), dist_coeffs, None, np.abs(intr), (photo_img.shape[1], photo_img.shape[0]), cv2.CV_32FC1)
            photo_img_undistort = cv2.remap(photo_img, map1, map2, cv2.INTER_AREA)
            photo_img = photo_img_undistort

            mean_repr_err = np.mean(np.linalg.norm(all_src_pts - pt_2d, axis=1))
            print("Mean reprojection error: ", mean_repr_err)
            with open(bestpose_info_outfile, 'w') as info:
                info.write("Mean reprojection error: " + str(mean_repr_err) + "\n")
                info.write("Num inliers: " + str(final_inliers) + "\n")
                info.write("FOV: " + str(fov) + "\n")
                info.write("FOV BA: " + str(fov_bundle) + "\n")

            fig = plt.figure(dpi=600)
            plt.imshow(photo_img)
            plt.scatter(all_src_pts[sel, 0], all_src_pts[sel, 1], c='green', s=1.0)
            plt.scatter(pt_2d[sel, 0], pt_2d[sel, 1], c='blue', s=1.0)
            plt.savefig(bestpose_reproj_matched_outfile)
            plt.close(fig)

            print("END POSE: ", timer() - start_pose)

            np.save(matched_pt2D, all_src_pts[sel])
            np.save(matched_pt3D, all_coords3d[sel])
            PoseFinder.exportPoseToNVM(self.query_img_path, scene_center, photo_img, intr, photo_estimated_MV, all_src_pts[sel], all_coords3d[sel], output_dir_nvm)

            p4d = np.concatenate((all_coords3d_nosel, np.ones((all_coords3d_nosel.shape[0], 1))), axis=1)

            pt_2d = projectWithIntrinsics(p4d, photo_img.shape[1], photo_img.shape[0], photo_estimated_MV, intr, radial)
            sel1 = np.logical_and(pt_2d[:, 0] > 0, pt_2d[:, 0] <= photo_img.shape[1])
            sel2 = np.logical_and(pt_2d[:, 1] > 0, pt_2d[:, 1] <= photo_img.shape[0])
            sel = np.logical_and(sel1, sel2)
            np.save(reproj_pt2D, pt_2d[sel])
            np.save(reproj_pt3D, all_coords3d_nosel[sel])

            return True, photo_estimated_MV, scene_center
        else:
            print("Unable to refine best pose. Not enough points to match.")

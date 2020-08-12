# @Date:   2020-08-06T16:49:36+02:00
# @Email:  ibrejcha@fit.vutbr.cz; brejchaja@gmail.com
# @Project: LandscapeAR
# @Last modified time: 2020-08-11T17:50:52+02:00
# @License: Copyright 2020 Brno University of Technology,
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
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF  FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse as ap
import torch
import cv2
import os
import exiftool
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import skimage
from sfm.Matcher import Matcher
import inspect
from time import time

# NCNet
from thirdparty.ncnet.lib.model import ImMatchNet

# D2net
from thirdparty.d2net.lib.model_test import D2Net

from pose_estimation import BundleAdjustment

# saddle keypoint detector
try:
    import pysaddlepts
    pysaddlepts_available = True
except Exception:
    pysaddlepts_available = False

# our code
from trainPatchesDescriptors import getCoordsAtSiftKeypoints, loadImageAndSift
from trainPatchesDescriptors import findLatestModelPath, plotMatches
from trainPatchesDescriptors import calculateErrors
from pose_estimation.patchSamplingDepth import generatePatchesImg
from pose_estimation.patchSamplingDepth import generatePatchesFastImg
from pose_estimation.patchSamplingDepth import loadDepth, unproject, getSizeFOV
from pose_estimation import FUtil
from pose_estimation.EstimatePose import poseFrom2D3DWithFOV, poseFrom2D3DP4Pf
from pose_estimation.EstimatePose import poseFrom2D3DWithFOVEPNPOurRansac
from pose_estimation.EstimatePose import poseEPNPBAIterative
from pose_estimation.PoseFinder import PoseFinder
from training.MultimodalPatchesDataset import MultimodalPatchesDataset
import training.Architectures as Architectures
from pose_estimation.KeypointDetector import KeypointDetector

plt.ioff()


class MatchesAnalyzer(object):

    def __init__(self, args):
        self.click_idx = 0
        self.num_clicked_pts = 4
        self.query_img_path = args.photo
        self.rendered_img_path = args.render
        self.cuda = args.cuda
        self.stride = args.stride
        self.refine = args.refine
        self.maxres = args.maxres

        self.use_depth = args.use_depth
        self.use_normals = args.use_normals
        self.use_silhouettes = args.use_silhouettes
        self.fcn_keypoints = args.fcn_keypoints
        self.fcn_keypoints_multiscale = args.fcn_keypoints_multiscale
        self.sift_keypoints = args.sift_keypoints
        self.sift_descriptors = args.sift_descriptors
        self.use_d2net = args.d2net
        self.needles = args.needles
        self.use_ncnet = args.ncnet
        self.dense_uniform_keypoints = args.dense_uniform_keypoints
        self.dense_halton_keypoints = args.dense_halton_keypoints
        self.saddle_keypoints = args.saddle_keypoints
        if (self.saddle_keypoints and not pysaddlepts_available):
            raise Exception("Module pysaddlepts not available. \
            Please, install pysaddlepts to your PYTHONPATH from: \
            https://github.com/brejchajan/saddle_detector.git")
        if self.saddle_keypoints:
            self.sorb = pysaddlepts.cmp_SORB(nfeatures=10000)

        self.p4pf = args.p4pf
        self.epnpor = args.epnpor
        self.p4pf_epnp_iterative = args.p4pf_epnp_iterative

        if not os.path.isfile(self.query_img_path):
            raise RuntimeError("Unable to find query image file.")

        self.device = torch.device("cuda:0" if args.cuda else "cpu")

        log_dir = os.path.join(args.log_dir, args.snapshot)

        models_path = os.path.join(log_dir, "models")
        if args.model_name:
            model_path = os.path.join(models_path, args.model_name)
            model_name = os.path.basename(model_path).split("_epoch_")[0]
        else:
            model_path = findLatestModelPath(models_path)
            model_name = os.path.basename(model_path).split("_epoch_")[0]

        # Our detector
        module = __import__("training").Architectures
        net_class = getattr(module, model_name)

        if self.fcn_keypoints or self.fcn_keypoints_multiscale:
            if  net_class.__name__ != "MultimodalKeypointPatchNet5lShared2lFCN":
                print("loading FCN variant")
                self.net = Architectures.MultimodalPatchNet5lShared2lFCN(args.normalize_output)
                self.net_keypoints = Architectures.MultimodalPatchNet5lShared2lFCN(args.normalize_output,
                strides=[1], strides_branch=[2, 1, 1, 1],
                dilations=[2], dilations_branch=[1, 1, 2, 2])
            else:
                self.net = Architectures.MultimodalKeypointPatchNet5lShared2lFCN(args.normalize_output, pretrained=False)
                self.net_keypoints = Architectures.MultimodalKeypointPatchNet5lShared2lFCN(
                    args.normalize_output,
                    strides=[1], strides_branch=[2, 1, 1, 1],
                    dilations=[2], dilations_branch=[1, 1, 2, 2],
                    pretrained=False
                )
        else:
            cls_args = inspect.getfullargspec(net_class.__init__).args
            if not 'needles' in cls_args:
                self.needles = 0
                self.net = net_class(args.normalize_output)
            else:
                self.net = net_class(args.normalize_output, needles=self.needles)
        self.loadModel(model_path)
        if args.cuda:
            self.net = self.net.to(self.device)
            if self.fcn_keypoints or self.fcn_keypoints_multiscale:
                self.net_keypoints = self.net_keypoints.to(self.device)
        self.net.eval()
        if self.fcn_keypoints or self.fcn_keypoints_multiscale:
            self.net_keypoints.eval()


        self.keypoints_heatmap = args.keypoints_heatmap
        self.has_keypoints = False
        if isinstance(self.net, Architectures.MultimodalKeypointPatchNet5lShared2l) or isinstance(self.net, Architectures.MultimodalKeypointPatchNet5lShared2lFCN):
            self.has_keypoints = True

        # D2net
        if self.use_d2net:
            self.d2net = D2Net(
                model_file='pretrained/d2_net/d2_tf.pth',
                use_relu=True,
                use_cuda=args.cuda
            )

        if self.use_ncnet:
            self.ncnet = ImMatchNet(use_cuda=True, checkpoint='pretrained/ncnet/ncnet_ivd.pth.tar')

    def loadModel(self, model_path):
        print("Loading model: ", model_path)
        checkpoint = torch.load(model_path, map_location=self.device)

        if self.fcn_keypoints or self.fcn_keypoints_multiscale:
            x = checkpoint['model_state_dict']['conv2.weight']
            checkpoint['model_state_dict']['conv2.weight'] = x.reshape(x.shape[0], x.shape[1], 1, 1)
            self.net_keypoints.load_state_dict(checkpoint['model_state_dict'])
        self.net.load_state_dict(checkpoint['model_state_dict'])

    def describePhoto(self, patches1, batchsize=100):
        p1 = []
        if self.has_keypoints:
            p1_scores = []
        else:
            p1_scores = np.ones(patches1.shape[0])
        start = time()
        for idx in range(0, patches1.shape[0], batchsize):
            batch = patches1[idx:idx + batchsize].to(self.device)
            batch = torch.nn.functional.interpolate(batch, (64, 64), mode='area')
            p1_fea = self.net.forward_photo(batch)
            if self.has_keypoints:
                p1_score = p1_fea[1].detach().cpu().numpy()
                p1_scores.append(p1_score)
                p1_fea = p1_fea[0]
            p1_fea = p1_fea.detach().cpu().numpy()
            p1.append(p1_fea)
        end = time()
        print("Description of ", patches1.shape[0], "patches took ", (end - start), "seconds.")
        p1 = np.concatenate(p1)
        p1 = p1.reshape(p1.shape[0], p1.shape[1])
        if self.has_keypoints:
            p1_scores = np.concatenate(p1_scores)
        return p1, p1_scores

    def describeRender(self, patches1, batchsize=100):
        p1 = []
        if self.has_keypoints:
            p1_scores = []
        else:
            p1_scores = np.ones(patches1.shape[0])
        for idx in range(0, patches1.shape[0], batchsize):
            batch = patches1[idx:idx + batchsize].to(self.device)
            batch = torch.nn.functional.interpolate(batch, (64, 64), mode='area')
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

    def createAndShowPhotoHeatmap(self, event):
        if event.inaxes:
            ax_idx = event.inaxes.get_geometry()[2]
            if ax_idx == 1:
                dist = self.photo_fea_scores #np.linalg.norm(self.render_fea - photo_fea, axis=1)
                norm = matplotlib.colors.Normalize()
                colors = plt.cm.jet(norm(dist))

                orig_dist_img = dist.reshape(self.photo_nw, self.photo_nh).transpose()
                fea1 = KeypointDetector.getOurKeypoints(self.photo, self.stride, self, self.maxres, photo=True)
                kp1 = fea1[1]
                coords = getCoordsAtSiftKeypoints(kp1)

                dist_img = colors.reshape(self.photo_nw, self.photo_nh, 4).transpose(1, 0, 2)
                dist_img = cv2.resize(dist_img, (self.photo.shape[1], self.photo.shape[0]), interpolation=cv2.INTER_AREA)
                orig_dist_img = cv2.resize(orig_dist_img, (self.photo.shape[1], self.photo.shape[0]), interpolation=cv2.INTER_AREA)

                self.photo_keypoints.set_offsets(np.flip(coords, axis=1))

                self.photo_heatmap.set_data(dist_img)
                self.fig.canvas.draw()

    def createAndShowHeatmap(self, event):
        if event.inaxes:
            ax_idx = event.inaxes.get_geometry()[2]
            if ax_idx == 1:
                # user clicked to the photo (which is the first axis)
                photo_coords = np.array([[event.ydata, event.xdata]])
                photo_patches = generatePatchesImg(self.photo, self.photo.shape, photo_coords, self.photo_fov, show=False, maxres=self.maxres)
                photo_patches = np.ascontiguousarray(np.asarray(photo_patches).transpose((0,3,1,2))).astype(np.float32) / 255.0
                photo_patches = torch.from_numpy(photo_patches)
                photo_fea, photo_score = self.describePhoto(photo_patches)

                if self.keypoints_heatmap:
                    dist = self.render_fea_scores
                else:
                    dist = np.linalg.norm(self.render_fea - photo_fea, axis=1)
                norm = matplotlib.colors.Normalize()
                colors = plt.cm.jet(norm(dist))

                orig_dist_img = dist.reshape(self.render_nw, self.render_nh).transpose()
                if self.keypoints_heatmap:
                    fea1 = KeypointDetector.getOurKeypoints(self.render, self.stride, self, self.maxres, photo=False)
                    kp1 = fea1[1]
                    coords = getCoordsAtSiftKeypoints(kp1)
                    self.render_keypoints.set_offsets(np.flip(coords, axis=1))

                dist_img = colors.reshape(self.render_nw, self.render_nh, 4).transpose(1, 0, 2)
                dist_img = cv2.resize(dist_img, (self.render.shape[1], self.render.shape[0]), interpolation=cv2.INTER_AREA)
                orig_dist_img = cv2.resize(orig_dist_img, (self.render.shape[1], self.render.shape[0]), interpolation=cv2.INTER_AREA)

                self.heatmap.set_data(dist_img)
                minpos = np.unravel_index(np.argmin(orig_dist_img), dims=orig_dist_img.shape)
                self.min_marker.set_offsets(np.array([minpos[1], minpos[0]]))
                self.click_marker.set_offsets(np.array([event.xdata, event.ydata]))
                self.fig.canvas.draw()

    @staticmethod
    def loadImage(img_path, fov=65.0):
        photo = cv2.imread(img_path)
        img_path = img_path.replace("_texture", "")
        photo_proj_name = os.path.splitext(img_path)[0] + "_projection.txt"
        if not os.path.isfile(photo_proj_name):
            fov_found = False
            with exiftool.ExifTool() as et:
                res = et.execute_json("-n", "-FOV", img_path)
                if 'Composite:FOV' in res[0]:
                    photo_fov = res[0]['Composite:FOV']
                    print("Found FOV in photo exif", photo_fov)
                    photo_fov = (photo_fov / 180.0) * np.pi
                    fov_found = True
                    photo_P = None
            if not fov_found:
                print("using defalt FOV", fov)
                photo_P = None
                photo_fov = (fov / 180.0) * np.pi
        else:
            print("loading intrinsics from file", photo_proj_name)
            photo_P = FUtil.loadMatrixFromFile(photo_proj_name)
            photo_fov, _ = FUtil.projectiveToFOV(photo_P)

        photo = np.flip(photo, 2)
        return photo, photo_fov, photo_P

    @staticmethod
    def getDenseRepresentations(img, fov, stride, describer,
                                maxres=2048, photo=False):
        # get patches from render and describe them
        rc_y = np.arange(0, img.shape[0], stride)
        nh = rc_y.shape[0]
        rc_x = np.arange(0, img.shape[1], stride)
        nw = rc_x.shape[0]
        rc_yv, rc_xv = np.meshgrid(rc_y, rc_x)
        render_coords = np.array([rc_yv.reshape(-1), rc_xv.reshape(-1)]).transpose()

        numpatches_batch = 5000
        render_fea = []
        render_fea_scores = []
        for idx in tqdm(range(0, render_coords.shape[0], numpatches_batch)):
            render_patches = generatePatchesImg(img, img.shape, render_coords[idx:idx + numpatches_batch], fov, show=False, maxres=maxres)
            render_patches = (np.asarray(render_patches)).transpose((0, 3, 1, 2)).astype(np.float32)
            render_patches[:, :3, :, :] = render_patches[:, :3, :, :] / 255.0
            render_patches = torch.from_numpy(render_patches)
            if photo:
                fea, fea_score = describer.describePhoto(render_patches)
            else:
                fea, fea_score = describer.describeRender(render_patches)
            render_fea_scores.append(fea_score)
            render_fea.append(fea)
        render_fea = np.concatenate(render_fea)
        render_fea_scores = np.concatenate(render_fea_scores)
        return render_fea, render_fea_scores, nw, nh

    @staticmethod
    def getDenseRepresentationsWithKp(img, fov, stride, describer,
                                      maxres=3000, photo=False):
        # resize the image so that it matches FOV scale so that we don't
        # extract more features than needed

        if not photo:
            stride = int(stride / np.sqrt(2.0))

        wp, hp, scale_p = getSizeFOV(img.shape[1], img.shape[0], fov, maxres=maxres)
        scale = img.shape[1] / wp
        img = skimage.transform.resize(img, (hp, wp))

        # get patches from render and describe them
        rc_y = np.arange(32, img.shape[0] - 32, stride)
        nh = rc_y.shape[0]
        rc_x = np.arange(32, img.shape[1] - 32, stride)
        nw = rc_x.shape[0]
        rc_yv, rc_xv = np.meshgrid(rc_y, rc_x)
        render_coords = np.array([rc_yv.reshape(-1), rc_xv.reshape(-1)]).transpose()

        # skip points in sky of the rendered image
        img_sel = img[render_coords[:, 0], render_coords[:, 1]]
        sky_sel = (np.all((img_sel != 0), axis=1)).reshape(-1)
        render_coords = render_coords[sky_sel]

        numpatches_batch = 5000
        render_fea = []
        render_fea_scores = []
        all_render_patches = []
        for idx in tqdm(range(0, render_coords.shape[0], numpatches_batch)):
            render_patches = generatePatchesImg(img, img.shape, render_coords[idx:idx + numpatches_batch], fov, show=False, maxres=maxres)
            render_patches = (np.asarray(render_patches)).transpose((0, 3, 1, 2)).astype(np.float32)
            render_patches[:, :3, :, :] = render_patches[:, :3, :, :] / 255.0
            render_patches = torch.from_numpy(render_patches)
            all_render_patches.append(render_patches)
            if photo:
                fea, fea_score = describer.describePhoto(render_patches)
            else:
                fea, fea_score = describer.describeRender(render_patches)
            render_fea.append(fea)
            render_fea_scores.append(fea_score)
        render_fea = np.concatenate(render_fea)
        render_fea_scores = np.concatenate(render_fea_scores)
        all_render_patches = np.concatenate(all_render_patches)
        kp = []
        for idx in range(0, render_coords.shape[0]):
            kp.append(cv2.KeyPoint(render_coords[idx, 1] * scale, render_coords[idx, 0] * scale, 1))

        return img, kp, None, render_fea, all_render_patches, nw, nh

    def analyzeDistance(self):
        self.photo, self.photo_fov, _ = MatchesAnalyzer.loadImage(self.query_img_path)
        maxw = 2048
        if np.max(self.photo.shape) > maxw:
            scale = maxw / np.max(self.photo.shape)
            self.photo = cv2.resize(self.photo, (int(self.photo.shape[1] * scale), int(self.photo.shape[0] * scale)), interpolation=cv2.INTER_AREA)

        self.render, self.render_fov, _ = MatchesAnalyzer.loadImage(self.rendered_img_path)
        if np.max(self.render.shape) > maxw:
            scale = maxw / np.max(self.render.shape)
            self.render = cv2.resize(self.render, (int(self.render.shape[1] * scale), int(self.render.shape[0] * scale)), interpolation=cv2.INTER_AREA)

        rendered_depth_path = os.path.splitext(self.rendered_img_path)[0] + "_depth.txt.gz"
        self.render_depth = loadDepth(rendered_depth_path)
        self.render_depth = cv2.resize(self.render_depth, (self.render.shape[1], self.render.shape[0]), interpolation=cv2.INTER_AREA)

        self.render = self.loadAdditionalChannels(self.render, self.render_depth)
        if self.keypoints_heatmap and self.fcn_keypoints:
            self.render_fea_scores = KeypointDetector.getKeypointProbaMap(self.render, self, False).transpose().copy()
            self.render_nw = self.render_fea_scores.shape[0]
            self.render_nh = self.render_fea_scores.shape[1]
            self.render_fea_scores = self.render_fea_scores.reshape(-1)
        else:
            self.render_fea, self.render_fea_scores, self.render_nw, self.render_nh = MatchesAnalyzer.getDenseRepresentations(self.render, self.render_fov, self.stride, self, maxres=self.maxres, photo=False)

        if self.keypoints_heatmap and self.fcn_keypoints:
            self.photo_fea_scores = KeypointDetector.getKeypointProbaMap(self.photo, self, True).transpose().copy()
            self.photo_nw = self.photo_fea_scores.shape[0]
            self.photo_nh = self.photo_fea_scores.shape[1]
            self.photo_fea_scores = self.photo_fea_scores.reshape(-1)
        else:
            self.photo_fea, self.photo_fea_scores, self.photo_nw, self.photo_nh = MatchesAnalyzer.getDenseRepresentations(self.photo, self.photo_fov, self.stride, self, maxres=self.maxres, photo=True)

        print("scores shape", self.render_fea_scores.shape)
        self.fig = plt.figure()
        ax1 = plt.subplot(1, 2, 1)
        ax1.imshow(self.photo)
        if self.keypoints_heatmap:
            self.photo_heatmap = ax1.imshow(np.zeros(self.photo[:, :, :3].shape), alpha=0.3)
            self.photo_keypoints = ax1.scatter(0, 0, alpha=1.0, color='blue', s=0.5)
        self.click_marker = ax1.scatter(0, 0, alpha=1.0, color='red', marker='x')
        plt.axis("off")
        ax2 = plt.subplot(1, 2, 2)
        ax2.imshow(self.render[:, :, :3].astype(np.int))
        self.heatmap = ax2.imshow(np.zeros(self.render[:, :, :3].shape), alpha=0.3)
        if self.keypoints_heatmap:
            self.render_keypoints = ax2.scatter(0, 0, alpha=1.0, color='blue', s=0.5)
        self.min_marker = ax2.scatter(0, 0, alpha=1.0, color='red', marker='x')
        plt.axis("off")

        cid = self.fig.canvas.mpl_connect('button_press_event', self.createAndShowHeatmap)
        if self.keypoints_heatmap:
            cid = self.fig.canvas.mpl_connect('button_press_event', self.createAndShowPhotoHeatmap)

        plt.show()

    def refineMatchAtRes(self, maxres, left_coords, right_coords, left_img, right_img, left_fov, right_fov):
        radius = 32
        photo_patches = generatePatchesImg(left_img, left_img.shape, left_coords, left_fov, show=False, maxres=maxres)
        photo_patches = np.ascontiguousarray(np.asarray(photo_patches).transpose((0,3,1,2))).astype(np.float32) / 255.0
        photo_patches = torch.from_numpy(photo_patches)
        photo_fea, photo_score = self.describePhoto(photo_patches)

        wp, hp, scale_p = getSizeFOV(right_img.shape[1], right_img.shape[0], right_fov, maxres=maxres)

        right_coords = right_coords * scale_p
        right_img = cv2.resize(right_img, (wp, hp), interpolation=cv2.INTER_AREA)
        from_y = int(right_coords[0])
        to_y = int(right_coords[0] + 2.0 * radius)
        from_x = int(right_coords[1])
        to_x = int(right_coords[1] + 2.0 * radius)
        right_img_pad = np.pad(right_img, ((radius, radius), (radius, radius), (0, 0)), 'constant')
        right_img_part = np.ascontiguousarray(np.copy(right_img_pad[from_y:to_y, from_x:to_x]))
        right_fov_part = (right_img_part.shape[1] / wp) * right_fov
        fea, fea_scores, nw, nh = MatchesAnalyzer.getDenseRepresentations(right_img_part, right_fov_part, 1, self, maxres=maxres) #self.stride

        dist = np.linalg.norm(fea - photo_fea, axis=1)
        norm = matplotlib.colors.Normalize()

        orig_dist_img = dist.reshape(nw, nh).transpose()
        orig_dist_img = cv2.resize(orig_dist_img, (right_img_part.shape[1], right_img_part.shape[0]), interpolation=cv2.INTER_AREA)

        colors = plt.cm.jet(norm(dist))
        dist_img = colors.reshape(nw, nh, 4).transpose(1, 0, 2)
        dist_img = cv2.resize(dist_img, (right_img_part.shape[1], right_img_part.shape[0]), interpolation=cv2.INTER_AREA)

        minpos = np.unravel_index(np.argmin(orig_dist_img), dims=orig_dist_img.shape)
        refined_pos = (np.array(minpos) + np.array([from_y - radius, from_x - radius])) / scale_p
        plt.figure()
        plt.imshow(right_img_part)
        plt.imshow(dist_img, alpha=0.3)
        plt.scatter(minpos[1], minpos[0], alpha=1.0, color='red', marker='x')
        plt.axis("off")
        plt.show()
        return refined_pos


    def handleClickMatch(self, event):
        if event.inaxes:
            ax_idx = event.inaxes.get_geometry()[2]
            if ax_idx == 1:
                self.left_pts_coords[self.click_idx, 0] = np.floor(event.xdata)
                self.left_pts_coords[self.click_idx, 1] = np.floor(event.ydata)
                self.left_click_markers.set_offsets(self.left_pts_coords)

                if True:
                    photo_coords = np.array([[event.ydata, event.xdata]])
                    photo_patches = generatePatchesImg(self.photo, self.photo.shape, photo_coords, self.photo_fov, show=False, maxres=self.maxres)
                    photo_patches = np.ascontiguousarray(np.asarray(photo_patches).transpose((0,3,1,2))).astype(np.float32) / 255.0
                    photo_patches = torch.from_numpy(photo_patches)
                    photo_fea, photo_score = self.describePhoto(photo_patches)

                    dist = np.linalg.norm(self.render_fea - photo_fea, axis=1)
                    norm = matplotlib.colors.Normalize()

                    orig_dist_img = dist.reshape(self.render_nw, self.render_nh).transpose()
                    orig_dist_img = cv2.resize(orig_dist_img, (self.render.shape[1], self.render.shape[0]), interpolation=cv2.INTER_AREA)

                    minpos = np.unravel_index(np.argmin(orig_dist_img), dims=orig_dist_img.shape)
                    self.right_pts_coords[self.click_idx, 0] = minpos[1]
                    self.right_pts_coords[self.click_idx, 1] = minpos[0]
                    self.right_click_markers.set_offsets(self.right_pts_coords)
                else:
                    self.right_pts_coords = np.copy(self.left_pts_coords)
                    self.right_click_markers.set_offsets(self.right_pts_coords)
            elif ax_idx == 2:
                self.right_pts_coords[self.click_idx, 0] = np.floor(event.xdata)
                self.right_pts_coords[self.click_idx, 1] = np.floor(event.ydata)
                self.right_click_markers.set_offsets(self.right_pts_coords)
            self.fig.canvas.draw()

    def handleKeypressMatch(self, event):
        if event.key:
            if event.key.isdigit():
                key = int(event.key)
                if key >= 0 and key < self.num_clicked_pts:
                    self.click_idx = key
                    self.matches_label.set_text('Click correspondence: ' + str(self.click_idx))
                    self.fig.canvas.draw()
            elif event.key == "p" or event.key == "P":
                self.estimatePose()
            elif event.key == "m" or event.key == "M":
                self.matching()
                self.left_click_markers.set_offsets(self.left_pts_coords)
                self.right_click_markers.set_offsets(self.right_pts_coords)
                self.fig.canvas.draw()
            elif event.key == "r" or event.key == "R":
                self.left_pts_coords = np.zeros((self.num_clicked_pts, 2))
                self.right_pts_coords = np.zeros((self.num_clicked_pts, 2))
                self.left_click_markers.set_offsets(self.left_pts_coords)
                self.right_click_markers.set_offsets(self.right_pts_coords)
                self.fig.canvas.draw()


    def estimatePose(self, reprojection_error=8):
        coords3d = np.ascontiguousarray(unproject(self.right_pts_coords, self.render_depth, self.render_MV, self.render_P)[:, :3])
        coords2d = np.ascontiguousarray(self.left_pts_coords[:,:2]).reshape((self.left_pts_coords.shape[0],1,2))

        if self.p4pf or self.p4pf_epnp_iterative:
            ret, R, t, f, mask = poseFrom2D3DP4Pf(coords2d[:, 0], coords3d, reprojection_error)
            if ret:
                estim_FOV = 2 * np.arctan2(self.photo.shape[1] / 2.0, f)
                print("orig FOV", (self.photo_fov * 180.0) / np.pi, "estimated FOV p4pf", (estim_FOV * 180.0) / np.pi, "p4pf focal", f, "num inliers", np.sum(mask))
                self.photo_fov = estim_FOV
                if self.p4pf_epnp_iterative:
                    R, t, estim_FOV, mask = poseEPNPBAIterative(R, t, estim_FOV, mask, self.photo.shape, coords2d[:, 0], coords3d, reprojection_error)
                    self.photo_fov = estim_FOV

        elif self.epnpor:
            ret, R, t, mask = poseFrom2D3DWithFOVEPNPOurRansac(coords2d[:, 0], self.photo_fov, self.photo.shape, coords3d, reprojection_error)
        else:
            ret, R, t, mask = poseFrom2D3DWithFOV(coords2d, self.photo_fov, self.photo.shape, coords3d, use_ransac=True, pnp_flags=cv2.SOLVEPNP_EPNP, reprojection_error=reprojection_error)
        if not ret:
            return None, None, None
        print("Initial pose result: ")
        print(R)
        print(t)
        R_gt = self.photo_MV[:3, :3]
        t_gt = self.photo_MV[:3, 3]
        o_err, t_err = calculateErrors(R_gt, t_gt, R, t)
        print("Orientation error: ", o_err, ", translation err: ", t_err, R_gt)
        print("num inliers: ", np.sum(mask))

        coords2d = np.ascontiguousarray(np.copy(coords2d[mask[:, 0] == 1]))
        coords3d = np.ascontiguousarray(np.copy(coords3d[mask[:, 0] == 1]))
        print("coords2d in", coords2d.shape)
        print("coords3d in", coords3d.shape)
        mask1 = mask

        scene_info_filepath = os.path.join(os.path.dirname(self.rendered_img_path), 'scene_info.txt')
        if (os.path.exists(scene_info_filepath)):
            print("exporting to nvm file")
            scene_center = MultimodalPatchesDataset.getSceneCenter(scene_info_filepath)
            intr1 = FUtil.fovToIntrinsics(self.photo_fov, self.photo.shape[1], self.photo.shape[0])
            pose = np.ones([4, 4])
            pose[:3, :3] = R
            pose[:3, 3] = t
            output_path = "nvm_export_" + os.path.splitext(os.path.basename(self.query_img_path))[0]
            PoseFinder.exportPoseToNVM(self.query_img_path, scene_center, self.photo, intr1, pose, coords2d[:, 0], coords3d, output_path)

        return R, t, mask1

    def detectSiftAndDescribePhoto(self, photo_name, fov,
                                   contrastThr=0.00, edgeThr=10, sigma=1.0, shape=None):

        img1, kp1, ds1_sift = loadImageAndSift(photo_name, contrastThr,
                                               edgeThr, sigma, nfeatures=10000, shape=shape)
        coords1 = getCoordsAtSiftKeypoints(kp1)
        patches1 = generatePatchesFastImg(img1, img1.shape, coords1, fov, maxres=self.maxres, needles=self.needles)
        patches1 = torch.from_numpy(patches1)

        if self.sift_descriptors:
            p1 = ds1_sift
        else:
            p1, p1_score = self.describePhoto(patches1)

        return img1, kp1, ds1_sift, p1, patches1

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

    def detectSiftAndDescribeRender(self, render_name, fov,
                                   contrastThr=0.02, edgeThr=15, sigma=1.0, shape=None):

        img1, kp1, ds1_sift = loadImageAndSift(render_name, contrastThr,
                                               edgeThr, sigma, nfeatures=10000, shape=shape)
        coords1 = getCoordsAtSiftKeypoints(kp1)

        depth_path = os.path.splitext(render_name)[0] + "_depth.txt.gz"
        depth = loadDepth(depth_path)
        depth = cv2.resize(self.render_depth, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_AREA)
        img1 = self.loadAdditionalChannels(img1, depth)

        coords1 = getCoordsAtSiftKeypoints(kp1)
        patches1 = generatePatchesFastImg(img1, img1.shape, coords1, fov, maxres=self.maxres, needles=self.needles)
        print(np.max(patches1))
        patches1 = torch.from_numpy(patches1)

        if self.sift_descriptors:
            p1 = ds1_sift
        else:
            p1, p1_score = self.describeRender(patches1)

        return img1, kp1, ds1_sift, p1, patches1

    def matching(self):
        photo_shp = (self.photo.shape[1], self.photo.shape[0])
        render_shp = (self.render.shape[1], self.render.shape[0])
        if self.use_d2net:
            fea_p = KeypointDetector.loadImageAndD2Net(self.query_img_path, photo_shp, self, photo=True)
            fea_r = KeypointDetector.loadImageAndD2Net(self.rendered_img_path, render_shp, self, photo=False)
        elif self.sift_keypoints:
            fea_p = self.detectSiftAndDescribePhoto(self.query_img_path, self.photo_fov, shape=photo_shp)
            fea_r = self.detectSiftAndDescribeRender(self.rendered_img_path, self.render_fov, shape=render_shp)
        elif self.use_ncnet:
            # ncnet does not give desciptors to be matched, it returns
            # matches
            pass
        elif self.fcn_keypoints:
            fea_p = KeypointDetector.getOurKeypoints(self.photo, self.stride, self, photo=True, fov=self.photo_fov, maxres=self.maxres)
            fea_r = KeypointDetector.getOurKeypoints(self.render, self.stride, self, photo=False, fov=self.render_fov, maxres=self.maxres)
        elif self.fcn_keypoints_multiscale:
            fea_p = KeypointDetector.getOurKeypoints(self.photo, self.stride, self, photo=True, maxres=self.maxres)
            fea_r = KeypointDetector.getOurKeypoints(self.render, self.stride, self, photo=False, maxres=self.maxres)
        elif self.dense_halton_keypoints:
            fea_p = KeypointDetector.getHaltonDenseRepresentationsWithKp(self.photo, self.photo_fov, self.stride, self, self.maxres, photo=True)
            fea_r = KeypointDetector.getHaltonDenseRepresentationsWithKp(self.render, self.render_fov, self.stride, self, self.maxres, photo=False)
        elif self.saddle_keypoints:
            fea_p = KeypointDetector.detectSaddleKeypointsAndDescribe(self.photo, self.photo_fov, self, photo=True)
            fea_r = KeypointDetector.detectSaddleKeypointsAndDescribe(self.render, self.render_fov, self, photo=False)
        else:
            # self.dense_uniform_keypoints == True
            fea_p = KeypointDetector.getDenseRepresentationsWithKp(self.photo, self.photo_fov, self.stride, self, self.maxres, photo=True)
            fea_r = KeypointDetector.getDenseRepresentationsWithKp(self.render, self.render_fov, self.stride, self, self.maxres, photo=False)


        if not self.use_ncnet:
            print("number of features photo", fea_p[3].shape[0])
            print("number of features render", fea_r[3].shape[0])

            good = Matcher.matchDescriptors(fea_p[3], fea_r[3], self.cuda)
            print("descriptors matched, estimating pose. Number of matches: ", len(good))

            kp1 = fea_p[1]
            kp2 = fea_r[1]
        else:
            kp1, kp2, good = PoseFinder.runNCNet(self.photo, self.render, self.ncnet)

        self.left_pts_coords = np.float64([kp1[m.queryIdx].pt for m in good])
        self.right_pts_coords = np.float64([kp2[m.trainIdx].pt for m in good])

        kp_p = np.float64([kp.pt for kp in kp1])
        kp_r = np.float64([kp.pt for kp in kp2])

        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(self.photo)
        plt.scatter(kp_p[:, 0], kp_p[:, 1], s=0.5)
        plt.subplot(1, 2, 2)
        plt.imshow(self.render)
        plt.scatter(kp_r[:, 0], kp_r[:, 1], s=0.5)

        plotMatches(self.photo, kp1, self.render[:, :, :3], kp2, good, None, None, show=False)
        R, t, mask = self.estimatePose()
        if R is None:
            print("Unable to estimate valid pose.")
            return
        plotMatches(self.photo, kp1, self.render[:, :, :3], kp2, good, None, mask, show=False)


        if self.refine:
            coords1_ref, coords2_ref, new_good, _src_desc, _dst_desc = PoseFinder.refineSparseAtRes(2.0 * self.maxres, self.photo, self.render, self.photo_fov, self.render_fov, kp1, kp2, good, self, radius=10)
            coords1_ref, coords2_ref, new_good, _src_desc, _dst_desc = PoseFinder.refineSparseAtRes(4.0 * self.maxres, self.photo, self.render, self.photo_fov, self.render_fov, kp1, kp2, new_good, self, radius=5)

            self.left_pts_coords = np.float64([kp1[m.queryIdx].pt for m in new_good])
            self.right_pts_coords = np.float64([kp2[m.trainIdx].pt for m in new_good])

            R, t, mask = self.estimatePose(reprojection_error=10)
            self.left_pts_coords = self.left_pts_coords[mask[:, 0] == 1]
            self.right_pts_coords = self.right_pts_coords[mask[:, 0] == 1]

            plotMatches(self.photo, kp1, self.render[:, :, :3], kp2, new_good, None, mask, show=False)
        else:
            self.left_pts_coords = self.left_pts_coords[mask[:, 0] == 1]
            self.right_pts_coords = self.right_pts_coords[mask[:, 0] == 1]
        pose = np.ones([4, 4])
        pose[:3, :3] = R
        pose[:3, 3] = t
        intr1 = FUtil.fovToIntrinsics(self.photo_fov, self.photo.shape[1], self.photo.shape[0])
        coords3d = np.ascontiguousarray(unproject(self.right_pts_coords, self.render_depth, self.render_MV, self.render_P)[:, :3])
        left_cam_idx = np.zeros(self.left_pts_coords.shape[0]).astype(np.int)
        left_cam_pt_idx = np.arange(0, self.left_pts_coords.shape[0]).astype(np.int)
        points2d = self.left_pts_coords - np.array([[self.photo.shape[1] / 2.0, self.photo.shape[0] / 2.0]])
        res_R, res_t, res_intr, success = BundleAdjustment.bundleAdjustment([pose], [intr1], coords3d, points2d, left_cam_idx, left_cam_pt_idx, show=True)

        if success:
            print("BA SUCESS")
            R_gt = self.photo_MV[:3, :3]
            t_gt = self.photo_MV[:3, 3]
            o_err, t_err = calculateErrors(R_gt, t_gt, res_R, res_t)
            print("Orientation error: ", o_err, ", translation err: ", t_err)
            print("Result Rt")
            print(res_R)
            print(res_t)
            print("result intr params", res_intr)
            orig_FOV = 2 * np.arctan2(self.photo.shape[1] / 2.0, intr1[0, 0])
            bundled_FOV = 2 * np.arctan2(self.photo.shape[1] / 2.0, res_intr[0])
            print("orig fov", (orig_FOV * 180.0) / np.pi, "bundled fov", (bundled_FOV * 180.0) / np.pi)
            print("GT Rt")
            print(R_gt)
            print(t_gt)

            scene_info_filepath = os.path.join(os.path.dirname(self.rendered_img_path), 'scene_info.txt')
            if (os.path.exists(scene_info_filepath)):
                print("exporting to nvm file")
                scene_center = MultimodalPatchesDataset.getSceneCenter(scene_info_filepath)
                intr1 = FUtil.fovToIntrinsics(self.photo_fov, self.photo.shape[1], self.photo.shape[0])
                intr1[0, 0] = res_intr[0] # update intrisics with adjusted focal
                intr1[1, 1] = -res_intr[0]
                radial = res_intr[1:3]

                # undistort the photo
                dist_coeffs = np.array([radial[0], radial[1], 0, 0, 0])
                map1, map2 = cv2.initUndistortRectifyMap(np.abs(intr1), dist_coeffs, None, np.abs(intr1), (self.photo.shape[1], self.photo.shape[0]), cv2.CV_32FC1)
                photo_img_undistort = cv2.remap(self.photo, map1, map2, cv2.INTER_AREA)

                pose = np.ones([4, 4])
                pose[:3, :3] = res_R
                pose[:3, 3] = res_t
                output_path = "nvm_export_ba" + os.path.splitext(os.path.basename(self.query_img_path))[0]
                PoseFinder.exportPoseToNVM(self.query_img_path, scene_center, photo_img_undistort, intr1, pose, points2d, coords3d, output_path)

        plt.show()

    def analyzeMatches(self):
        """ Allow the user to manually click minimum number of correspondences
            and estimate camera pose."""
        maxw = 1024
        self.photo, self.photo_fov, self.photo_P = MatchesAnalyzer.loadImage(self.query_img_path)
        if np.max(self.photo.shape) > maxw:
            scale = maxw / np.max(self.photo.shape)
            self.photo = cv2.resize(self.photo, (int(self.photo.shape[1] * scale), int(self.photo.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        bare_query_img_path = self.query_img_path.replace("_texture", "")
        photo_MV_path = os.path.splitext(bare_query_img_path)[0] + "_modelview.txt"
        if not os.path.isfile(photo_MV_path):
            self.photo_MV = np.eye(4, 4)
        else:
            self.photo_MV = FUtil.loadMatrixFromFile(photo_MV_path)
        self.orig_photo_t = self.photo_MV[:3, 3]

        self.render, self.render_fov, self.render_P = MatchesAnalyzer.loadImage(self.rendered_img_path)
        if np.max(self.render.shape) > maxw:
            scale = maxw / np.max(self.render.shape)
            self.render = cv2.resize(self.render, (int(self.render.shape[1] * scale), int(self.render.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        rendered_depth_path = os.path.splitext(self.rendered_img_path)[0] + "_depth.txt.gz"
        bare_rendered_path = self.rendered_img_path.replace("_texture", "")
        self.render_MV_path = os.path.splitext(bare_rendered_path)[0] + "_modelview.txt"
        if not os.path.isfile(self.render_MV_path):
            self.render_MV_path = os.path.splitext(bare_rendered_path)[0] + "_pose.txt"
        self.render_MV = FUtil.loadMatrixFromFile(self.render_MV_path)
        print(self.render_MV)

        self.render_depth = loadDepth(rendered_depth_path)
        self.render_depth = cv2.resize(self.render_depth, (self.render.shape[1], self.render.shape[0]), interpolation=cv2.INTER_AREA)

        self.render = self.loadAdditionalChannels(self.render, self.render_depth)

        self.left_pts_coords = np.zeros((self.num_clicked_pts, 2))
        self.right_pts_coords = np.zeros((self.num_clicked_pts, 2))
        print("before figure")
        self.fig = plt.figure()
        self.matches_label = plt.suptitle('Click correspondence: ' + str(self.click_idx))
        ax1 = plt.subplot(1, 2, 1)
        ax1.imshow(self.photo)
        self.left_click_markers = ax1.scatter(self.left_pts_coords[:, 0], self.left_pts_coords[:, 1], alpha=1.0, color='red', marker='x')
        plt.axis("off")
        ax2 = plt.subplot(1, 2, 2)
        ax2.imshow(self.render[:, :, :3].astype(np.int))
        self.right_click_markers = ax2.scatter(self.right_pts_coords[:, 0], self.right_pts_coords[:, 1], alpha=1.0, color='red', marker='x')
        plt.axis("off")

        cid = self.fig.canvas.mpl_connect('button_press_event', self.handleClickMatch)
        cid = self.fig.canvas.mpl_connect('key_press_event', self.handleKeypressMatch)
        print("before show")
        plt.show()




def buildArgumentParser():
    parser = ap.ArgumentParser()
    parser.add_argument("photo", help="Photograph to be matched with the \
                        rendered view.")
    parser.add_argument("render", help="Rendered view to be matched with the \
                        photograph.")
    parser.add_argument("snapshot", help="""Restores model from
                        the snapshot with given name. If no model is specified
                        using -m, loads latest model.""")
    parser.add_argument("mode", help="Define the what should be analyzed. \
                        Available options: distance, matches.")
    parser.add_argument("-l", "--log_dir", help="""Directory used for saving
                        training progress logs and models. Defaults to cwd
                        when not specified.""", default=os.getcwd())
    parser.add_argument("-m", "--model_name", help="""Specify exact model name
                        to be restored using -r option.""")
    parser.add_argument("-c", "--cuda", action="store_true", help="If this \
                        flag is used, cuda will be used for neural network\
                        processing.")
    parser.add_argument("-no", "--normalize_output", action="store_true",
                        help="If set, the descriptors are normalized to unit \
                        hypersphere.")
    parser.add_argument("--stride", help="Stride used to compute the dense \
                        representations. Default=12", default=12, type=int)
    parser.add_argument("--refine", action='store_true', help="Use this flag \
                        to run refinement of the matching using multiple \
                        scales.")
    parser.add_argument("--use-depth", action='store_true')
    parser.add_argument("--use-normals", action='store_true')
    parser.add_argument("--use-silhouettes", action='store_true')
    parser.add_argument("--maxres", type=int, default=4096,
                        help="Resolution for recalculating images size \
                        w.r.t its FOV. Maxres corrsponds to FOV=180deg.")
    parser.add_argument("--keypoints-heatmap", action='store_true', help="Can be used \
                        only in distance mode. With this option on, the \
                        keypoint scores will be visualized. To use this \
                        feature, a network of class \
                        MultimodalKeypointPatchNet5lShared2l \
                        must be used, otherwise all scores will be one.")
    parser.add_argument("--fcn-keypoints", action='store_true', help="Use \
                        fully convolutional variant of our keypoint \
                        two-branch net. This allows dense extraction of \
                        keypoints from the whole image at one step. Uses \
                        FOV of the image to detect keypoints at single \
                        scale")
    parser.add_argument("--fcn-keypoints-multiscale",
                        action='store_true', help="Use \
                        fully convolutional variant of our keypoint \
                        two-branch net. This allows dense extraction of \
                        keypoints from the whole image at one step. Does not \
                        use FOV and detects keypoints on multiple scales.")
    parser.add_argument("--sift-keypoints", action='store_true')
    parser.add_argument("--saddle-keypoints", action='store_true')
    parser.add_argument("--d2net", action='store_true', help=" \
                        User original D2Net keypoints and descriptors for \
                        matching.")
    parser.add_argument("--ncnet", action='store_true', help="User original \
                        NCNet matching.")
    parser.add_argument("--dense-uniform-keypoints", action='store_true',
                        help="Use densely uniformly sampled keypoints \
                        instead of any other keypoint detector.")
    parser.add_argument("--dense-halton-keypoints", action='store_true',
                        help="Use densely uniformly sampled keypoints \
                        instead of any other keypoint detector.")
    parser.add_argument("--needles", default=0, type=int, help="If number \
                        greater than zero is used, then instead of a single \
                        patch a whole needle of patches will be extracted. Our\
                        network then takes several patches in a form of a \
                        needle encoded to channels of the input. This \
                        approach is described here: Lotan and Irani: \
                        Needle-Match: Reliable Patch Matching under \
                        High Uncertainty, CVPR 2016")
    parser.add_argument("--sift-descriptors", help="Uses SIFT descriptors \
                        for matching, neural descriptors are not used.",
                        action='store_true')
    parser.add_argument("--p4pf", help="P4Pf and our own RANSAC will be used \
                        for matching. No FOV initialization is needed, since \
                        P4Pf estimates the focal length.", action='store_true')
    parser.add_argument("--p4pf-epnp-iterative",
                        help="P4Pf and our own RANSAC will be used \
                        to estimate initial solution of camera pose and FOV. \
                        The estimated FOV is then used to re-estimate the \
                        camera pose iteratively using \
                        epnp + ransac + bundle adjustment until the number of \
                        inliers keeps improving.",
                        action='store_true')
    parser.add_argument("--epnpor", help="EPNP and our own RANSAC will be used \
                        for matching.", action='store_true')

    return parser


if __name__ == "__main__":
    parser = buildArgumentParser()
    args = parser.parse_args()

    matchesAnalyzer = MatchesAnalyzer(args)

    if args.mode == "distance":
        matchesAnalyzer.analyzeDistance()
    elif args.mode == "matches":
        matchesAnalyzer.analyzeMatches()
    else:
        raise RuntimeError("Unknown mode: " + args.mode + ". Use one of: \
                           distance, matches.")

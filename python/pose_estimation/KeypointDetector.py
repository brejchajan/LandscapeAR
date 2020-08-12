# @Date:   2020-08-06T16:16:58+02:00
# @Email:  ibrejcha@fit.vutbr.cz; brejchaja@gmail.com
# @Project: LandscapeAR
# @Last modified time: 2020-08-11T17:46:38+02:00
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



import torch
import numpy as np
from scipy import ndimage as ndi
import skimage
import skimage.morphology
import cv2
from tqdm import tqdm

#D2Net
import imageio
import scipy.misc
import skimage.transform
from thirdparty.d2net.lib.utils import preprocess_image
from thirdparty.d2net.lib.pyramid import process_multiscale

import matplotlib.pyplot as plt

# halton sequence
from thirdparty.halton.halton import halton

# our code
from pose_estimation.patchSamplingDepth import generatePatchesFastImg, generatePatchesFastImgNoscale, generatePatchesImg, getSizeFOV
from trainPatchesDescriptors import getCoordsAtSiftKeypoints
plt.ioff()

class KeypointDetector(object):

    @staticmethod
    def getLocalMaxima(map, min_distance=3, mean_filter=True, mean_filter_val=0.75):
        image_max = ndi.maximum_filter(map, size=min_distance, mode='constant')
        non_plateau_mask = skimage.morphology.erosion(map)
        non_plateau_mask = map > non_plateau_mask
        if mean_filter:
            # mean = np.mean(map[non_plateau_mask])
            mean = np.quantile(map[non_plateau_mask], mean_filter_val) #0.75
        else:
            mean = -10000  # always true
        coords = np.where(
            np.logical_and(
                np.logical_and(map >= image_max, non_plateau_mask),
                map >= mean
            )
        )
        coords = np.array(coords).transpose()
        return coords

    @staticmethod
    def getKeypointProbaMap(img, describer, photo):
        img_gpu = torch.from_numpy(img.transpose(2, 1, 0).copy())[None, :]
        img_gpu = img_gpu.float().to(describer.device)
        if photo:
            photo_dense = describer.net_keypoints.forward_photo(img_gpu)
        else:
            photo_dense = describer.net_keypoints.forward_render(img_gpu)
        if type(photo_dense) is tuple:
            print("photo dense", photo_dense)
            photo_dense_score = photo_dense[1]
            proba_map = photo_dense_score.detach().cpu().numpy()[0, 0]
            proba_map = proba_map.transpose(1, 0)
        else:
            print("photo dense shape", photo_dense.shape)
            proba_map = torch.mean(photo_dense[0], dim=0).detach().cpu().numpy() #torch.median(photo_dense[0], dim=0)[0].detach().cpu().numpy()
            proba_map = np.transpose(proba_map)
            #print("proba map shape", proba_map.shape)
        return proba_map

    @staticmethod
    def getOurKeypointsScaleFCN(img, describer, photo):
        wp = img.shape[1]
        proba_map = KeypointDetector.getKeypointProbaMap(img, describer, photo)
        nw = proba_map.shape[1]

        coords = KeypointDetector.getLocalMaxima(
            proba_map, min_distance=5, mean_filter=True
        ) * (wp / nw)
        proba_map = cv2.resize(
            proba_map, (img.shape[1], img.shape[0]),
            interpolation=cv2.INTER_AREA
        )

        patchsize = 32
        sel = np.logical_and(coords > patchsize, coords < (np.array(img.shape[:2]).reshape(1, 2) - patchsize))
        sel = np.logical_and(sel[:, 0], sel[:, 1])

        coords = coords[sel]
        # plt.figure()
        # plt.imshow(proba_map)
        # plt.scatter(coords[:, 1], coords[:, 0], s = 0.5)
        # plt.show()

        return coords

    @staticmethod
    def computeBayesVariance(photo_dense, photo_dense_shp, numsamples):
        photo_dense = photo_dense.reshape(photo_dense.shape[0], -1).transpose(0, 1)
        pred = torch.mean(photo_dense, dim=1)  # .detach().cpu().numpy()
        p_hat = photo_dense  # .detach().cpu().numpy()
        temp = p_hat - torch.unsqueeze(pred, 1)
        epistemic = (torch.bmm(temp.unsqueeze(1), temp.unsqueeze(2)) / numsamples)
        epistemic = epistemic.reshape(photo_dense_shp[2], photo_dense_shp[3])
        aleatoric = pred - (torch.bmm(p_hat.unsqueeze(1), p_hat.unsqueeze(2)) / numsamples).reshape(-1)
        aleatoric = aleatoric.reshape(photo_dense_shp[2], photo_dense_shp[3])
        return epistemic.T.cpu().numpy(), aleatoric.T.cpu().numpy()

    @staticmethod
    def getOurKeypointsScaleBayesPart(img_gpu, describer, photo, numsamples=10, plot_variance=False, crop_borders=False):
        wp = img_gpu.shape[2]
        all_photo_dense = []
        for idx in range(0, numsamples):
            #if photo:
                #photo_dense = describer.net_keypoints.forward_photo(img_gpu, sample=True)
            photo_dense = describer(img_gpu, sample=True)
            #else:
                #photo_dense = describer.net_keypoints.forward_render(img_gpu, sample=True)
            #    photo_dense = describer(img_gpu, sample=True)
            photo_dense = torch.sigmoid(photo_dense)
            all_photo_dense.append(photo_dense.detach())
        photo_dense_orig = torch.cat(all_photo_dense)
        print("photo dense orig", photo_dense_orig.shape)
        #photo_dense = torch.mean(photo_dense, dim=1)
        photo_dense_shp = photo_dense.shape

        epistemic = []
        aleatoric = []
        for idx in range(0, photo_dense_shp[1]):
            # compute variance per dimension of the descriptor
            # (range = (0,128) for our descriptors)
            photo_dense = photo_dense_orig[:, idx]
            ep, al = KeypointDetector.computeBayesVariance(
                photo_dense, photo_dense_shp, numsamples
            )
            epistemic.append(ep)
            aleatoric.append(al)
        epistemic = np.array(epistemic)
        aleatoric = np.array(aleatoric)
        variance = epistemic + aleatoric

        proba_map = 1.0 - np.min(variance, axis=0)

        nw = proba_map.shape[1]
        coords = KeypointDetector.getLocalMaxima(
            proba_map, min_distance=0, mean_filter=True, mean_filter_val=0.5
        ) * (wp / nw)
        proba_map = cv2.resize(
            proba_map, (img_gpu.shape[2], img_gpu.shape[3]),
            interpolation=cv2.INTER_AREA
        )
        if plot_variance:
            plt.figure()
            plt.subplot(3, 1, 1)
            plt.imshow(np.min(epistemic, axis=0))
            plt.title("Epistemic")

            plt.subplot(3, 1, 2)
            plt.imshow(np.min(aleatoric, axis=0))
            plt.title("Aleatoric")

            plt.subplot(3, 1, 3)
            plt.imshow(proba_map)
            plt.title("Proba: 1 - (Aleatoric + Epistemic)")
            plt.show()
        if crop_borders:
            patchsize = 32
            sel = np.logical_and(coords > patchsize, coords < (np.array(img_gpu.shape[2:3]).reshape(1, 2) - patchsize))
            sel = np.logical_and(sel[:, 0], sel[:, 1])
            coords = coords[sel]

        return coords

    @staticmethod
    def getOurKeypointsScaleBayes(img, describer, photo, numsamples=50, plot_variance=False, crop_borders=False):
        if photo:
            desc0 = describer.net_keypoints.branch1.forward_part0
            desc1 = describer.net_keypoints.branch1.forward_part1
            desc2 = describer.net_keypoints.branch1.forward_part2
            desc3 = describer.net_keypoints.forward_photo
        else:
            desc0 = describer.net_keypoints.branch2.forward_part0
            desc1 = describer.net_keypoints.branch2.forward_part1
            desc2 = describer.net_keypoints.branch2.forward_part2
            desc3 = describer.net_keypoints.forward_render

        desc = [desc1]

        img_gpu = torch.from_numpy(img.transpose(2, 1, 0).copy())[None, :]
        img_gpu = img_gpu.float().to(describer.device)
        describer.net_keypoints.reshape = False

        coords = []
        for d in desc:
            c = KeypointDetector.getOurKeypointsScaleBayesPart(img_gpu, d, photo, numsamples, plot_variance=True, crop_borders=False)
            coords.append(c)
        coords = np.vstack(coords)
        return coords


    @staticmethod
    def getOurKeypointsScale(img, stride, describer, maxres, photo):
        wp = img.shape[1]
        # get patches from render and describe them
        rc_y = np.arange(0, img.shape[0], stride)
        nh = rc_y.shape[0]
        rc_x = np.arange(0, img.shape[1], stride)
        nw = rc_x.shape[0]
        rc_yv, rc_xv = np.meshgrid(rc_y, rc_x)
        render_coords = np.array(
            [rc_yv.reshape(-1), rc_xv.reshape(-1)]
        ).transpose()

        numpatches_batch = 5000
        render_fea = []
        render_fea_scores = []
        all_render_patches = []

        for idx in tqdm(range(0, render_coords.shape[0], numpatches_batch)):
            render_patches = generatePatchesFastImgNoscale(
                img, img.shape, render_coords[idx:idx + numpatches_batch],
                show=False, maxres=maxres
            )
            render_patches = torch.from_numpy(render_patches)
            all_render_patches.append(render_patches)
            if photo:
                print("keypoints describe photo")
                fea, fea_score = describer.describePhoto(render_patches)
            else:
                print("keypoints describe render")
                fea, fea_score = describer.describeRender(render_patches)
            render_fea.append(fea)
            render_fea_scores.append(fea_score)
        render_fea = np.concatenate(render_fea)
        render_fea_scores = np.concatenate(render_fea_scores)
        all_render_patches = np.concatenate(all_render_patches)

        proba_map = render_fea_scores.reshape(nw, nh).transpose()
        print("proba map info", np.max(proba_map), np.min(proba_map))

        coords = KeypointDetector.getLocalMaxima(
            proba_map, min_distance=3, mean_filter=True
        ) * (wp / nw)

        return coords

    @staticmethod
    def getOurKeypoints(img, stride, describer, maxres=4096, photo=False, fov=-1):
        # resize the image so that it matches FOV scale so that we don't
        # extract more features than needed
        orig_img = img.copy()

        scales = [0.25, 0.5, 0.75, 1.0] #0.25, 0.5, 0.75, 1.0,

        if fov > 0:
           #coords = coords / scale
           wp, hp, scale_p = getSizeFOV(orig_img.shape[1], orig_img.shape[0], fov, maxres=maxres)
           #img = cv2.resize(orig_img, (wp, hp), interpolation=cv2.INTER_AREA)
           scales = [scale_p]
           #coords = coords * scale_p

        all_coords = []

        render_fea = []
        render_fea_scores = []
        all_render_patches = []

        for scale in scales:
            print("scale ", scale)
            kpscale = 2
            wp = int(orig_img.shape[1] * scale * kpscale)
            hp = int(orig_img.shape[0] * scale * kpscale)
            img = cv2.resize(orig_img, (wp, hp), interpolation=cv2.INTER_AREA)
            new_stride = max(int(stride * scale * 2), 1)
            if describer.net_keypoints.__class__.__name__ == "BBBMultimodalPatchNet5lShared2l":
                coords = KeypointDetector.getOurKeypointsScaleBayes(
                    img, describer, photo
                )
            elif describer.fcn_keypoints or describer.fcn_keypoints_multiscale:
                coords = KeypointDetector.getOurKeypointsScaleFCN(
                    img, describer, photo
                )
            else:
                coords = KeypointDetector.getOurKeypointsScale(
                    img, new_stride, describer, maxres, photo
                )
            coords = coords / kpscale
            wp = int(orig_img.shape[1] * scale)
            hp = int(orig_img.shape[0] * scale)
            img = cv2.resize(orig_img, (wp, hp), interpolation=cv2.INTER_AREA)
            numpatches_batch = 5000

            for idx in tqdm(range(0, coords.shape[0], numpatches_batch)):
                render_patches = generatePatchesFastImgNoscale(
                    img, img.shape, coords[idx:idx + numpatches_batch],
                    show=False, maxres=maxres
                )
                render_patches = torch.from_numpy(render_patches)
                all_render_patches.append(render_patches)
                if photo:
                    fea, fea_score = describer.describePhoto(render_patches)
                else:
                    fea, fea_score = describer.describeRender(render_patches)
                render_fea.append(fea)
                render_fea_scores.append(fea_score)

            all_coords.append(coords / scale)

        render_fea = np.concatenate(render_fea)
        render_fea_scores = np.concatenate(render_fea_scores)
        all_render_patches = np.concatenate(all_render_patches)
        coords = np.concatenate(all_coords)

        kp = []
        for idx in range(0, coords.shape[0]):
            kp.append(cv2.KeyPoint(coords[idx, 1], coords[idx, 0], 1))

        return orig_img, kp, None, render_fea, all_render_patches

    @staticmethod
    def getDenseRepresentationsWithKp(img, fov, stride, describer, maxres=4096, photo=False):
        # resize the image so that it matches FOV scale so that we don't
        # extract more features than needed
        orig_img = img.copy()

        if not photo:
            stride = int(stride / np.sqrt(2.0))

        scale = 1.0
        #wp, hp, scale_p = getSizeFOV(img.shape[1], img.shape[0], fov, maxres=maxres)
        #scale = img.shape[1] / wp
        #img = skimage.transform.resize(img, (hp, wp))

        print("get dense representations image size", img.shape)

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
            #render_patches = generatePatchesFastImgNoscale(img, img.shape, render_coords[idx:idx + numpatches_batch], show=False, maxres=maxres)
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
            #print("coords", render_coords[idx, 0], render_coords[idx, 1])
            #print("score", render_fea_scores[idx])
            kp.append(cv2.KeyPoint(render_coords[idx, 1] * scale, render_coords[idx, 0] * scale, 1))

        return orig_img, kp, None, render_fea, all_render_patches


    @staticmethod
    def getHaltonDenseRepresentationsWithKp(img, fov, stride, describer, maxres=4096, photo=False):
        # resize the image so that it matches FOV scale so that we don't
        # extract more features than needed
        orig_img = img.copy()

        wp, hp, scale_p = getSizeFOV(img.shape[1], img.shape[0], fov, maxres=maxres)
        scale = img.shape[1] / wp
        img = skimage.transform.resize(img, (hp, wp))

        # get patches from render and describe them
        patchsize = 32
        render_kp_coeff = 1
        if not photo:
            render_kp_coeff = 2
        numpoints = int((wp / float(stride)) * (hp / float(stride)))
        render_coords = halton(2, numpoints * render_kp_coeff)
        render_coords[:, 0] = (render_coords[:, 0] * (hp - 2.0 * patchsize) + patchsize)
        render_coords[:, 1] = (render_coords[:, 1] * (wp - 2.0 * patchsize) + patchsize)
        render_coords = render_coords.astype(np.int32)

        # skip points in sky of the rendered image
        img_sel = img[render_coords[:, 0], render_coords[:, 1]]
        sky_sel = (np.all((img_sel != 0), axis=1)).reshape(-1)
        render_coords = render_coords[sky_sel]

        numpatches_batch = 5000
        render_fea = []
        render_fea_scores = []
        all_render_patches = []
        for idx in tqdm(range(0, render_coords.shape[0], numpatches_batch)):
            render_patches = generatePatchesFastImgNoscale(img, img.shape, render_coords[idx:idx + numpatches_batch], show=False, maxres=maxres)
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
            #print("coords", render_coords[idx, 0], render_coords[idx, 1])
            #print("score", render_fea_scores[idx])
            kp.append(cv2.KeyPoint(render_coords[idx, 1] * scale, render_coords[idx, 0] * scale, 1))

        return orig_img, kp, None, render_fea, all_render_patches

    @staticmethod
    def loadImageAndD2Net(photo_name, photo_shape, describer, photo=False):
        image = imageio.imread(photo_name)
        if len(image.shape) == 2:
            image = image[:, :, np.newaxis]
            image = np.repeat(image, 3, -1)

        # TODO: switch to PIL.Image due to deprecation of scipy.misc.imresize.
        resized_image = image
        resized_image = cv2.resize(
            resized_image, (photo_shape[0], photo_shape[1])
        ).astype('float')

        return KeypointDetector.describeD2Net(resized_image, describer, photo)

    @staticmethod
    def describeD2Net(resized_image, describer, photo=False):
        input_image = preprocess_image(
            resized_image,
            preprocessing='caffe'
        )
        with torch.no_grad():
            keypoints, scores, descriptors = process_multiscale(
                torch.tensor(
                    input_image[np.newaxis, :, :, :].astype(np.float32),
                    device=describer.device
                ),
                describer.d2net
            )

        # Input image coordinates
        keypoints = keypoints[:, [1, 0, 2]]

        kp1 = KeypointDetector.d2NetToCVKeypoints(keypoints)

        return resized_image, kp1, scores, descriptors, None

    @staticmethod
    def d2NetToCVKeypoints(keypoints):
        kp1 = []
        for idx in range(keypoints.shape[0]):
            kp1.append(cv2.KeyPoint(keypoints[idx][0], keypoints[idx][1], 1))
        return kp1

    @staticmethod
    def sadKeyPointsToCVKeyPoints(sad_kpts):
        kps = []
        for sad_kp in sad_kpts:
            kp = cv2.KeyPoint(
                sad_kp.pt[0], sad_kp.pt[1], sad_kp.size, sad_kp.angle,
                sad_kp.response, sad_kp.octave, sad_kp.class_id)
            kps.append(kp)
        return kps

    @staticmethod
    def detectSaddleKeypointsAndDescribe(
        img1, fov, describer, photo=False
    ):

        sad_kpts = describer.sorb.detectSadKeypoints(img1)
        kp1 = KeypointDetector.sadKeyPointsToCVKeyPoints(sad_kpts)
        coords1 = getCoordsAtSiftKeypoints(kp1)
        patches1 = generatePatchesFastImg(
            img1, img1.shape, coords1, fov,
            maxres=describer.maxres, needles=describer.needles
        )
        patches1 = torch.from_numpy(patches1)
        if photo:
            p1, p1_score = describer.describePhoto(patches1)
        else:
            p1, p1_score = describer.describeRender(patches1)
        return img1, kp1, None, p1, patches1

# @Date:   2020-08-07T12:34:17+02:00
# @Email:  ibrejcha@fit.vutbr.cz; brejchaja@gmail.com
# @Project: LandscapeAR
# @Last modified time: 2020-08-11T17:49:06+02:00
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

import os

import cv2
import exiftool

# D2Net
from thirdparty.d2net.lib.model_test import D2Net
from thirdparty.d2net.lib.utils import preprocess_image
from thirdparty.d2net.lib.pyramid import process_multiscale

import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from pose_estimation.patchSamplingDepth import generatePatchesImgScale, generatePatchesFastImg
from Matcher import Matcher
from pose_estimation import FUtil


class FeatureExtractor(object):
    def __init__(self, input_dir, image_list_file, cuda=True):
        super(FeatureExtractor, self).__init__()
        self.input_dir = input_dir
        self.image_list_file = image_list_file
        self.cuda = cuda
        self.maxw = 1024
        self.loadImageNames()
        self.device = torch.device("cuda:0" if self.cuda else "cpu")

    def loadImageNames(self):
        with open(self.image_list_file, 'r') as f:
            self.image_names = [
                os.path.join(self.input_dir, l.strip()) for l in f.readlines()
            ]

    def extract(self):
        raise RuntimeWarning("Unimplemented abstract method.")


class D2NetFeatureExtractor(FeatureExtractor):
    def __init__(self, input_dir, image_list_file, cuda=True, recompute=False):
        super(D2NetFeatureExtractor, self).__init__(
            input_dir, image_list_file, cuda
        )
        self.recompute = recompute

        current_path = os.path.dirname(os.path.realpath(__file__))
        self.d2net = D2Net(
            model_file=os.path.join(
                current_path,
                '../pretrained/d2_net/d2_tf.pth'
            ),
            use_relu=True,
            use_cuda=self.cuda
        )

    def extract(self):
        for name in tqdm(self.image_names):
            outfile = name + ".npz"
            outfile_exists = os.path.isfile(outfile)
            if not outfile_exists or (outfile_exists and self.recompute):
                img = cv2.imread(name)
                img = np.flip(img, 2)

                scale = 1
                if np.max(img.shape) > self.maxw:
                    scale = self.maxw / np.max(img.shape)
                    resized_image = cv2.resize(
                        img,
                        (int(img.shape[1] * scale), int(img.shape[0] * scale)),
                        interpolation=cv2.INTER_AREA
                    )
                else:
                    resized_image = img
                input_image = preprocess_image(
                    resized_image,
                    preprocessing='caffe'
                )
                with torch.no_grad():
                    keypoints, scores, descriptors = process_multiscale(
                        torch.tensor(
                            input_image[np.newaxis, :, :, :].astype(np.float32),
                            device=self.device
                        ),
                        self.d2net
                    )

                # Input image coordinates
                keypoints /= scale
                # i, j -> u, v
                keypoints = keypoints[:, [1, 0, 2]]

                # plt.ioff()
                # plt.imshow(img)
                # plt.scatter(keypoints[:, 0], keypoints[:, 1], s=0.5, c='red')
                # plt.show()

                # save
                with open(outfile, 'wb') as output_file:
                    np.savez(
                        output_file,
                        keypoints=keypoints,
                        scores=scores,
                        descriptors=descriptors
                    )


class CrossDomainFeatureExtractor(FeatureExtractor):
    def __init__(
        self, input_dir, image_list_file, cuda=True, recompute=False,
        patchsize=64, maxres=3000, photo_fov_default=50
    ):
        super(CrossDomainFeatureExtractor, self).__init__(
            input_dir, image_list_file, cuda
        )
        self.recompute = recompute
        self.patchsize = patchsize
        self.maxres = maxres
        self.photo_fov_default = (photo_fov_default / 180.0) * np.pi

        current_path = os.path.dirname(os.path.realpath(__file__))
        log_dir = os.path.join(
            current_path,
            "../pretrained", "landscapeAR", "Ours-aux"
        )
        models_path = os.path.join(log_dir, "models")
        model_path = os.path.join(
            models_path, "MultimodalPatchNet5lShared2l_epoch_21_step_1210000"
        )
        model_name = os.path.basename(model_path).split("_epoch_")[0]
        module = __import__("training").Architectures
        net_class = getattr(module, model_name)
        self.net = net_class(True)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.net = self.net.to(self.device)
        self.net.eval()

    @staticmethod
    def getCoordsAtSiftKeypoints(kp):
        coords = []
        sizes = []
        for key in kp:
            coords.append(np.array([key.pt[1], key.pt[0]]))
            sizes.append(key.size)
        coords = np.array(coords)
        sizes = np.array(sizes)
        return coords, sizes

    def computeSift(self, img, contrastThr=0.04, edgeThr=10,
                    sigma=1.6, nfeatures=5000, scale=False):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        sift = cv2.xfeatures2d.SIFT_create(nfeatures=nfeatures,
                                           contrastThreshold=contrastThr,
                                           edgeThreshold=edgeThr,
                                           sigma=sigma)
        kp = sift.detect(gray, None)
        # in case we get no keypoints add dummy keypoint in order not to
        # crash the whole pipeline
        if len(kp) == 0:
            kp = [cv2.KeyPoint(img.shape[0] / 2.0, img.shape[1] / 2.0, 1)]

        coords, sizes = CrossDomainFeatureExtractor.getCoordsAtSiftKeypoints(kp)
        #sel = np.logical_and(
        #    coords > self.patchsize,
        #    coords < (np.array(img.shape[:2]).reshape(1, 2) - self.patchsize)
        #)
        #sel = np.logical_and(sel[:, 0], sel[:, 1])

        return coords, sizes

    def describeRender(self, patches1, batchsize=1000):
        p1 = []
        for idx in range(0, patches1.shape[0], batchsize):
            batch = patches1[idx:idx + batchsize].to(self.device)
            p1_fea = self.net.forward_render(batch)
            p1_fea = p1_fea.detach().cpu().numpy()
            p1.append(p1_fea)
        p1 = np.concatenate(p1)
        p1 = p1.reshape(p1.shape[0], p1.shape[1])
        return p1

    def describePhoto(self, patches1, batchsize=1000):
        p1 = []
        for idx in range(0, patches1.shape[0], batchsize):
            batch = patches1[idx:idx + batchsize].to(self.device)
            p1_fea = self.net.forward_photo(batch)
            p1_fea = p1_fea.detach().cpu().numpy()
            p1.append(p1_fea)
        p1 = np.concatenate(p1)
        p1 = p1.reshape(p1.shape[0], p1.shape[1])
        return p1

    def extract(self):
        for name in tqdm(self.image_names):
            outfile = name + ".npz"
            outfile_exists = os.path.isfile(outfile)
            if not outfile_exists or (outfile_exists and self.recompute):
                img = cv2.imread(name)
                resized_image = img

                if 'render' in name:
                    # render
                    coords, sizes = self.computeSift(
                        resized_image, contrastThr=0.02, edgeThr=15,
                        sigma=1.0, nfeatures=10000
                    )
                    img = np.flip(img, 2)
                    patches = generatePatchesImgScale(
                        img, img.shape, coords, sizes,
                        show=False, maxres=self.maxres
                    )
                    patches = torch.from_numpy(patches)
                    descriptors = self.describeRender(patches)
                else:
                    # photo
                    photo_fov = self.photo_fov_default
                    with exiftool.ExifTool() as et:
                        res = et.execute_json("-n", "-FOV", name)
                        if 'Composite:FOV' in res[0]:
                            photo_fov = res[0]['Composite:FOV']
                            photo_fov = (photo_fov / 180.0) * np.pi

                    coords, sizes = self.computeSift(
                        resized_image, contrastThr=0.01, edgeThr=10,
                        sigma=1.0, nfeatures=10000
                    )
                    img = np.flip(img, 2)
                    patches = generatePatchesImgScale(
                        img, img.shape, coords, sizes,
                        show=False, maxres=self.maxres
                    )
                    patches = torch.from_numpy(patches)
                    descriptors = self.describePhoto(patches)

                sizes = sizes.reshape(-1, 1)
                coords = np.flip(coords, 1)
                keypoints = np.concatenate([coords, sizes], axis=1)
                # save
                with open(outfile, 'wb') as output_file:
                    np.savez(
                        output_file,
                        keypoints=keypoints,
                        scores=np.ones(coords.shape[0]),
                        descriptors=descriptors
                    )


if __name__ == "__main__":
    input_dir = "/home/ibrejcha/hdd_data/data/locate/photoparam_raw/flickr_download/geolocated/switzerland_wallis_30km_maximg/small_dset_real_synth_reconstruction"
    image_list_file = "/home/ibrejcha/hdd_data/data/locate/photoparam_raw/flickr_download/geolocated/switzerland_wallis_30km_maximg/small_dset_real_synth_reconstruction/image_list.txt"
    extractor = D2NetFeatureExtractor(input_dir, image_list_file)
    #extractor.extract()

    extractorOur = CrossDomainFeatureExtractor(input_dir, image_list_file, recompute=True)
    extractorOur.extract()

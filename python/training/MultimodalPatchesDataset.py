# @Date:   2020-08-10T17:50:38+02:00
# @Email:  ibrejcha@fit.vutbr.cz; brejchaja@gmail.com
# @Project: LandscapeAR
# @Last modified time: 2020-08-11T17:50:07+02:00
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

from __future__ import print_function, division

# our code
from pose_estimation.patchSamplingDepth import generatePatchesFast, savePointCloudToPly
from pose_estimation.patchSamplingDepth import loadDepth, getSizeFOV, project, unproject

from timeit import default_timer as timer
from sklearn.neighbors import KDTree
from sklearn.neighbors import KernelDensity
from pose_estimation import FUtil

import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from tqdm import tqdm
from scipy import ndimage
import cv2
cv2.setNumThreads(0)


class MultimodalPatchesDataset(Dataset):
    """Class responsible for loading multimodal patches dataset and triplet
    sampling."""

    def __init__(self, dataset_dir, images_path, list=[],
                 numpatches=900, numneg=3, pos_thr=50.0, reject=True,
                 mode='train', rejection_radius=3000, dist_type='3D',
                 patch_radius=None, use_depth=False, use_normals=False,
                 use_silhouettes=False, color_jitter=False, greyscale=False,
                 maxres=4096, scale_jitter=False, photo_jitter=False,
                 uniform_negatives=False, needles=0, render_only=False):
        """Loads the patches dataset.
           @param dataset_dir String directory where the dataset of sampled
           points is located
           @param images_path path to the images to sample patches from
           @param list List of subdirectory names to be loaded with this
           loader. Use this to specify train/test/val splits.
           @param numneg Int number of generated negatives per positive pair.
           @param pos_thr Float threshold in meters used to define negatives.
           If the distance of two 3D points exceeds this threshold, the
           correspondence is considered negative. The lower the threshold, the
           harder the negatives are.
           @param reject [bool] True turns on rejetion sampling - for each
           patch we calculate density of 3D reprojected point cloud within 1km
           radius. Then the probability of rejection is calculated as
           num_points_1km_radius/max_num_points, where max_num_points is
           maximum taken across all queried samples until the current one.
           @param mode options: train|eval, default: train. If train is used,
           then the additional metadata per patch (which are used for some
           plots during validation are not generated and therefore the training
           shall be faster.
           @type string
           @param dist_type type of the distance used to generate positives and
           negatives. Can be `2D` or `3D`. Default: 3D.
           @type int
           @param patch_radius when set to None, the patch radius will be
           loaded from the patches dataset. Otherwise the defined patch radius
           will be used. Please note that if you use larger patch_radius than
           the one defined within the patches dataset, the source image will be
           padded automatically and so the patch may contain black edges.
           @param needles If number greater than zero is used, then instead of
           a single patch a whole needle of patches will be extracted. Our
           network then takes several patches in a form of a needle encoded to
           channels of the input. This approach is described here:
           Lotan and Irani: Needle-Match: Reliable Patch Matching under
           High Uncertainty, CVPR 2016.
        """
        self.item_idx = -1
        self.dataset_dir = dataset_dir
        self.images_path = images_path
        self.numneg = numneg
        self.pos_thr = pos_thr
        self.loaded_imgs_pts = []
        self.all_coords3d = []
        self.max_num_points = 0
        self.reject = reject
        self.query_radius = rejection_radius
        self.dist_type = dist_type
        self.use_depth = use_depth
        self.use_normals = use_normals
        self.use_silhouettes = use_silhouettes
        self.color_jitter = color_jitter
        self.greyscale = greyscale
        self.left_maxres = maxres
        self.right_maxres = maxres
        self.scale_jitter = scale_jitter
        self.photo_jitter = photo_jitter
        self.uniform_negatives = uniform_negatives
        self.needles = needles
        self.render_only = render_only

        scene_info_file = os.path.join(
            os.path.dirname(images_path), "scene_info.txt"
        )
        self.scene_center = MultimodalPatchesDataset.getSceneCenter(scene_info_file)

        self.numch_1 = 3
        self.numch_2 = 3
        if self.greyscale:
            self.numch_1 = 1
            self.numch_2 = 1

        if self.use_depth:
            self.numch_2 += 1
        if self.use_normals:
            self.numch_2 += 3
        if self.use_silhouettes:
            self.numch_2 += 1

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(0.5, 0.5, 1.0, 0.5),
            transforms.ToTensor()
        ])

        print("Rejection radius: ", self.query_radius, "mode", mode)
        self.mode = mode
        if len(list) == 0:
            self.dataset_items = [d for d in os.listdir(self.dataset_dir) if
                              os.path.isdir(os.path.join(self.dataset_dir, d))]
        else:
            self.dataset_items = []
            if self.mode == 'eval':
                # choose only pairs where left view does not repeat
                print("Choosing non-repeating photographs for validation...")
                keyset = set()
                for item in tqdm(list):
                    item_path = os.path.join(self.dataset_dir, item)
                    info_path = os.path.join(item_path, "info.npy")
                    info = np.load(info_path, encoding='latin1', allow_pickle=True).flatten()[0]
                    img1_base = os.path.basename(info['img1_name'])
                    key = os.path.splitext(img1_base)[0]
                    if key in keyset:
                        continue
                    keyset.add(key)
                    self.dataset_items.append(item)
            else:
                self.dataset_items = list

        if (len(self.dataset_items) > 0):
            item_path = os.path.join(self.dataset_dir, self.dataset_items[0])
            info_path = os.path.join(item_path, "info.npy")
            self.info = np.load(info_path, encoding='latin1', allow_pickle=True).flatten()[0]
            self.numpatches = self.info['coords2d_1'].shape[0]
            if patch_radius is not None:
                self.patch_radius = patch_radius
            else:
                self.patch_radius = self.info['patch_radius']
            if numpatches != self.numpatches:
                raise RuntimeError("Wrong number of patches in the first \
                        item of the dataset. Expected: " + str(numpatches) + ", obtained: " + str(self.numpatches))
            self.load3DPoints()
            self.kdt = KDTree(self.all_coords3d[:, :3], leaf_size=40, metric='euclidean')

            translation_frac = np.sqrt(5) / (self.patch_radius * 2) # at most 5px
            self.photo_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ColorJitter(0.2, (0.9, 1.001), 0.2, 0.2),
                transforms.RandomAffine(22.5, (translation_frac, translation_frac), shear = 5),
                transforms.CenterCrop(self.patch_radius * 2),
                transforms.ToTensor()
            ])

            if self.photo_jitter:
                self.prcoef = 1.25
            else:
                self.prcoef = 1


            # FIXME: remove since this is unneeded for training and is slow. Just for research.
            #self.saveDensityPointcloud()

        else:
            raise RuntimeError("No dataset items at specified location.")

    @staticmethod
    def getSceneCenter(scene_info_name):
        center = None
        with open(scene_info_name, 'r') as sin:
            lines = sin.readlines()
            center = np.fromstring(lines[0].split(":")[1].strip(), sep=" ")
        if center is None:
            raise RuntimeError("Unable to load scene info " + scene_info_name)
        return center

    def saveDensityPointcloud(self):
        count = self.calculateDensityRadius(self.all_coords3d[:, :3])
        cnt_max = np.max(count)
        self.max_num_points = cnt_max
        density = np.log(count / cnt_max)
        norm = matplotlib.colors.Normalize()
        colors = np.floor(plt.cm.jet(norm(density)) * 255)
        savePointCloudToPly(self.all_coords3d[:, :3], colors[:, :3], "dataset_pointcloud_heatmap_10.ply")

    def load3DPoints(self):
        checked_dataset_items = []
        left_imgs_dict = {}
        for item in tqdm(self.dataset_items):
            item_path = os.path.join(self.dataset_dir, item)
            base_path_parts = item.split("-")
            left_img = base_path_parts[0]
            right_img = "-".join(base_path_parts[1:])
            to_process = [left_img, right_img]
            info_path = os.path.join(item_path, "info.npy")
            if os.path.isfile(info_path):
                checked_dataset_items.append(item)
                if left_img in left_imgs_dict.keys():
                    left_imgs_dict[left_img].append(right_img)
                else:
                    left_imgs_dict.update({left_img: [right_img]})
                info_loaded = False
                info = []
                for img_name in to_process:
                    if img_name not in self.loaded_imgs_pts:
                        self.loaded_imgs_pts.append(img_name)
                        if not info_loaded:
                            info = np.load(info_path, encoding='latin1', allow_pickle=True).flatten()[0]
                            info_loaded = True
                        for idx in range(0, len(base_path_parts)):
                            if (base_path_parts[idx] == img_name):
                                coords3d = info['coords3d_' + str(idx + 1)]
                                self.all_coords3d.append(coords3d)
        if self.mode == 'train':
            # randomly select non-repeating pairs
            checked_dataset_items = []
            print("left imgs keys number: ", len(left_imgs_dict.keys()))
            for left_img in left_imgs_dict.keys():
                right_imgs = left_imgs_dict[left_img]
                randi = np.random.randint(0, len(right_imgs))
                item = left_img + "-" + right_imgs[randi]
                checked_dataset_items.append(item)
        self.dataset_items = checked_dataset_items
        self.all_coords3d = np.concatenate(self.all_coords3d)
        norm = matplotlib.colors.Normalize()
        colors = np.floor(plt.cm.jet(norm(np.arange(0,self.all_coords3d.shape[0]).astype(np.int).tolist())) * 255).astype(np.int)
        #colors = np.random.uniform(0, 255, (self.all_coords3d.shape[0],3)).astype(np.int)
        #red_colors = np.tile(red_c, (self.all_coords3d.shape[0],1))
        #savePointCloudToPly(self.all_coords3d[:, :3], colors[:, :3], "dataset_pointcloud.ply")

    def __len__(self):
        return len(self.dataset_items) * self.numpatches

    def sampleNegatives(self, patch_idx, info):
        if self.uniform_negatives:
            dist = self.dists_neg[patch_idx, :]
            dist_idx = np.arange(0, dist.size)
            row_dist_idx = dist_idx[dist > (self.pos_thr * 2.0)]
            np.random.shuffle(row_dist_idx)
            return row_dist_idx[:self.numneg], None
        else:
            dist_idx = np.argsort(self.dists[patch_idx, :])
            dist_sorted = np.sort(self.dists[patch_idx, :])
            pos_row_dist_idx = dist_idx[dist_sorted <= self.pos_thr]
            if pos_row_dist_idx.shape[0] == 0:
                pos_row_dist_idx = np.array([patch_idx])
            else:
                np.random.shuffle(pos_row_dist_idx)
            row_dist_idx = dist_idx[dist_sorted > (self.pos_thr * 2.0)]
            np.random.shuffle(row_dist_idx)
            return row_dist_idx[:self.numneg], pos_row_dist_idx[0]


    def generateSample(self, pid):
        ng, pt = self.sampleNegatives(pid, self.info)

        coords_rand_neg = None

        if self.mode == 'train':
            #return 3D coords during training for hardnet
            if self.dist_type == '3D':
                coords1 = self.info['coords3d_1'][pid][:3] + self.scene_center
                coords2 = self.info['coords3d_2'][pid][:3] + self.scene_center
            else:
                #return 2D coordinates corresponding to the right view
                coords1 = self.v1_2_2d[pid]
                coords2 = self.v2_2_2d[pid]
        else:
            coords1 = self.info['coords2d_1'][pid]
            coords2 = self.info['coords2d_2'][pid]

        if self.uniform_negatives:
            if self.dist_type == '3D':
                coords_rand_neg = self.uniform_randcoords_out_3D[ng][:, :3] + self.scene_center
            else:
                coords_rand_neg = self.uniform_randcoords_out[ng]
        else:
            if self.dist_type == '3D':
                coords_rand_neg = self.info['coords3d_2'][ng][:, :3]
            else:
                coords_rand_neg = self.v2_2_2d[ng]
        if self.numneg == 1 and coords_rand_neg.shape[0] >= 1:
            coords_rand_neg = coords_rand_neg[0]

        if self.photo_jitter:
            ap = self.photo_transform(self.patches_1[pid])
            pp = self.photo_transform(self.patches_2p[pid])
            negp = self.patches_2negp[ng]

            for ineg in range(0, self.numneg):
                negp[ineg] = self.photo_transform(negp[ineg])
        else:
            ap = self.patches_1[pid]
            pp = self.patches_2p[pid]
            negp = self.patches_2negp[ng]

        return ap, self.patches_2[pid], self.patches_2negr[ng], self.patches_1r[pid], pp, negp, coords1, coords2, coords_rand_neg

    @staticmethod
    def loadAndSaveDepth(img1_depth_path, img1_depth_path_npy, w, h):
        img1_depth = loadDepth(img1_depth_path)
        img1_depth = cv2.resize(img1_depth, (w, h), interpolation=cv2.INTER_AREA)[:, :, None]
        if img1_depth.shape[1] != w or img1_depth.shape[0] != h:
            print("WRONGLY RESIZED depth.", img1_depth.shape, w, h)
        np.save(img1_depth_path_npy, img1_depth)
        return img1_depth

    def createAndSaveNormals(self, img1_depth, img1_depth_path, img1_depth_path_npy, img1_normals_path):
        if img1_depth is None:
            img1_depth = self.loadAndCacheDepth(img1_depth_path, img1_depth_path_npy)
        img1_normals = MultimodalPatchesDataset.normalsFromDepth(img1_depth)
        img1_normals_to_save = np.floor(np.flip(img1_normals * 65535, 2)).astype(np.uint16)
        cv2.imwrite(img1_normals_path, img1_normals_to_save)
        return img1_normals

    def createAndSaveSilhouettes(self, img1_depth, img1_depth_path, img1_depth_path_npy, img1_silhouettes_path):
        if img1_depth is None:
            img1_depth = self.loadAndCacheDepth(img1_depth_path, img1_depth_path_npy)
        img1_silhouettes = MultimodalPatchesDataset.silhouettesFromDepth(img1_depth)
        img1_silhouettes_to_save = np.floor(np.flip(img1_silhouettes * 255, 2)).astype(np.uint8)
        cv2.imwrite(img1_silhouettes_path, img1_silhouettes_to_save)
        return img1_silhouettes

    def loadAndCacheDepth(self, img1_depth_path, img1_depth_path_npy, w, h):
        if os.path.isfile(img1_depth_path_npy):
            try:
                depth = np.load(img1_depth_path_npy)
                if depth.shape[1] != w or depth.shape[0] != h:
                    depth = MultimodalPatchesDataset.loadAndSaveDepth(img1_depth_path, img1_depth_path_npy, w, h)
                return depth
            except Exception:
                return MultimodalPatchesDataset.loadAndSaveDepth(img1_depth_path, img1_depth_path_npy, w, h)
        else:
            return MultimodalPatchesDataset.loadAndSaveDepth(img1_depth_path, img1_depth_path_npy, w, h)

    def loadAndCacheNormals(self, img1_normals_path, img1_depth, img1_depth_path, img1_depth_path_npy, w, h):
        if os.path.isfile(img1_normals_path):
            try:
                normals = cv2.imread(img1_normals_path, -1)
                normals = np.flip(normals, 2) / 65535.0
                if normals.shape[1] != w or normals.shape[0] != h:
                    return self.createAndSaveNormals(img1_depth, img1_depth_path, img1_depth_path_npy, img1_normals_path)
                return normals
            except Exception:
                return self.createAndSaveNormals(img1_depth, img1_depth_path, img1_depth_path_npy, img1_normals_path)
        else:
            return self.createAndSaveNormals(img1_depth, img1_depth_path, img1_depth_path_npy, img1_normals_path)

    def loadAndCacheSilhouettes(self, img1_silhouettes_path, img1_depth, img1_depth_path, img1_depth_path_npy, w, h):
        if os.path.isfile(img1_silhouettes_path):
            try:
                silhouettes = cv2.imread(img1_silhouettes_path, -1)
                silhouettes = np.flip(silhouettes, 2) / 255.0
                if silhouettes.shape[1] != w or silhouettes.shape[0] != h:
                    return self.createAndSaveSilhouettes(img1_depth, img1_depth_path, img1_depth_path_npy, img1_silhouettes_path)
                return silhouettes
            except Exception:
                return self.createAndSaveSilhouettes(img1_depth, img1_depth_path, img1_depth_path_npy, img1_silhouettes_path)
        else:
            return self.createAndSaveSilhouettes(img1_depth, img1_depth_path, img1_depth_path_npy, img1_silhouettes_path)

    @staticmethod
    def normalsFromDepth(depth):
        K = 1
        orig_shape = depth.shape
        depth_pad = np.pad(depth, ((K, K), (K, K), (0, 0)), 'constant')
        dzdx = (depth_pad[1:-1, 2:] - depth_pad[1:-1, 0:-2]) / 2.0
        dzdy = (depth_pad[2:, 1:-1] - depth_pad[0:-2, 1:-1]) / 2.0
        d = np.concatenate([dzdx, dzdy, np.ones(orig_shape)], axis=2)
        norm = np.sqrt(np.sum(np.power(d, 2), axis=2))[:, :, None]
        n = d / norm
        n = (n + 1.0) / (2.0) # so that it can be saved as an image
        return n

    @staticmethod
    def silhouettesFromDepth(depth):
        depth_log = np.log(depth)
        Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
        Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

        Ix = ndimage.convolve(depth_log[:, :, 0], Kx)
        Iy = ndimage.convolve(depth_log[:, :, 0], Ky)

        G = np.hypot(Ix, Iy)
        stdg = np.std(G)
        G[G > stdg] = stdg
        #G = (G - np.mean(G)) / np.std(G)
        ming = np.min(G)
        G = 1.0 - (G - ming) / (G.max() - ming + 1e-10)
        return G[:, :, None]

    def loadImages(self, ext = ".png"):
        img1_base = os.path.splitext(self.info['img1_name'])[0]
        img2_base = os.path.splitext(self.info['img2_name'])[0]
        #print("IMG BASE", img2_base)
        MV1_name = os.path.join(self.images_path, img1_base + "_modelview.txt")
        MV2_name = os.path.join(self.images_path, img2_base.replace("_texture", "") + "_modelview.txt")

        P1_name = os.path.join(self.images_path, img1_base + "_projection.txt")
        P2_name = os.path.join(self.images_path, img2_base.replace("_texture", "") + "_projection.txt")

        self.RT1 = (FUtil.loadMatrixFromFile(MV1_name))
        self.RT2 = (FUtil.loadMatrixFromFile(MV2_name))

        self.P1 = FUtil.loadMatrixFromFile(P1_name)
        self.P2 = FUtil.loadMatrixFromFile(P2_name)

        # photograph
        if self.render_only:
            img1_path = os.path.join(self.images_path, img1_base + "_texture" + ext)
        else:
            img1_path = os.path.join(self.images_path, img1_base + ext)
        # render
        img2_path = os.path.join(self.images_path, img2_base + ext)

        # render
        img1_render_path = os.path.join(self.images_path, img1_base + "_texture" + ext)
        if not self.render_only:
            # remote _texture to convert the path to photo
            img2_base = img2_base.replace("_texture", "")
        img2_photo_path = os.path.join(self.images_path, img2_base + ext)

        # start = timer()
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        #print(img1_path)
        img1 = np.flip(img1, 2)  # was needed with cv2.imread
        img2 = np.flip(img2, 2)  # was needed with cv2.imread

        img1_render = cv2.imread(img1_render_path)
        img2_photo = cv2.imread(img2_photo_path)
        img1_render = np.flip(img1_render, 2)  # was needed with cv2.imread
        img2_photo = np.flip(img2_photo, 2)  # was needed with cv2.imread

        #if self.photo_jitter:
        # apply gamma to simulate different camera response functions
        #gamma = np.random.uniform(0.8, 1.1)
        #img1 = np.clip(img1 ** (1 / gamma), 0, 255)
        #gamma = np.random.uniform(0.8, 1.1)
        #img2_photo = np.clip(img2_photo ** (1 / gamma), 0, 255)

        w1, h1, s1 = getSizeFOV(self.info['img1_shape'][1], self.info['img1_shape'][0], self.info['fov_1'], maxres=self.left_maxres)
        w2, h2, s2 = getSizeFOV(self.info['img2_shape'][1], self.info['img2_shape'][0], self.info['fov_2'], maxres=self.right_maxres)

        img1 = cv2.resize(img1, (w1, h1), interpolation=cv2.INTER_AREA)
        img2 = cv2.resize(img2, (w2, h2), interpolation=cv2.INTER_AREA)
        img1_render = cv2.resize(img1_render, (w1, h1), interpolation=cv2.INTER_AREA)

        if self.color_jitter:
            # jitter color of rendered images
            img2 = self.transform(torch.from_numpy(img2.transpose((2, 0, 1)))).numpy().transpose((1, 2, 0))
            img1_render = self.transform(torch.from_numpy(img1_render.transpose((2, 0, 1)))).numpy().transpose((1, 2, 0))

        img2_photo = cv2.resize(img2_photo, (w2, h2), interpolation=cv2.INTER_AREA)

        # print("image loaded in: ", timer() - start)

        img1_depth_path_npy = os.path.join(self.images_path, img1_base + "_texture_depth.npy")
        img1_depth_path = os.path.join(self.images_path, img1_base + "_texture_depth.txt.gz")
        img2_depth_path_npy = os.path.join(self.images_path, img2_base + "_texture_depth.npy")
        img2_depth_path = os.path.join(self.images_path, img2_base + "_texture_depth.txt.gz")
        img1_normals_path = os.path.join(self.images_path, img1_base + "_texture_normals.png")
        img2_normals_path = os.path.join(self.images_path, img2_base + "_texture_normals.png")
        img1_silhouettes_path = os.path.join(self.images_path, img1_base + "_texture_silhouettes.jpg")
        img2_silhouettes_path = os.path.join(self.images_path, img2_base + "_texture_silhouettes.jpg")

        img1_depth = None
        img2_depth = None
        img1_normals = None
        img2_normals = None

        if self.use_depth:
            # load depths
            img1_depth = self.loadAndCacheDepth(img1_depth_path, img1_depth_path_npy, w1, h1)
            img2_depth = self.loadAndCacheDepth(img2_depth_path, img2_depth_path_npy, w2, h2)

            # divide by max depth so that we feed comparable data to the network
            img1_render = np.concatenate([img1_render, img1_depth / 500000.0], axis=2)
            img2 = np.concatenate([img2, img2_depth / 500000.0], axis=2)

        if self.use_normals:
            img1_normals = self.loadAndCacheNormals(img1_normals_path, img1_depth, img1_depth_path, img1_depth_path_npy, w1, h1)
            img2_normals = self.loadAndCacheNormals(img2_normals_path, img2_depth, img2_depth_path, img2_depth_path_npy, w2, h2)

            img1_render = np.concatenate([img1_render, img1_normals], axis=2)
            img2 = np.concatenate([img2, img2_normals], axis=2)

        if self.use_silhouettes:
            img1_silhouettes = self.loadAndCacheSilhouettes(img1_silhouettes_path, img1_depth, img1_depth_path, img1_depth_path_npy, w1, h1)
            img2_silhouettes = self.loadAndCacheSilhouettes(img2_silhouettes_path, img2_depth, img2_depth_path, img2_depth_path_npy, w2, h2)

            img1_render = np.concatenate([img1_render, img1_silhouettes], axis=2)
            img2 = np.concatenate([img2, img2_silhouettes], axis=2)

        #project 3D coords to first view for distance calculations
        v1_d = self.info['coords3d_1']
        v2_d = self.info['coords3d_2']
        # distances in 3D for negative patch selection (not used anymore so that validation loss is comparable throughout different experiments)
        #self.dists = np.linalg.norm(v1_d[:, None] - v2_d, axis=2)

        v1_1_2d = project(v1_d, h1, w1, self.RT1, self.P1)
        v2_2_2d = project(v2_d, h2, w2, self.RT2, self.P2)
        self.v2_2_2d = v2_2_2d
        self.v2_1_2d = project(v2_d, h1, w1, self.RT1, self.P1)
        self.v1_2_2d = project(v1_d, h2, w2, self.RT2, self.P2)
        #distances in 2D for negative patch selection (used always)
        self.dists = np.linalg.norm(v1_1_2d[:, None] - self.v2_1_2d, axis=2)

        # patchsize = self.patch_radius
        # sel1 = np.logical_and(v1_1_2d > patchsize, v1_1_2d < (np.array([w1, h1]) - patchsize))
        # sel1 = np.logical_and(sel1[:, 0], sel1[:, 1])
        # sel2 = np.logical_and(v2_2_2d > patchsize, v2_2_2d < (np.array([w2, h2]) - patchsize))
        # sel2 = np.logical_and(sel2[:, 0], sel2[:, 1])
        # sel = sel1 #np.logical_and(sel1, sel2)
        # v1_1_2d = v1_1_2d[sel]
        # v2_2_2d = v2_2_2d[sel]

        self.info['coords2d_1'] = np.flip(v1_1_2d, axis=1) / s1
        self.info['coords2d_2'] = np.flip(v2_2_2d, axis=1) / s2

        self.uniform_randcoords = np.array([np.random.randint(self.patch_radius, self.info['img2_shape'][0] - self.patch_radius, self.numpatches), np.random.randint(self.patch_radius, self.info['img2_shape'][1] - self.patch_radius, self.numpatches)]).astype(np.float).transpose()
        self.uniform_randcoords_out = np.flip(self.uniform_randcoords, axis=1).copy() * s2
        self.dists_neg = np.linalg.norm(self.v1_2_2d[:, None] - self.uniform_randcoords_out, axis=2)

        if self.uniform_negatives and self.dist_type == '3D':
            if img2_depth is None:
                img2_depth = self.loadAndCacheDepth(img2_depth_path, img2_depth_path_npy, w2, h2)
            self.uniform_randcoords_out_3D = unproject(self.uniform_randcoords_out, img2_depth, self.RT2, self.P2)

        return img1, img2, img1_render, img2_photo

    def calculateDensityKernel(self, pt_3d, num_neigh=10, noprogress=True):
        dists, nn_idxs = self.kdt.query(pt_3d, num_neigh)
        densities = []
        for i in tqdm(range(0, pt_3d.shape[0]), disable=noprogress):
            nn_coords = self.all_coords3d[nn_idxs[i], :3]
            density = KernelDensity().fit(nn_coords)
            log_density = density.score(pt_3d[i].reshape(1, -1))
            density = np.exp(log_density)
            densities.append(density)
        densities = np.array(densities).reshape(-1)
        return densities

    def calculateDensityMeanDist(self, pt_3d, num_neigh=10):
        dists, nn_idxs = self.kdt.query(pt_3d, num_neigh)
        return np.mean(dists, axis=1)

    def calculateDensityRadius(self, pt_3d, radius=10.0):
        return self.kdt.query_radius(pt_3d, radius, count_only=True)


    def __getitem__(self, idx):
        #print("getitem", idx)
        try:
            global_start = timer()
            new_item_idx = int(idx / self.numpatches)
            do_select = True
            color_patch = torch.ones([64, 64, 3])
            if self.item_idx != new_item_idx:
                # randomly jitter maxres
                if self.scale_jitter:
                    self.left_maxres = int(np.random.uniform(2048, 8192))
                    self.right_maxres = int(np.random.uniform(2048, 8192))

                self.item_idx = new_item_idx
                item_path = os.path.join(self.dataset_dir,
                                         self.dataset_items[self.item_idx])
                #print("Loading images at: ", item_path)
                info_path = os.path.join(item_path, "info.npy")
                try:
                    self.info = np.load(info_path, encoding='latin1', allow_pickle=True).flatten()[0]
                except Exception as eofe:
                    print("Skipping ", info_path)
                    return None

                #start = timer()
                img1, img2, img1r, img2p = self.loadImages()
                #end = timer()
                #print("Load images took: ", end - start)

                #start = timer()
                #print("generating patches...")

                if self.reject:

                    pt_3d = self.info['coords3d_1'][:, :3]
                    self.num_points = self.calculateDensityRadius(pt_3d, self.query_radius)
                    max = np.max(self.num_points)
                    if self.max_num_points < max:
                        self.max_num_points = max
                    proba = np.random.uniform(0.0, 1.0, (self.numpatches))
                    self.proba_sample = 1.0 - (self.num_points / self.max_num_points)
                    self.select = proba <= self.proba_sample

                start = timer()
                p1, p2 = generatePatchesFast(
                    img1, img2,
                    self.info['img1_shape'], self.info['img2_shape'],
                    self.info['coords2d_1'], self.info['coords2d_2'],
                    self.info['fov_1'], self.info['fov_2'],
                    False, int(self.patch_radius * self.prcoef),
                    self.patch_radius,
                    maxres=self.left_maxres, maxres2=self.right_maxres,
                    needles=self.needles
                )
                p1r, p2p = generatePatchesFast(
                    img1r, img2p,
                    self.info['img1_shape'], self.info['img2_shape'],
                    self.info['coords2d_1'], self.info['coords2d_2'],
                    self.info['fov_1'], self.info['fov_2'],
                    False, int(self.patch_radius * self.prcoef), self.patch_radius,
                    maxres=self.left_maxres, maxres2=self.right_maxres,
                    needles=self.needles
                )
                chann = (3 * (self.needles + 1))
                p1[:, :chann, :, :] = p1[:, :chann, :, :] / 255.0
                p2p[:, :chann, :, :] = p2p[:, :chann, :, :] / 255.0

                if self.uniform_negatives:
                    pneg2r, pneg2p = generatePatchesFast(
                        img2, img2p,
                        self.info['img2_shape'], self.info['img2_shape'],
                        self.uniform_randcoords, self.uniform_randcoords,
                        self.info['fov_2'], self.info['fov_2'],
                        False, self.patch_radius,
                        maxres=self.right_maxres, maxres2=self.right_maxres,
                        needles=self.needles
                    )

                    pneg2p[:, :chann, :, :] = pneg2p[:, :chann, :, :] / 255.0

                if not self.color_jitter:
                    p2[:, :chann, :, :] = p2[:, :chann, :, :] / 255.0
                    p1r[:, :chann, :, :] = p1r[:, :chann, :, :] / 255.0
                    if self.uniform_negatives:
                        pneg2r[:, :chann, :, :] = pneg2r[:, :chann, :, :] / 255.0

                if self.greyscale:

                    p1 = (0.299 * p1[:, 0, :, :]) + (0.587 * p1[:, 1, :, :]) + (0.114 * p1[:, 2, :, :])
                    p1 = p1[:, None, :, :]
                    p2 = (0.299 * p2[:, 0, :, :]) + (0.587 * p2[:, 1, :, :]) + (0.114 * p2[:, 2, :, :])
                    p2 = p2[:, None, :, :]
                    p1r = (0.299 * p1r[:, 0, :, :]) + (0.587 * p1r[:, 1, :, :]) + (0.114 * p1r[:, 2, :, :])
                    p1r = p1r[:, None, :, :]
                    p2p = (0.299 * p2p[:, 0, :, :]) + (0.587 * p2p[:, 1, :, :]) + (0.114 * p2p[:, 2, :, :])
                    p2p = p2p[:, None, :, :]
                    if self.uniform_negatives:
                        pneg2p = (0.299 * pneg2p[:, 0, :, :]) + (0.587 * pneg2p[:, 1, :, :]) + (0.114 * pneg2p[:, 2, :, :])
                        pneg2p = pneg2p[:, None, :, :]
                        pneg2r = (0.299 * pneg2r[:, 0, :, :]) + (0.587 * pneg2r[:, 1, :, :]) + (0.114 * pneg2r[:, 2, :, :])
                        pneg2r = pneg2r[:, None, :, :]

                self.patches_1 = torch.from_numpy(p1)
                self.patches_2 = torch.from_numpy(p2)
                self.patches_1r = torch.from_numpy(p1r)
                self.patches_2p = torch.from_numpy(p2p)
                if self.uniform_negatives:
                    self.patches_2negr = torch.from_numpy(pneg2r)
                    self.patches_2negp = torch.from_numpy(pneg2p)
                else:
                    self.patches_2negr = self.patches_2
                    self.patches_2negp = self.patches_2p

                #start = timer()

                #end = timer()
                #print("generateDists took: ", end - start)

                if not self.reject:
                    if self.dists.shape[0] != self.numpatches or self.dists.shape[1] != self.numpatches:
                        raise RuntimeError("Wrong shape of dists! item: " + item_path + " shape: " + str(self.dists.shape))
                    if not (self.patches_1.shape[0] == self.patches_2.shape[0] == self.numpatches):
                        raise RuntimeError("Wrong shape of patches!")
                    if self.patches_1.shape[1] != self.numch_1:
                        raise RuntimeError("Wrong number of channels in patches_1!")
                    if self.patches_2.shape[1] != self.numch_2:
                        raise RuntimeError("Wrong number of channels in patches_2!")

            patch_idx = idx % self.numpatches
            if self.reject:
                do_select = self.select[patch_idx]

            sample = self.generateSample(patch_idx)
            if self.mode == 'eval':
                color_patch = color_patch * torch.tensor(plt.cm.jet(1.0 - self.proba_sample[patch_idx]))[:3]

            if sample[2].shape[0] != self.numneg:
                raise RuntimeError("Wrong number of generated negatives. Got "
                        + str(sample[2].shape[0]) + ", expected " + str(self.numneg))

            select_arr = np.array([do_select]).astype(int)
            select_byte = torch.from_numpy(select_arr).type(torch.ByteTensor)

            if self.mode == 'eval':
                return sample[0], sample[1], sample[2], sample[3], sample[4], sample[5], sample[6], sample[7], sample[8], color_patch.permute(2,0,1), select_byte, torch.tensor(self.item_idx).int()
            else:
                return sample[0], sample[1], sample[2], sample[3], sample[4], sample[5], sample[6], sample[7], sample[8], select_byte, torch.tensor(self.item_idx).int()

        except Exception as re:
            #print("min dist:", np.min(self.dists), "max dist:", np.max(self.dists), "pos thr:", self.pos_thr * 2.0)
            print("Error in MultimodalPatchesDataset", re)
            return None


def plotSample(dataset, i):
    p1, p2, neg = dataset[i]
    plt.figure()
    plt.subplot(2, 3, 1)
    plt.imshow(p1)
    plt.subplot(2, 3, 2)
    plt.imshow(p2)
    for j in range(0, neg.shape[0]):
        plt.subplot(2, 3, 4 + j)
        plt.imshow(neg[j])
    plt.show()


def plotSampleBatched(ibatch, sample_batched, dist_type):
    c1 = sample_batched[6].numpy()
    c2 = sample_batched[7].numpy()
    cneg = sample_batched[8].numpy()

    if dist_type == '2D':
        plt.figure()
        ax = plt.axes([0,0,1,1], frameon=False)
        ax.set_axis_off()
        ax.set_xlim(0,np.max(c1[:, 0]))
        ymax = np.max(c1[:, 1])
        ax.set_ylim(0, ymax)
        pos_patches = np.transpose(sample_batched[0][:, :3].numpy(), [0, 2, 3, 1])
        for idx in range(0, pos_patches.shape[0]):
            plt.imshow(pos_patches[idx], extent=(c1[idx, 0] - 32, c1[idx, 0] + 32, ymax - (c1[idx, 1] + 32), ymax - (c1[idx, 1] - 32)))

        plt.figure()
        ax = plt.axes([0,0,1,1], frameon=False)
        ax.set_axis_off()
        ax.set_xlim(0,np.max(c2[:, 0]))
        ymax = np.max(c2[:, 1])
        ax.set_ylim(0, ymax)
        pos_patches = np.transpose(sample_batched[1][:, :3].numpy(), [0, 2, 3, 1])
        for idx in range(0, pos_patches.shape[0]):
            plt.imshow(pos_patches[idx], extent=(c2[idx, 0] - 32, c2[idx, 0] + 32, ymax - (c2[idx, 1] + 32), ymax - (c2[idx, 1] - 32)))

        plt.figure()
        ax = plt.axes([0,0,1,1], frameon=False)
        ax.set_axis_off()
        ax.set_xlim(0,np.max(cneg[:, 0]))
        ymax = np.max(cneg[:, 1])
        ax.set_ylim(0, ymax)
        neg_patches = np.transpose(sample_batched[2][:, 0, :3].numpy(), [0, 2, 3, 1])
        for idx in range(0, neg_patches.shape[0]):
            plt.imshow(neg_patches[idx], extent=(cneg[idx, 0] - 32, cneg[idx, 0] + 32, ymax - (cneg[idx, 1] + 32), ymax - (cneg[idx, 1] - 32)))
        plt.show()
    else:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(c1[:, 0], c1[:, 1], c1[:, 2], marker='o')
        ax.scatter(c2[:, 0], c2[:, 1], c2[:, 2], marker='^')
        neg_sel = np.linalg.norm(cneg[:, :3], axis=1) < 100000 #filter out too big distances
        ax.scatter(cneg[neg_sel, 0], cneg[neg_sel, 1], cneg[neg_sel, 2], marker='x', c='red')

    neg = sample_batched[2][:, :, :3]
    nneg = neg.shape[1]
    grid1 = utils.make_grid(sample_batched[0][:, :3], nrow=18)
    plt.figure("Positive Matches")
    plt.subplot(1, 2 + nneg, 1)
    plt.imshow(grid1.numpy().transpose((1, 2, 0)))
    #print("depth", sample_batched[1][:, :3]) #4:7
    grid2 = utils.make_grid(sample_batched[1][:, :3], nrow=18) #4:7
    plt.subplot(1, 2 + nneg, 2)
    plt.imshow(grid2.numpy().transpose((1, 2, 0)))
    for idx in range(0, nneg):
        plt.subplot(nneg, 2 + nneg, 3 + idx)
        grid3 = utils.make_grid(neg[:, idx, :], nrow=18)
        plt.imshow(grid3.numpy().transpose((1, 2, 0)))
    plt.show()


def loadListFile(filename):
    listfile = []
    if (os.path.isfile(filename)):
        with open(filename) as f:
            listfile = [d.strip() for d in f.readlines()]
    else:
        raise RuntimeError("The file " + filename + " does not exist.")
    return listfile


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    #dset = "/home/ibrejcha/data/switzerland_wallis_30km_maximg/patches_dataset_all"
    dset = "/mnt/matylda1/ibrejcha/adobe_intern/data/switzerland_wallis_30km_maximg/patches_dataset_all_3000_val" #smaller dataset
    #dset = '/mnt/matylda1/ibrejcha/data/matching/input/geoPose3K_trainval/rendered_dataset_random_sample_20_200_10/patches_dataset_all'
    #dset = "/Users/janbrejcha/Downloads/test_patches_6"
    images_path = "/home/ibrejcha/data/switzerland_wallis_30km_maximg/final_dataset/real"
    #images_path = '/mnt/matylda1/ibrejcha/data/matching/input/geoPose3K_trainval/rendered_dataset_random_sample_20_200_10/real'
    #images_path = "/mnt/matylda1/ibrejcha/adobe_intern/data/switzerland_wallis_30km_maximg/final_dataset/real"
    train_path = os.path.join(dset, "val.txt")
    train = loadListFile(train_path)
    #random.shuffle(train)
    dataset = MultimodalPatchesDataset(dset, images_path, train, numneg=1, pos_thr=1.0, use_depth=False, use_normals=False, reject=False, dist_type='3D', rejection_radius=10, photo_jitter=False, uniform_negatives=True)
    dataloader = DataLoader(dataset, batch_size=300, shuffle=False, num_workers=8)
    for i_batch, sample_batched in tqdm(enumerate(dataloader)):
        if i_batch % 100 == 0:
            plotSampleBatched(i_batch, sample_batched, dataset.dist_type)
            print(i_batch)
    #for i in range(0, 100):
    #   dataset[i]

    #validate if dataset is correct
    # subdirs = [f for f in os.listdir(dset) if os.path.isdir(os.path.join(dset,f))]
    # with open(os.path.join(os.getcwd(), "correct_items.txt"), "w") as f:
    #     for subdir in tqdm(subdirs):
    #         flist = [subdir]
    #         correct = True
    #         try:
    #             dataset = MultimodalPatchesDataset(dset, images_path, flist, numneg=1, pos_thr=500.0)
    #             dataloader = DataLoader(dataset, batch_size=300, shuffle=False,
    #                                 num_workers=8)
    #             for i_batch, sample_batched in enumerate(dataloader):
    #                 #pass
    #                 plotSampleBatched(i_batch, sample_batched)
    #                 #print(i_batch)
    #         except Exception as e:
    #             print(e)
    #             correct = False
    #         if correct:
    #             print("correct: " + subdir)
    #             f.write(subdir + "\n")

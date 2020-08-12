# @Author: Jan Brejcha <ibrejcha>
# @Date:   2020-08-11T17:49:09+02:00
# @Email:  ibrejcha@fit.vutbr.cz; brejchaja@gmail.com
# @Project: LandscapeAR
# @Last modified by:   ibrejcha
# @Last modified time: 2020-08-12T11:45:15+02:00
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

import os
# import time
import numpy as np
import cv2
import torch
import pymap3d as pm3d
import exiftool
import matplotlib.pyplot as plt

# our code
from pose_estimation import FUtil
from pyquaternion import Quaternion
from tqdm import tqdm
from pose_estimation.patchSamplingDepth import unproject, project, loadDepth, findIndicesOfCorresponding3DPointsWithDist
from trainPatchesDescriptors import plotMatches
from training.MultimodalPatchesDataset import MultimodalPatchesDataset

from torch.multiprocessing import Pool
cv2.setNumThreads(0)
plt.ioff()


class Matcher(object):
    def __init__(
        self, input_dir, image_list_file,
        database, suffix, recompute, cuda=True, num_processes=1
    ):
        super(Matcher, self).__init__()
        self.input_dir = input_dir
        self.image_list_file = image_list_file
        self.db = database
        self.suffix = suffix
        self.recompute = recompute
        self.cuda = cuda
        self.num_processes = num_processes
        self.image_ids = {}
        # Extend this class, overwrite the match() method and fill
        # self.pairs_to_match with tuples of image IDs of images which shall
        # be matched. Image ID corresponds to the index of a given image
        # in Matcher.image_names.
        # Then call superclass match() method, which will do the matching.
        # Make sure not to add one pair twice.
        self.pairs_to_match = []

        scene_center_filename = os.path.join(self.input_dir, "scene_info.txt")
        if os.path.exists(scene_center_filename):
            self.scene_center = MultimodalPatchesDataset.getSceneCenter(scene_center_filename)
        else:
            print("WARNING: ", scene_center_filename, " NOT FOUND, setting",
                  "scene center to [0, 0, 0].")
            self.scene_center = np.array([0, 0, 0])

        self.loadImageNames()
        self.addImagesToDatabase()

    @staticmethod
    ### This is able to get more inliers, but overall does not work better
    ### and is much slower than the nearest neighbor matching.
    def matchDescriptorsUnique(desc1, desc2, cuda=True):
        desc1 = torch.from_numpy(desc1).float()
        desc2 = torch.from_numpy(desc2).float()

        flip = False
        if desc1.shape[0] > desc2.shape[0]:
            flip = True
            tmp = desc1
            desc1 = desc2
            desc2 = tmp

        if cuda:
            desc1 = desc1.cuda()
            desc2 = desc2.cuda()

        # Divide the problem into blocks of given size and calculate the
        # distance between each block to avoid GPU memory issues
        # for large problems.
        block_size = 15000
        current_size = desc1.shape[0] * desc2.shape[0] * desc1.shape[1]
        maximum_size = block_size * block_size * 512
        if current_size > maximum_size:
            # general implementation that handles arbitrary sizes
            desc1_blocks = (desc1.shape[0] // block_size)
            desc2_blocks = (desc2.shape[0] // block_size)
            dists = []
            for d1b in range(0, desc1_blocks + 1):
                dists_cols = []
                for d2b in range(0, desc2_blocks + 1):
                    d1bl = d1b * block_size
                    d1bu = d1bl + block_size
                    if d1bu > desc1.shape[0]:
                        d1bu = desc1.shape[0]
                    d2bl = d2b * block_size
                    d2bu = d2bl + block_size
                    if d2bu > desc2.shape[0]:
                        d2bu = desc2.shape[0]
                    desc1_sq = torch.sum(
                        torch.mul(desc1[d1bl:d1bu], desc1[d1bl:d1bu]), dim=1
                    )
                    desc2_sq = torch.sum(
                        torch.mul(desc2[d2bl:d2bu], desc2[d2bl:d2bu]), dim=1
                    )
                    mul = torch.mm(desc1[d1bl:d1bu], desc2[d2bl:d2bu].t())
                    dists_block = torch.sqrt(
                        desc1_sq[:, None] + desc2_sq - 2 * mul
                    )
                    dists_cols.append(dists_block.cpu())
                blk = torch.cat(dists_cols, dim=1)
                dists.append(blk)

            dists = torch.cat(dists)
        else:
            # fast implementation which will fit into 8GB of GPU RAM with
            # descriptor dimensionality 512.
            desc1_sq = torch.sum(torch.mul(desc1, desc1), dim=1)
            desc2_sq = torch.sum(torch.mul(desc2, desc2), dim=1)
            mul = torch.mm(desc1, desc2.t())
            dists = torch.sqrt(desc1_sq[:, None] + desc2_sq - 2 * mul)

        #plt.figure()
        #plt.hist(dists.flatten().cpu(), bins=1000)
        #histogram, bins = np.histogram(dists.flatten().detach().cpu().numpy(), bins=1000)
        #print(histogram)
        #hist_amax = np.argmax(histogram)
        #thr = bins[hist_amax]
        #print("found threshold", thr)

        dmax = torch.max(dists) + 1e-8
        good = []
        for idx in tqdm(range(0, desc1.shape[0])):
            m_ind = torch.argmin(dists)
            i = int(m_ind // dists.shape[1])
            j = int(m_ind % dists.shape[1])
            m_dist = dists[i, j].clone()
            dists[i, :] = dmax
            dists[:, j] = dmax

            if m_dist > 1.2:
                break

            if flip:
                match = cv2.DMatch(j, i, m_dist)
            else:
                match = cv2.DMatch(i, j, m_dist)
            good.append(match)

        return good

    @staticmethod
    def matchDescriptors(desc1, desc2, cuda=True):
        desc1 = torch.from_numpy(desc1).float()
        desc2 = torch.from_numpy(desc2).float()

        flip = False
        if desc1.shape[0] > desc2.shape[0]:
            flip = True
            tmp = desc1
            desc1 = desc2
            desc2 = tmp

        if cuda:
            desc1 = desc1.cuda()
            desc2 = desc2.cuda()

        # Divide the problem into blocks of given size and calculate the
        # distance between each block to avoid GPU memory issues
        # for large problems.
        block_size = 10000
        current_size = desc1.shape[0] * desc2.shape[0] * desc1.shape[1]
        maximum_size = block_size * block_size * 512
        if current_size > maximum_size:
            # general implementation that handles arbitrary sizes
            desc1_blocks = (desc1.shape[0] // block_size)
            desc2_blocks = (desc2.shape[0] // block_size)
            dists = []
            for d1b in range(0, desc1_blocks + 1):
                dists_cols = []
                for d2b in range(0, desc2_blocks + 1):
                    d1bl = d1b * block_size
                    d1bu = d1bl + block_size
                    if d1bu > desc1.shape[0]:
                        d1bu = desc1.shape[0]
                    d2bl = d2b * block_size
                    d2bu = d2bl + block_size
                    if d2bu > desc2.shape[0]:
                        d2bu = desc2.shape[0]
                    desc1_sq = torch.sum(
                        torch.mul(desc1[d1bl:d1bu], desc1[d1bl:d1bu]), dim=1
                    )
                    desc2_sq = torch.sum(
                        torch.mul(desc2[d2bl:d2bu], desc2[d2bl:d2bu]), dim=1
                    )
                    mul = torch.mm(desc1[d1bl:d1bu], desc2[d2bl:d2bu].t())
                    dists_block = torch.sqrt(
                        desc1_sq[:, None] + desc2_sq - 2 * mul
                    )
                    dists_cols.append(dists_block.cpu())
                blk = torch.cat(dists_cols, dim=1)
                dists.append(blk)

            dists = torch.cat(dists)
        else:
            # fast implementation which will fit into 8GB of GPU RAM with
            # descriptor dimensionality 512.
            desc1_sq = torch.sum(torch.mul(desc1, desc1), dim=1)
            desc2_sq = torch.sum(torch.mul(desc2, desc2), dim=1)
            mul = torch.mm(desc1, desc2.t())
            dists = torch.sqrt(desc1_sq[:, None] + desc2_sq - 2 * mul)

        dist, indices = torch.min(dists, dim=1)
        dist_cpu = dist.cpu()
        indices_cpu = indices.cpu()
        del dist
        del indices
        del desc1
        del desc2
        del desc1_sq
        del desc2_sq
        good = []

        if flip:
            for idx in range(0, len(dist_cpu)):
                match = cv2.DMatch(indices_cpu[idx], idx, dist_cpu[idx])
                good.append(match)
        else:
            for idx in range(0, len(dist_cpu)):
                match = cv2.DMatch(idx, indices_cpu[idx], dist_cpu[idx])
                good.append(match)

        good, train_idxs = Matcher.bestBuddy(good, flip=flip)

        return good

    @staticmethod
    def bestBuddy(good, flip=False):
        closest = {}
        if flip:
            for m in good:
                if m.queryIdx in closest:
                    closest_match = closest[m.queryIdx]
                    if m.distance < closest_match.distance:
                        closest_match = m
                    closest[m.queryIdx] = closest_match
                else:
                    closest.update({m.queryIdx: m})
            good = []
            train_idxs = []
            for trainIdx, match in closest.items():
                good.append(match)
                train_idxs.append(match.queryIdx)
        else:
            for m in good:
                if m.trainIdx in closest:
                    closest_match = closest[m.trainIdx]
                    if m.distance < closest_match.distance:
                        closest_match = m
                    closest[m.trainIdx] = closest_match
                else:
                    closest.update({m.trainIdx: m})
            good = []
            train_idxs = []
            for trainIdx, match in closest.items():
                good.append(match)
                train_idxs.append(match.trainIdx)

        return good, train_idxs

    def isRender(self, name):
        return "_texture" in name or "render" in name

    def getRenderedImageParams(self, name):
        abs_name = os.path.join(self.input_dir, name)
        img = cv2.imread(abs_name)
        img = cv2.flip(img, 2)

        # add the render camera
        width = img.shape[1]
        height = img.shape[0]

        P, pose = Matcher.loadRenderedCameraParams(abs_name)
        R_corrected = pose[:3, :3]
        t = pose[:3, 3]

        intr = FUtil.projectiveToIntrinsics(P, width, height)
        focal = intr[0, 0]
        cx = abs(intr[0, 2])
        cy = abs(intr[1, 2])
        params1 = np.array((focal, cx, cy, 0.0))  # radial distortion 0

        cam_center = self.scene_center + np.dot(-R_corrected.transpose(), t)

        quat = Quaternion(matrix=R_corrected)
        rot = np.array([quat.w, quat.x, quat.y, quat.z])

        return width, height, params1, rot, t, cam_center

    def addImagesToDatabase(self):
        sparse_model_dir = os.path.join(
            self.input_dir, "sparse_model" + self.suffix
        )
        if not os.path.exists(sparse_model_dir):
            os.makedirs(sparse_model_dir)
        images_filename = os.path.join(sparse_model_dir, "images.txt")
        cameras_filename = os.path.join(sparse_model_dir, "cameras.txt")
        points3D_filename = os.path.join(sparse_model_dir, "points3D.txt")
        geo_filename = os.path.join(
            self.input_dir, "georegistration" + self.suffix + ".txt"
        )
        images_file = open(images_filename, 'w')
        cameras_file = open(cameras_filename, 'w')
        geo_file = open(geo_filename, 'w')

        # should be left empty
        points3D_file = open(points3D_filename, 'w')
        points3D_file.close()
        for name in self.image_names:
            abs_name = os.path.join(self.input_dir, name)
            rot = np.zeros(4)
            t = np.zeros(3)
            cam_center = np.zeros(3)
            if self.isRender(name):
                (
                    width, height, params1, rot, t, cam_center
                ) = self.getRenderedImageParams(name)
            else:
                # add normal image
                img = cv2.imread(abs_name)
                img = cv2.flip(img, 2)
                width = img.shape[1]
                height = img.shape[0]
                # initialize unknown intrinsics to some reasonable values
                # radial distortion 0
                params1 = np.array((width, width/2.0, height/2.0, 0.0))

            cam_id = self.db.add_camera(2, width, height, params1)

            # add image to db
            image_id1 = self.db.add_image(name, cam_id, rot, t)
            self.image_ids.update(
                {self.image_names.index(name): image_id1}
            )

            # add keypoints to image
            fea = Matcher.loadFeatures(abs_name)
            # format: [x, y, scale]
            keypoints = fea['keypoints']
            numkp = keypoints.shape[0]
            # format: [x, y, 0 (theta), scale]
            keypoints1 = np.zeros([numkp, 4])
            keypoints1[:, :2] = keypoints[:, :2]
            keypoints1[:, 3] = keypoints[:, 2]
            self.db.add_keypoints(image_id1, keypoints1)

            if self.isRender(name):
                images_file.write(
                    str(image_id1) + " "
                    + str(rot[0]) + " " + str(rot[1]) + " "
                    + str(rot[2]) + " " + str(rot[3]) + " "
                    + str(t[0]) + " " + str(t[1]) + " " + str(t[2]) + " "
                    + str(cam_id) + " " + name + "\n\n"
                )
                cameras_file.write(
                    str(cam_id) + " SIMPLE_RADIAL "
                    + str(width) + " " + str(height) + " "
                    + str(params1[0]) + " " + str(params1[1]) + " "
                    + str(params1[2]) + " " + str(params1[3]) + "\n"
                )

                geo_file.write(
                    name + " " + str(cam_center[0]) + " "
                    + str(cam_center[1]) + " " + str(cam_center[2]) + "\n"
                )

        images_file.close()
        cameras_file.close()
        geo_file.close()

    def loadImageNames(self):
        with open(self.image_list_file, 'r') as f:
            # create tuple so that it is immutable
            self.image_names = tuple([
                l.strip() for l in f.readlines()
            ])

    def match(self):
        pool = Pool(self.num_processes)
        results = []

        total = len(self.pairs_to_match)
        counter = 0

        matches_pairs_fname = os.path.join(
            self.input_dir, "matches_pairs" + self.suffix + ".txt"
        )
        mp_file = open(matches_pairs_fname, 'w')

        for pair in tqdm(self.pairs_to_match):
            orig_name1 = self.image_names[pair[0]]
            orig_name2 = self.image_names[pair[1]]
            name1 = os.path.join(self.input_dir, orig_name1)
            name2 = os.path.join(self.input_dir, orig_name2)
            image_id1 = self.image_ids[pair[0]]
            image_id2 = self.image_ids[pair[1]]

            if self.cuda:
                # use single process
                data = self.matchPairCached(
                    orig_name1, orig_name2, image_id1,
                    image_id2, self.recompute, self.cuda
                )
                self.db.add_matches(data[0], data[1], data[2])
                mp_file.write(data[3] + " " + data[4] + "\n")
                counter += 1
                if counter % 50000 == 0:
                    counter = 0
                    self.db.commit()
                # print("Matched", counter, "out of", total)
            else:
                # use multiple processes
                res = pool.apply_async(
                    self.matchPairCached,
                    (
                        orig_name1, orig_name2, image_id1,
                        image_id2, self.recompute, self.cuda
                    )
                )
                results.append(res)
        if not self.cuda:
            # if more processes were used, collect the results
            counter = 0
            for res in tqdm(results):
                data = res.get()
                if data is None:
                    continue
                self.db.add_matches(data[0], data[1], data[2])
                name1 = data[3]
                name2 = data[4]
                mp_file.write(name1 + " " + name2 + "\n")
                counter += 1
                if counter % 50000 == 0:
                    counter = 0
                    self.db.commit()
        # be sure to commit everything at the end
        self.db.commit()
        mp_file.close()

    @staticmethod
    def loadFeatures(name):
        feature_file = name + ".npz"
        fea = np.load(feature_file)
        return fea

    @staticmethod
    def matchPair(
        name1, name2, image_id1, image_id2,
        recompute, cuda, matches_path
    ):
        fea1 = Matcher.loadFeatures(name1)
        fea2 = Matcher.loadFeatures(name2)
        # start = time.time()
        good = Matcher.matchDescriptors(
            fea1['descriptors'], fea2['descriptors'], cuda
        )
        # print("Matching elapsed: ", time.time() - start)

        matches = np.zeros([len(good), 2], dtype=np.int32)
        for idx in range(0, len(good)):
            matches[idx, 0] = good[idx].queryIdx
            matches[idx, 1] = good[idx].trainIdx

        np.savez(matches_path, matches=matches)
        return matches

    @staticmethod
    def loadRenderedCameraParams(name):
        img_path = name.replace("_texture", "")
        P_name = os.path.splitext(img_path)[0] + "_projection.txt"
        MV_name = os.path.splitext(img_path)[0] + "_modelview.txt"
        if not os.path.isfile(MV_name):
            MV_name = os.path.splitext(img_path)[0] + "_pose.txt"
        P = FUtil.loadMatrixFromFile(P_name)
        pose = FUtil.loadMatrixFromFile(MV_name)

        R = pose[:3, :3]
        t = pose[:3, 3]
        R[1:3, :] = -R[1:3, :]
        t[1:3] = -t[1:3]

        corr_R = ((3 * np.eye(3)) - (np.dot(R, R.transpose()))) / 2.0
        R_corrected = np.dot(corr_R, R)
        pose[:3, :3] = R_corrected
        return P, pose

    @staticmethod
    def loadRenderedCameraParamsNoNeg(name):
        img_path = name.replace("_texture", "")
        P_name = os.path.splitext(img_path)[0] + "_projection.txt"
        MV_name = os.path.splitext(img_path)[0] + "_modelview.txt"
        if not os.path.isfile(MV_name):
            MV_name = os.path.splitext(img_path)[0] + "_pose.txt"
        P = FUtil.loadMatrixFromFile(P_name)
        pose = FUtil.loadMatrixFromFile(MV_name)

        R = pose[:3, :3]

        corr_R = ((3 * np.eye(3)) - (np.dot(R, R.transpose()))) / 2.0
        R_corrected = np.dot(corr_R, R)
        pose[:3, :3] = R_corrected
        return P, pose

    @staticmethod
    def loadRenderedCameraDepth(name):
        depth_filename = os.path.splitext(name)[0] + "_depth.txt.gz"
        return loadDepth(depth_filename)

    @staticmethod
    def getKeypoints3D(name1):
        fea1 = Matcher.loadFeatures(name1)
        # keep 2D points only
        kp1 = fea1['keypoints'][:, :2]
        P1, MV1 = Matcher.loadRenderedCameraParamsNoNeg(name1)
        D1 = Matcher.loadRenderedCameraDepth(name1)
        # unproject keypoints from the left camera to 3D
        kp1_3D = unproject(kp1, D1, MV1, P1)
        return kp1_3D

    @staticmethod
    def matchRenderedPair(
        name1, name2, image_id1, image_id2,
        recompute, cuda, matches_path, max_reproj_error=40
    ):
        #print("Matching rendered pair:", name1, name2)

        kp1_3D_filename = os.path.splitext(name1)[0] + "_kp_3D.npz"
        kp2_3D_filename = os.path.splitext(name2)[0] + "_kp_3D.npz"

        if not recompute and os.path.exists(kp1_3D_filename):
            kp1_3D = np.load(kp1_3D_filename)['kp_3D']
        else:
            kp1_3D = Matcher.getKeypoints3D(name1)
            np.savez(kp1_3D_filename, kp_3D=kp1_3D)

        if not recompute and os.path.exists(kp2_3D_filename):
            kp2_3D = np.load(kp2_3D_filename)['kp_3D']
        else:
            kp2_3D = Matcher.getKeypoints3D(name2)
            np.savez(kp2_3D_filename, kp_3D=kp2_3D)

        v1_idx, v2_idx, dist = findIndicesOfCorresponding3DPointsWithDist(
            kp1_3D[:, :3], kp2_3D[:, :3], threshold=max_reproj_error
        )

        good = []
        for idx in range(0, len(v1_idx)):
            good.append(cv2.DMatch(v1_idx[idx], v2_idx[idx], dist[idx]))
        good, _ = Matcher.bestBuddy(good, flip=True)

        matches = np.zeros([len(good), 2], dtype=np.int32)
        for idx in range(0, len(good)):
            matches[idx, 0] = good[idx].queryIdx
            matches[idx, 1] = good[idx].trainIdx

        np.savez(matches_path, matches=matches)
        return matches

    def matchPairCached(
        self, orig_name1, orig_name2, image_id1, image_id2,
        recompute=False, cuda=True
    ):
        name1 = os.path.join(self.input_dir, orig_name1)
        name2 = os.path.join(self.input_dir, orig_name2)

        both_renders = False
        if (self.isRender(orig_name1) and self.isRender(orig_name2)):
            both_renders = True

        fname1 = os.path.splitext(os.path.basename(name1))[0]
        fname2 = os.path.splitext(os.path.basename(name2))[0]
        matches_path = os.path.join(
            os.path.dirname(name1), "matches_" + fname1 + "_" + fname2 + ".npz"
        )

        matches_exists = os.path.isfile(matches_path)
        if not matches_exists or (matches_exists and recompute):
            if both_renders:
                matches = Matcher.matchRenderedPair(
                    name1, name2, image_id1, image_id2,
                    recompute, cuda, matches_path
                )
            else:
                matches = Matcher.matchPair(
                    name1, name2, image_id1, image_id2,
                    recompute, cuda, matches_path
                )
        else:
            try:
                m = np.load(matches_path)
                matches_exists = 'matches' in m.files
                matches = m['matches']
            except Exception:
                print("Could not load matches: ", matches_path, "recomputing.")
                if both_renders:
                    matches = Matcher.matchRenderedPair(
                        name1, name2, image_id1, image_id2,
                        recompute, cuda, matches_path
                    )
                else:
                    matches = Matcher.matchPair(
                        name1, name2, image_id1, image_id2,
                        recompute, cuda, matches_path
                    )

        return [image_id1, image_id2, matches, orig_name1, orig_name2]


class ExhaustiveMatcher(Matcher):
    def __init__(
        self, input_dir, image_list_file,
        database, suffix, recompute=False, cuda=True, num_processes=1,
        disable_photo2photo_matching=False
    ):
        super(ExhaustiveMatcher, self).__init__(
            input_dir, image_list_file, database, suffix, recompute, cuda,
            num_processes
        )
        self.disable_photo2photo_matching = disable_photo2photo_matching

    def match(self):
        for id1 in range(0, len(self.image_names)):
            for id2 in range(0, len(self.image_names)):
                image_id1 = self.image_ids[id1]
                image_id2 = self.image_ids[id2]
                if image_id1 >= image_id2:
                    # this pair is already matched
                    continue
                if self.disable_photo2photo_matching:
                    id1_is_photo = (not self.isRender(self.image_names[id1]))
                    id2_is_photo = (not self.isRender(self.image_names[id2]))
                    if id1_is_photo and id2_is_photo:
                        continue
                self.pairs_to_match.append((id1, id2))
        super().match()


class SpatialMatcher(Matcher):
    def __init__(
        self, input_dir, image_list_file,
        database, suffix, recompute=False, cuda=True, num_processes=1,
        max_distance=5, disable_photo2photo_matching=False
    ):
        super(SpatialMatcher, self).__init__(
            input_dir, image_list_file, database, suffix, recompute, cuda,
            num_processes
        )
        self.disable_photo2photo_matching = disable_photo2photo_matching
        self.max_distance = max_distance
        print("Loading positional info...")
        self.loadPositionalInfo()
        self.calculateDistances()

    def loadPositionalInfo(self):
        num_imgs = len(self.image_names)
        image_coords_path = os.path.join(
            self.input_dir, "image_coords_ecef" + self.suffix + ".npz"
        )
        recompute = True
        if os.path.exists(image_coords_path):
            positional_info = np.load(image_coords_path, allow_pickle=True)
            self.image_ecef = positional_info['image_ecef']
            self.image_have_rot = positional_info['image_have_rot']
            self.image_rot = positional_info['image_rot']
            if self.image_ecef.shape[0] == num_imgs:
                recompute = self.recompute

        if recompute:
            self.image_ecef = np.zeros([num_imgs, 3])
            self.image_have_rot = np.zeros([num_imgs, 1])
            self.image_rot = [
                Quaternion(axis=[1, 0, 0], angle=0) for i in range(0, num_imgs)
            ]
            for id in tqdm(range(0, num_imgs)):
                name = self.image_names[id]
                abs_name = os.path.join(self.input_dir, name)
                if self.isRender(name):
                    # load the modelview matrix, get cam center,
                    # add scene center
                    # and convert to WGS84 coordinates, then convert back to
                    # ECEF with altitude=0 since we usually don't have
                    # precise altitude for real photographs.
                    img_path = abs_name.replace("_texture", "")
                    MV_name = os.path.splitext(img_path)[0] + "_modelview.txt"
                    if not os.path.isfile(MV_name):
                        MV_name = os.path.splitext(img_path)[0] + "_pose.txt"
                    pose = FUtil.loadMatrixFromFile(MV_name)
                    R = pose[:3, :3]
                    corr_R = (
                        ((3 * np.eye(3)) - (np.dot(R, R.transpose()))) / 2.0
                    )
                    R_corrected = np.dot(corr_R, R)
                    self.image_rot[id] = Quaternion(matrix=R_corrected)
                    self.image_have_rot[id] = 1
                    C = np.dot(-R.transpose(), pose[:3, 3]) + self.scene_center
                    lat, lon, alt = pm3d.ecef.ecef2geodetic(C[0], C[1], C[2])
                    #print("rendered lat lon alt: ", lat, lon, alt)
                    x, y, z = pm3d.ecef.geodetic2ecef(lat, lon, 0)
                else:
                    # load GPS from the photograph
                    x, y, z = (0, 0, 0)
                    lat, lon, alt = (-1, -1, -1)
                    with exiftool.ExifTool() as et:
                        res = et.execute_json(
                            "-n", "-GPSLatitude", "-GPSLatitudeRef", "-GPSLongitudeRef", "-GPSLongitude",
                            "-GPSAltitude",
                            abs_name
                        )
                        if (
                            ('EXIF:GPSLatitude' in res[0])
                            and ('EXIF:GPSLongitude' in res[0])
                            and ('EXIF:GPSLatitudeRef' in res[0])
                            and ('EXIF:GPSLongitudeRef' in res[0])
                        ):
                            lat = res[0]['EXIF:GPSLatitude']
                            lon = res[0]['EXIF:GPSLongitude']
                            lat_ref = res[0]['EXIF:GPSLatitudeRef']
                            lon_ref = res[0]['EXIF:GPSLongitudeRef']
                            if lat_ref == 'S':
                                lat = -lat
                            if lon_ref == 'W':
                                lon = -lon
                            # if 'EXIF:GPSAltitude' in res[0]:
                            # alt = res[0]['EXIF:GPSAltitude']
                            #print("photo lat lon lat_ref lon_ref: ", lat, lon, lat_ref, lon_ref)
                            try:
                                x, y, z = pm3d.ecef.geodetic2ecef(lat, lon, 0)
                            except ValueError as ve:
                                print(
                                    "Unable to use GPS from the image", name,
                                    "reason:", ve, "resetting the position \
                                    to unknown."
                                )
                                x, y, z = (0, 0, 0)

                # Set image_ecef to zeros if x,y,z are zeros
                # in order to mask them correctly im match() method.
                if x == y == z == 0:
                    self.image_ecef[id, :] = np.array([x, y, z])

                # coords in km due to double imprecision (/1000)
                self.image_ecef[id, :] = (
                    (np.array([x, y, z]) - self.scene_center) / 1000.0
                )

                np.savez(
                    image_coords_path, image_ecef=self.image_ecef,
                    image_rot=self.image_rot,
                    image_have_rot=self.image_have_rot
                )

    def calculateDistances(self):
        coords1 = self.image_ecef[:, :3]
        coords2 = self.image_ecef[:, :3]
        coords1_sq = np.sum(np.square(coords1), axis=1)
        coords2_sq = np.sum(np.square(coords2), axis=1)
        mul = np.dot(coords1, coords2.transpose())
        self.dists = np.sqrt(
            (coords1_sq[:, None] + coords2_sq - 2 * mul)
        )

    def calculateRotMask(self, rot_thr=np.pi / 3.0):
        both_have_rot = self.image_have_rot[:, None] * self.image_have_rot
        rot_mask = np.logical_not(both_have_rot)
        rot_idx = np.where(both_have_rot)
        for idx in range(0, len(rot_idx[0])):
            id1 = rot_idx[0][idx]
            id2 = rot_idx[1][idx]
            rot1 = self.image_rot[id1]
            rot2 = self.image_rot[id2]
            dist = Quaternion.distance(rot1, rot2)
            if dist < rot_thr:
                rot_mask[id1, id2] = True
        return rot_mask.reshape(rot_mask.shape[0], rot_mask.shape[0])

    def match(self):
        # zero out distances for photos for which no positional info
        # is available
        no_pos = np.all(self.image_ecef == np.array([0, 0, 0]), axis=1)
        no_pos_idx = np.where(no_pos)[0]
        self.dists[no_pos_idx, :] = np.zeros(self.dists.shape[1])
        # select images which have distance lower than the maximum distance
        to_match = self.dists < self.max_distance
        luidx = np.tril_indices(self.dists.shape[0])
        # forbid matching of the same pair twice, and matching a photo
        # with itself
        to_match[luidx] = False

        # account for rotations - match renders with renders only if they
        # point to similar direction
        rot_mask = self.calculateRotMask()
        to_match = np.logical_and(to_match, rot_mask)

        pair_idx = np.where(to_match)
        for idx in range(0, len(pair_idx[0])):
            id1 = pair_idx[0][idx]
            id2 = pair_idx[1][idx]
            if self.disable_photo2photo_matching:
                id1_is_photo = (not self.isRender(self.image_names[id1]))
                id2_is_photo = (not self.isRender(self.image_names[id2]))
                if id1_is_photo and id2_is_photo:
                    continue
            self.pairs_to_match.append((id1, id2))
        super().match()

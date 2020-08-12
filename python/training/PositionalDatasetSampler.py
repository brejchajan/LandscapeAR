# @Author: Jan Brejcha <ibrejcha>
# @Date:   2020-08-11T17:50:13+02:00
# @Email:  ibrejcha@fit.vutbr.cz; brejchaja@gmail.com
# @Project: LandscapeAR
# @Last modified by:   ibrejcha
# @Last modified time: 2020-08-12T11:45:53+02:00
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

from tqdm import tqdm
import os
import numpy as np
from sklearn.neighbors import KDTree

import matplotlib
import matplotlib.pyplot as plt

# Our code
import pose_estimation.FUtil as FUtil
from pose_estimation.patchSamplingDepth import savePointCloudToPly


class PositionalDatasetSampler(object):
    """Samples the pairs in the dataset based on density of camera centers.
       It is intended to be used to generate a subset of training pairs
       which are then fed into the newly created dataset instance for each
       epoch.
    """

    def __init__(self, dataset_dir, images_path, list, radius=500):
        self.dataset_dir = dataset_dir
        self.images_path = images_path
        self.dataset_items = list

        # create list of images
        self.imgs_base = set()
        print("Preparing list of images...")
        for pair in tqdm(self.dataset_items):
            parts = pair.split("-")
            self.imgs_base.update({parts[0], parts[1]})

        # load camera center for each image
        self.all_names = []
        self.all_centers = []
        print("Loading camera info...")
        for image_base in tqdm(self.imgs_base):
            MV_name = image_base + "_modelview.txt"
            MV_path = os.path.join(self.images_path, MV_name)
            RT = FUtil.loadMatrixFromFile(MV_path)
            C = np.dot(-RT[:3, :3].transpose(), RT[:3, 3])
            self.all_names.append(image_base)
            self.all_centers.append(C)
        self.all_centers = np.array(self.all_centers)
        self.kdt = KDTree(self.all_centers, leaf_size=40, metric='euclidean')

        count = self.kdt.query_radius(self.all_centers, radius, count_only=True)
        cnt_max = np.max(count)
        self.max_num_points = cnt_max
        count = count / cnt_max

        density = np.log(count)
        norm = matplotlib.colors.Normalize()
        colors = np.floor(plt.cm.jet(norm(density)) * 255)
        ply_name = "dataset_camera_centers_heatmap_" + str(radius) + ".ply"
        # savePointCloudToPly(self.all_centers, colors[:, :3], ply_name)

        self.name_density = {}
        idx = 0
        for name in self.all_names:
            self.name_density.update({name: count[idx]})
            idx += 1

    def sample(self):
        """Samples a subset of camera pairs from the original dataset.
           Rejects camera pairs based on the density of camera centers.
        """
        print("Sampling dataset positions...")
        selected = []
        for pair in tqdm(self.dataset_items):
            parts = pair.split("-")
            left = parts[0]
            right = parts[1]
            density_left = self.name_density[left]
            density_right = self.name_density[right]

            proba = np.random.uniform(0.0, 1.0)
            proba_sample_left = 1.0 - density_left
            proba_sample_right = 1.0 - density_right
            select = proba <= proba_sample_left and proba <= proba_sample_right
            if (select):
                selected.append(pair)
        return selected

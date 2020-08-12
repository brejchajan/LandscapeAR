# @Author: Jan Brejcha <ibrejcha>
# @Date:   2020-08-11T17:50:11+02:00
# @Email:  ibrejcha@fit.vutbr.cz; brejchaja@gmail.com
# @Project: LandscapeAR
# @Last modified by:   ibrejcha
# @Last modified time: 2020-08-12T11:45:48+02:00
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

from torch.utils.data import Dataset, DataLoader
from training.MultimodalPatchesDataset import loadListFile, plotSampleBatched, MultimodalPatchesDataset
from training.PositionalDatasetSampler import PositionalDatasetSampler
from tqdm import tqdm
import os
import random
import numpy as np

import cv2
cv2.setNumThreads(0)


class MultimodalPatchesDatasetAll(Dataset):
    def __init__(self, input_dir, dataset_list, rejection_radius_position=0,
                 numpatches=900, numneg=3, pos_thr=50.0, reject=True,
                 mode='train', rejection_radius=3000, dist_type='3D',
                 patch_radius=None, use_depth=False, use_normals=False,
                 use_silhouettes=False, color_jitter=False, greyscale=False,
                 maxres=4096, scale_jitter=False, photo_jitter=False,
                 uniform_negatives=False, needles=0, render_only=False):

        self.rejection_radius_position = rejection_radius_position
        self.numpatches = numpatches
        self.numneg = numneg
        self.pos_thr = pos_thr
        self.reject = reject
        self.mode = mode
        self.rejection_radius = rejection_radius
        self.dist_type = dist_type
        self.patch_radius = patch_radius
        self.use_depth = use_depth
        self.use_normals = use_normals
        self.use_silhouettes = use_silhouettes
        self.color_jitter = color_jitter
        self.greyscale = greyscale
        self.maxres = maxres
        self.scale_jitter = scale_jitter
        self.photo_jitter = photo_jitter
        self.uniform_negatives = uniform_negatives
        self.needles = needles
        self.render_only = render_only

        self.datasets = []
        self.datasets_idx = []
        self.datasets_len = []

        for d in dataset_list:
            print("Loading dataset:", d)
            train_sampled, di_path, pd_path = MultimodalPatchesDatasetAll.getItemList(
                input_dir, d, self.mode, self.rejection_radius_position
            )

            dataset = MultimodalPatchesDataset(
                pd_path, di_path, list=train_sampled,
                numpatches=self.numpatches, numneg=self.numneg,
                pos_thr=self.pos_thr, reject=self.reject,
                mode=self.mode, rejection_radius=self.rejection_radius,
                dist_type=self.dist_type,
                patch_radius=self.patch_radius, use_depth=self.use_depth,
                use_normals=self.use_normals,
                use_silhouettes=self.use_silhouettes,
                color_jitter=self.color_jitter, greyscale=self.greyscale,
                maxres=self.maxres, scale_jitter=self.scale_jitter,
                photo_jitter=self.photo_jitter,
                uniform_negatives=self.uniform_negatives, needles=self.needles,
                render_only=self.render_only
            )

            self.datasets.append(dataset)
            self.datasets_len.append(len(dataset))
            self.datasets_idx.append(0)

        self.datasets_len = np.array(self.datasets_len)
        self.datasets_idx = np.array(self.datasets_idx)

        self.dataset_percentage = self.datasets_len / np.sum(self.datasets_len)

        self.genItems()

    @staticmethod
    def getItemList(input_dir, dataset_name, mode, rejection_radius_position):
        dataset_path = os.path.join(input_dir, dataset_name)
        di_path = os.path.join(dataset_path, "final_dataset", "real")
        pd_path = os.path.join(dataset_path, "patches_dataset_all")
        dfile = 'train.txt'
        if mode != 'train':
            dfile = 'val.txt'
        dataset_train_list_path = os.path.join(pd_path, dfile)
        dataset_train_list = loadListFile(dataset_train_list_path)

        if rejection_radius_position > 0:
            d_pds = PositionalDatasetSampler(
                pd_path, di_path, dataset_train_list,
                rejection_radius_position
            )
            train_sampled = d_pds.sample()
        else:
            train_sampled = dataset_train_list
        if mode == 'train':
            random.shuffle(train_sampled)
        return train_sampled, di_path, pd_path

    @staticmethod
    def getListAll(input_dir, dataset_list, mode, rejection_radius_position):
        list_all = []
        for d in dataset_list:
            train_sampled, di_path, pd_path = MultimodalPatchesDatasetAll.getItemList(
                input_dir, d, mode, rejection_radius_position
            )
            list_all.append(train_sampled)
        if len(list_all) > 1:
            concat_list_all = list_all[0]
            for idx in range(1, len(list_all)):
                concat_list_all += list_all[idx]
            list_all = concat_list_all
        else:
            list_all = list_all[0]
        return list_all


    def getNextDatasetId(self, idx):
        # in case we are at the beginning, return the item from the
        # dataset with largest amount of items
        if idx == 0:
            return np.argmax(self.datasets_len)

        # Else check each dataset and return the dataset with the lowest
        # percentage of already used number of items which is also lower
        # than the percentage of all dataset items in the total number of
        # items across all datasets.
        current_percentage = self.datasets_idx / np.sum(self.datasets_idx)

        sorted_ids = np.argsort(current_percentage)
        sel_1 = (
            current_percentage[sorted_ids]
            <= self.dataset_percentage[sorted_ids]
        )
        selected_ids = sorted_ids[sel_1]
        return selected_ids[0]

    def genItems(self):
        self.dataset_order = []
        self.items = []
        print("Generating the ordering of items from all datasets...")
        for idx in tqdm(range(0, len(self))):
            dataset_id = self.getNextDatasetId(idx)
            self.items.append(self.datasets_idx[dataset_id])
            self.dataset_order.append(dataset_id)
            self.datasets_idx[dataset_id] += 1

    def __getitem__(self, idx):
        current_dataset = self.datasets[self.dataset_order[idx]]
        #print("current dataset", self.dataset_order[idx], "item:", self.items[idx], "idx:", idx)
        return current_dataset[self.items[idx]]

    def __len__(self):
        return np.sum(self.datasets_len)


if __name__ == "__main__":
    dataset_list = ["alps_chamonix_45.99681_7.055562_30"] #"geoPose3K_trainval" "alps_matterhorn_45.999444_7.832778_30"
    dataset = MultimodalPatchesDatasetAll("/mnt/scratch01/tmp/ibrejcha/data/matching/datasets", dataset_list, numneg=1, pos_thr=1.0, use_depth=False, use_normals=False, reject=False, dist_type='3D', rejection_radius=10, photo_jitter=False, uniform_negatives=True)
    dataloader = DataLoader(dataset, batch_size=300, shuffle=False, num_workers=8)
    for i_batch, sample_batched in tqdm(enumerate(dataloader)):
        if i_batch % 100 == 0:
            #plotSampleBatched(i_batch, sample_batched, dataset.dist_type)
            print(i_batch)

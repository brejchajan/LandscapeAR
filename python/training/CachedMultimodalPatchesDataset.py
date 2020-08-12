# @Date:   2020-08-06T16:28:03+02:00
# @Email:  ibrejcha@fit.vutbr.cz; brejchaja@gmail.com
# @Project: LandscapeAR
# @Last modified time: 2020-08-11T17:50:03+02:00
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

# our code

from training.MultimodalPatchesDatasetAll import MultimodalPatchesDatasetAll
from training.MultimodalPatchesDataset import loadListFile
from training.PositionalDatasetSampler import PositionalDatasetSampler

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch
import sklearn
import numpy as np
import os
import sys
from torch.utils.data.dataloader import default_collate

import random

import shutil

from torch.multiprocessing import Process, JoinableQueue, Event, Lock, Value


class CachedMultimodalPatchesDataset(Dataset):
    def __init__(self, multimodal_patches_cache):
        super(CachedMultimodalPatchesDataset, self).__init__()
        cache_path = multimodal_patches_cache.getCurrentCache()
        self.cache_path = cache_path
        self.use_depth = multimodal_patches_cache.use_depth
        self.use_normals = multimodal_patches_cache.use_normals
        self.mode = multimodal_patches_cache.mode
        self.numpatches = multimodal_patches_cache.numpatches

    def __len__(self):
        length = len([
            name for name in os.listdir(self.cache_path)
            if os.path.isfile(os.path.join(self.cache_path, name))
        ])
        return length

    def __getitem__(self, idx):
        # print("getting item: ", idx)
        batch_fname = os.path.join(
            self.cache_path, 'batch_' + str(idx) + '.pt'
        )

        batch = torch.load(batch_fname)
        anchor = batch['anchor']
        pos = batch['pos']
        neg = batch['neg']
        anchor_r = batch['anchor_r']
        pos_p = batch['pos_p']
        neg_p = batch['neg_p']
        c1 = batch['c1']
        c2 = batch['c2']
        cneg = batch['cneg']
        id = batch['id'].int()

        if not (self.use_depth or self.use_normals):
            # no need to store image data as float, convert to uint
            anchor = (anchor.float() / 255.0)
            pos = (pos.float() / 255.0)
            neg = (neg.float() / 255.0)
            anchor_r = (anchor_r.float() / 255.0)
            pos_p = (pos_p.float() / 255.0)
            neg_p = (neg_p.float() / 255.0)

        result = (anchor, pos, neg, anchor_r, pos_p, neg_p, c1, c2, cneg, id)
        # print("returning item", idx)
        return result


class MultimodalPatchesCache(object):

    def __init__(
        self, cache_dir, dataset_dir, dataset_list, cuda, batch_size=500,
        num_workers=3, renew_frequency=5, rejection_radius_position=0,
        numpatches=900, numneg=3, pos_thr=50.0, reject=True,
        mode='train', rejection_radius=3000, dist_type='3D',
        patch_radius=None, use_depth=False, use_normals=False,
        use_silhouettes=False, color_jitter=False, greyscale=False,
        maxres=4096, scale_jitter=False, photo_jitter=False,
        uniform_negatives=False, needles=0, render_only=False, maxitems=200,
        cache_once=False
    ):
        super(MultimodalPatchesCache, self).__init__()
        self.cache_dir = cache_dir
        self.dataset_dir = dataset_dir
        #self.images_path = images_path
        self.dataset_list = dataset_list
        self.cuda = cuda
        self.batch_size = batch_size

        self.num_workers = num_workers
        self.renew_frequency = renew_frequency
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

        self.cache_done_lock = Lock()
        self.all_done = Value('B', 0)  # 0 is False
        self.cache_done = Value('B', 0)  # 0 is False

        self.wait_for_cache_builder = Event()
        # prepare for wait until initial cache is built
        self.wait_for_cache_builder.clear()
        self.cache_builder_resume = Event()

        self.maxitems = maxitems
        self.cache_once = cache_once

        if self.mode == 'eval':
            self.maxitems = -1
        self.cache_builder = Process(
            target=self.buildCache,
            args=[self.maxitems]
        )
        self.current_cache_build = Value('B', 0)  # 0th cache
        self.current_cache_use = Value('B', 1)  # 1th cache

        self.cache_names = ["cache1", "cache2"]  # constant

        rebuild_cache = True
        if self.mode == 'eval':
            validation_dir = os.path.join(
                self.cache_dir,
                self.cache_names[self.current_cache_build.value])
            if os.path.isdir(validation_dir):
                # we don't need to rebuild validation cache
                # TODO: check if cache is VALID
                rebuild_cache = False
        elif cache_once:
            build_dataset_dir = os.path.join(
                self.cache_dir,
                self.cache_names[self.current_cache_build.value])
            if os.path.isdir(build_dataset_dir):
                # we don't need to rebuild training cache if we are training
                # on limited subset of the training set
                rebuild_cache = False

        if rebuild_cache:
            # clear the caches if they already exist
            build_dataset_dir = os.path.join(
                self.cache_dir,
                self.cache_names[self.current_cache_build.value]
            )
            if os.path.isdir(build_dataset_dir):
                shutil.rmtree(build_dataset_dir)
            use_dataset_dir = os.path.join(
                self.cache_dir,
                self.cache_names[self.current_cache_use.value]
            )
            if os.path.isdir(use_dataset_dir):
                shutil.rmtree(use_dataset_dir)

            os.makedirs(build_dataset_dir)

            self.cache_builder_resume.set()
            self.cache_builder.start()

            # wait until initial cache is built
            # print("before wait to build")
            # print("wait for cache builder state",
            #       self.wait_for_cache_builder.is_set())
            self.wait_for_cache_builder.wait()
            # print("after wait to build")

        # we have been resumed
        if self.mode != 'eval' and (not self.cache_once):
            # for training, we can set up the cache builder to build
            # the second cache
            self.restart()
        else:
            # else for validation we don't need second cache
            # we just need to switch the built cache to the use cache in order
            # to use it
            tmp = self.current_cache_build.value
            self.current_cache_build.value = self.current_cache_use.value
            self.current_cache_use.value = tmp

        # initialization finished, now this dataset can be used

    def getCurrentCache(self):
        # Lock should not be needed - cache_done is not touched
        # and cache_len is read only for cache in use, which should not
        # been touched by other threads
        # self.cache_done_lock.acquire()
        h5_dataset_filename = os.path.join(
            self.cache_dir,
            self.cache_names[self.current_cache_use.value]
        )
        # self.cache_done_lock.release()
        return h5_dataset_filename

    def restart(self):
        # print("Restarting - waiting for lock...")
        self.cache_done_lock.acquire()
        # print("Restarting cached dataset...")
        if self.cache_done.value and (not self.cache_once):
            cache_changed = True
            tmp_cache_name = self.current_cache_use.value
            self.current_cache_use.value = self.current_cache_build.value
            self.current_cache_build.value = tmp_cache_name
            # clear the old cache if exists
            build_dataset_dir = os.path.join(
                self.cache_dir,
                self.cache_names[self.current_cache_build.value])
            if os.path.isdir(build_dataset_dir):
                shutil.rmtree(build_dataset_dir)
            os.makedirs(build_dataset_dir)
            self.cache_done.value = 0  # 0 is False
            self.cache_builder_resume.set()
            # print("Switched cache to: ",
            #       self.cache_names[self.current_cache_use.value]
            # )
        else:
            cache_changed = False
            # print(
            #     "New cache not ready, continuing with old cache:",
            #     self.cache_names[self.current_cache_use.value]
            # )
        all_done_value = self.all_done.value
        self.cache_done_lock.release()
        # returns true if no more items are available to be loaded
        # this object should be destroyed and new dataset should be created
        # in order to start over.
        return cache_changed, all_done_value

    def buildCache(self, limit):
        # print("Building cache: ",
        #       self.cache_names[self.current_cache_build.value]
        # )
        dataset = MultimodalPatchesDatasetAll(
            self.dataset_dir, self.dataset_list,
            rejection_radius_position=self.rejection_radius_position,
            #self.images_path, list=train_sampled,
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
        n_triplets = len(dataset)

        if limit == -1:
            limit = n_triplets

        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False,
            pin_memory=False, num_workers=1,  # self.num_workers
            collate_fn=MultimodalPatchesCache.my_collate)

        qmaxsize = 15
        data_queue = JoinableQueue(maxsize=qmaxsize)

        # cannot load to cuda from background, therefore use cpu device
        preloader_resume = Event()
        preloader = Process(
            target=MultimodalPatchesCache.generateTrainingData,
            args=(data_queue, dataset, dataloader,
                  self.batch_size, qmaxsize, preloader_resume,
                  True, True)
        )
        preloader.do_run_generate = True
        preloader.start()
        preloader_resume.set()

        i_batch = 0
        data = data_queue.get()
        i_batch = data[0]

        counter = 0
        while i_batch != -1:

            self.cache_builder_resume.wait()

            build_dataset_dir = os.path.join(
                self.cache_dir,
                self.cache_names[self.current_cache_build.value]
            )
            batch_fname = os.path.join(
                build_dataset_dir, 'batch_' + str(counter) + '.pt'
            )

            # print("ibatch", i_batch,
            #        "___data___", data[3].shape, data[6].shape)

            anchor = data[1]
            pos = data[2]
            neg = data[3]
            anchor_r = data[4]
            pos_p = data[5]
            neg_p = data[6]
            c1 = data[7]
            c2 = data[8]
            cneg = data[9]
            id = data[10]

            if not (self.use_depth or self.use_normals):
                #no need to store image data as float, convert to uint
                anchor = (anchor * 255.0).to(torch.uint8)
                pos = (pos * 255.0).to(torch.uint8)
                neg = (neg * 255.0).to(torch.uint8)
                anchor_r = (anchor_r * 255.0).to(torch.uint8)
                pos_p = (pos_p * 255.0).to(torch.uint8)
                neg_p = (neg_p * 255.0).to(torch.uint8)

            tosave = {
                'anchor': anchor,
                'pos': pos,
                'neg': neg,
                'anchor_r': anchor_r,
                'pos_p': pos_p,
                'neg_p': neg_p,
                'c1': c1,
                'c2': c2,
                'cneg': cneg,
                'id': id
            }

            try:
                torch.save(tosave, batch_fname)
                torch.load(batch_fname)
                counter += 1
            except Exception as e:
                print(
                    "Could not save ", batch_fname, ", due to:",
                    e, "skipping...", file=sys.stderr
                )
                if os.path.isfile(batch_fname):
                    os.remove(batch_fname)

            data_queue.task_done()

            if counter >= limit:
                self.cache_done_lock.acquire()
                self.cache_done.value = 1  # 1 is True
                self.cache_done_lock.release()
                counter = 0
                # sleep until calling thread wakes us
                self.cache_builder_resume.clear()
                # resume calling thread so that it can work
                self.wait_for_cache_builder.set()

            data = data_queue.get()
            i_batch = data[0]
            #print("ibatch", i_batch)

        data_queue.task_done()

        self.cache_done_lock.acquire()
        self.cache_done.value = 1  # 1 is True
        self.all_done.value = 1
        print("Cache done ALL")
        self.cache_done_lock.release()
        # resume calling thread so that it can work
        self.wait_for_cache_builder.set()
        preloader.join()
        preloader = None
        data_queue = None

    @staticmethod
    def loadBatch(sample_batched, mode, device, keep_all=False):
        if mode == 'eval':
            coords1 = sample_batched[6]
            coords2 = sample_batched[7]
            coords_neg = sample_batched[8]
            keep = sample_batched[10]
            item_id = sample_batched[11]
        else:
            coords1 = sample_batched[6]
            coords2 = sample_batched[7]
            coords_neg = sample_batched[8]
            keep = sample_batched[9]
            item_id = sample_batched[10]
        if keep_all:
            # requested to return fill batch
            batchsize = sample_batched[0].shape[0]
            keep = torch.ones(batchsize).byte()
        keep = keep.reshape(-1)
        keep = keep.bool()
        anchor = sample_batched[0]
        pos = sample_batched[1]
        neg = sample_batched[2]

        # swapped photo to render
        anchor_r = sample_batched[3]
        pos_p = sample_batched[4]
        neg_p = sample_batched[5]

        anchor = anchor[keep].to(device)
        pos = pos[keep].to(device)
        neg = neg[keep].to(device)

        anchor_r = anchor_r[keep]
        pos_p = pos_p[keep]
        neg_p = neg_p[keep]

        coords1 = coords1[keep]
        coords2 = coords2[keep]
        coords_neg = coords_neg[keep]
        item_id = item_id[keep]
        return anchor, pos, neg, anchor_r, pos_p, neg_p, coords1, coords2, \
            coords_neg, item_id

    @staticmethod
    def generateTrainingData(queue, dataset, dataloader, batch_size, qmaxsize,
                             resume, shuffle=True, disable_tqdm=False):
        local_buffer_a = []
        local_buffer_p = []
        local_buffer_n = []

        local_buffer_ar = []
        local_buffer_pp = []
        local_buffer_np = []

        local_buffer_c1 = []
        local_buffer_c2 = []
        local_buffer_cneg = []
        local_buffer_id = []
        nbatches = 10
        # cannot load to cuda in batckground process!
        device = torch.device('cpu')

        buffer_size = min(qmaxsize * batch_size, nbatches * batch_size)
        bidx = 0
        for i_batch, sample_batched in enumerate(dataloader):
            # tqdm(dataloader, disable=disable_tqdm)
            resume.wait()
            anchor, pos, neg, anchor_r, \
                pos_p, neg_p, c1, c2, cneg, id = \
                MultimodalPatchesCache.loadBatch(
                    sample_batched, dataset.mode, device
                )
            if anchor.shape[0] == 0:
                continue
            local_buffer_a.extend(list(anchor))  # [:current_batches]
            local_buffer_p.extend(list(pos))
            local_buffer_n.extend(list(neg))

            local_buffer_ar.extend(list(anchor_r))
            local_buffer_pp.extend(list(pos_p))
            local_buffer_np.extend(list(neg_p))

            local_buffer_c1.extend(list(c1))
            local_buffer_c2.extend(list(c2))
            local_buffer_cneg.extend(list(cneg))
            local_buffer_id.extend(list(id))
            if len(local_buffer_a) >= buffer_size:
                if shuffle:
                    local_buffer_a, local_buffer_p, local_buffer_n, \
                        local_buffer_ar, local_buffer_pp, local_buffer_np, \
                        local_buffer_c1, local_buffer_c2, local_buffer_cneg, \
                        local_buffer_id = sklearn.utils.shuffle(
                            local_buffer_a,
                            local_buffer_p,
                            local_buffer_n,
                            local_buffer_ar,
                            local_buffer_pp,
                            local_buffer_np,
                            local_buffer_c1,
                            local_buffer_c2,
                            local_buffer_cneg,
                            local_buffer_id
                        )
                curr_nbatches = int(np.floor(len(local_buffer_a) / batch_size))
                for i in range(0, curr_nbatches):
                    queue.put([bidx, torch.stack(local_buffer_a[:batch_size]),
                               torch.stack(local_buffer_p[:batch_size]),
                               torch.stack(local_buffer_n[:batch_size]),
                               torch.stack(local_buffer_ar[:batch_size]),
                               torch.stack(local_buffer_pp[:batch_size]),
                               torch.stack(local_buffer_np[:batch_size]),
                               torch.stack(local_buffer_c1[:batch_size]),
                               torch.stack(local_buffer_c2[:batch_size]),
                               torch.stack(local_buffer_cneg[:batch_size]),
                               torch.stack(local_buffer_id[:batch_size])])
                    del local_buffer_a[:batch_size]
                    del local_buffer_p[:batch_size]
                    del local_buffer_n[:batch_size]
                    del local_buffer_ar[:batch_size]
                    del local_buffer_pp[:batch_size]
                    del local_buffer_np[:batch_size]
                    del local_buffer_c1[:batch_size]
                    del local_buffer_c2[:batch_size]
                    del local_buffer_cneg[:batch_size]
                    del local_buffer_id[:batch_size]
                    bidx += 1
        remaining_batches = len(local_buffer_a) // batch_size
        for i in range(0, remaining_batches):
            queue.put([bidx, torch.stack(local_buffer_a[:batch_size]),
                       torch.stack(local_buffer_p[:batch_size]),
                       torch.stack(local_buffer_n[:batch_size]),
                       torch.stack(local_buffer_ar[:batch_size]),
                       torch.stack(local_buffer_pp[:batch_size]),
                       torch.stack(local_buffer_np[:batch_size]),
                       torch.stack(local_buffer_c1[:batch_size]),
                       torch.stack(local_buffer_c2[:batch_size]),
                       torch.stack(local_buffer_cneg[:batch_size]),
                       torch.stack(local_buffer_id[:batch_size])])
            del local_buffer_a[:batch_size]
            del local_buffer_p[:batch_size]
            del local_buffer_n[:batch_size]
            del local_buffer_ar[:batch_size]
            del local_buffer_pp[:batch_size]
            del local_buffer_np[:batch_size]
            del local_buffer_c1[:batch_size]
            del local_buffer_c2[:batch_size]
            del local_buffer_cneg[:batch_size]
            del local_buffer_id[:batch_size]
        ra = torch.randn(batch_size, 3, 64, 64)
        queue.put([-1, ra, ra, ra])
        queue.join()

    @staticmethod
    def my_collate(batch):
        batch = list(filter(lambda x: x is not None, batch))
        return default_collate(batch)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    dataset_list = ["alps_chamonix_45.99681_7.055562_30"] #"geoPose3K_trainval" "alps_matterhorn_45.999444_7.832778_30"
    #dset = "/mnt/matylda1/ibrejcha/adobe_intern/data/switzerland_wallis_30km_maximg/patches_dataset_all_3000_val" #smaller dataset
    #dset = '/mnt/matylda1/ibrejcha/data/matching/input/geoPose3K_trainval/rendered_dataset_random_sample_20_200_10/patches_dataset_all'
    #dset = "/Users/janbrejcha/Downloads/test_patches_6"
    #images_path = "/home/ibrejcha/data/switzerland_wallis_30km_maximg/final_dataset/real"
    #images_path = '/mnt/matylda1/ibrejcha/data/matching/input/geoPose3K_trainval/rendered_dataset_random_sample_20_200_10/real'
    #images_path = "/mnt/matylda1/ibrejcha/adobe_intern/data/switzerland_wallis_30km_maximg/final_dataset/real"
    #train_path = os.path.join(dset, "val.txt")
    #train = loadListFile(train_path)
    #random.shuffle(train)
    #cache_dir = "/home/ibrejcha/data/cache/val"
    cache_dir="/tmp/ibrejcha/cache"
    cache = MultimodalPatchesCache(cache_dir, "/mnt/scratch01/tmp/ibrejcha/data/matching/datasets", dataset_list, True, batch_size=500, mode='train',
    num_workers=3, renew_frequency=5, rejection_radius_position=0, numneg=1, pos_thr=1.0, use_depth=False, use_normals=False, reject=False, dist_type='3D', rejection_radius=10, photo_jitter=False, uniform_negatives=False)
    cached_dataset = CachedMultimodalPatchesDataset(cache)
    print("dataset length: ", len(cached_dataset))
    cnt = 0
    device = torch.device('cpu')
    while True:
        print("Starting cached sub epoch:", cnt)
        for i in tqdm(range(len(cached_dataset))):
            data = cached_dataset[i]
            anchor = data[0].to(device, non_blocking=True)
            pos = data[1].to(device, non_blocking=True)
            print("data1 shape", data[1].shape)
            print("data2 shape", data[2].shape)
            neg = data[2].reshape(-1, data[2].shape[2], data[2].shape[3], data[2].shape[4]).to(device, non_blocking=True)
            anchor_r = data[3].to(device, non_blocking=True)
            pos_p = data[4].to(device, non_blocking=True)
            neg_p = data[5].reshape(-1, data[5].shape[2], data[5].shape[3], data[5].shape[4]).to(device, non_blocking=True)
            c1 = data[6].to(device, non_blocking=True) #coords per patch in 3D
            c2 = data[7].to(device, non_blocking=True)
            cneg = data[8].to(device, non_blocking=True)
            id = data[9].to(device, non_blocking=True)
            # print(cached_dataset[i][0])
        cnt += 1
        cache_changed, cache_done_all = cache.restart()
        if cache_changed:
            cached_dataset = CachedMultimodalPatchesDataset(cache)
        if cache_done_all:
            print("Finished!")
            break

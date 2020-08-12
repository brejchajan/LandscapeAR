# @Author: Jan Brejcha <ibrejcha>
# @Date:   2020-08-11T17:52:14+02:00
# @Email:  ibrejcha@fit.vutbr.cz; brejchaja@gmail.com
# @Project: LandscapeAR
# @Last modified by:   ibrejcha
# @Last modified time: 2020-08-12T11:48:39+02:00
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

from __future__ import print_function

import torch.multiprocessing

import torch
import subprocess

from training.Architectures import MultimodalPatchNet, MultimodalPatchNet5lShared2l
from training.Architectures import MultimodalPatchNet5lShared2lBN, MultimodalKeypointPatchNet5lShared2l
from training.MultimodalPatchesDatasetAll import MultimodalPatchesDatasetAll
from training.MultimodalPatchesDataset import loadListFile
from training.CachedMultimodalPatchesDataset import MultimodalPatchesCache, CachedMultimodalPatchesDataset
from pose_estimation.EstimatePose import poseFrom2D3D
from visualization.drawMatches import drawMatches
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from tqdm import tqdm
import os
from time import gmtime, strftime
import argparse as ap
import numpy as np

from sklearn.neighbors import NearestNeighbors
from pose_estimation.patchSamplingDepth import generatePatchesImg, loadDepth
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import cv2
import pose_estimation.FUtil as FUtil
import torchvision
import re
import time

from training.LossesAdobetrips import loss_HardNetWithDist
from training.LossesAdobetrips import select_HardNetMultimodal

from torch.utils.data.dataloader import default_collate

plt.ion()

def adjust_learning_rate(optimizer, args, n_triplets, n_epochs):
    """Taken from the HardNet codebase.
    Updates the learning rate given the learning rate decay.
    The routine has been implemented according to the original Lua SGD optimizer
    """
    for group in optimizer.param_groups:
        if 'step' not in group:
            group['step'] = 0.
        else:
            group['step'] += 1.
        group['lr'] = args.learning_rate * (
            1.0 - float(group['step']) * float(args.batch_size) / (n_triplets * float(n_epochs)))
    return


def saveModel(log_dir, net, ep, step, optimizer, loss, device, hard_dist_coeff):
    outdir = os.path.join(log_dir, "models")
    if (not os.path.isdir(outdir)):
        os.makedirs(outdir)
    save_path = os.path.join(outdir,
                             type(net).__name__ + "_epoch_" + str(ep)
                             + "_step_" + str(step))

    print("Saving checkpoint: " + save_path)
    torch.save({
        'epoch': ep,
        'step': step,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'hard_dist_coeff': hard_dist_coeff,
        'loss': loss
    }, save_path)


def findLatestModelPath(models_path):
    model_files = [f for f in os.listdir(models_path) if os.path.isfile(os.path.join(models_path, f))]
    filtered_model_files = []
    epochs = []
    for file in model_files:
        if not file.startswith("."):
            parts = file.split("_")
            epochs.append(int(parts[2]))
            filtered_model_files.append(file)
    epochs = np.array(epochs)
    idx = np.argsort(epochs)
    latest_model = filtered_model_files[idx[-1]]
    latest_model_path = os.path.join(models_path, latest_model)
    return latest_model_path


def buildArgumentParser():
    parser = ap.ArgumentParser()
    parser.add_argument("dataset_path", help="""Root path of all datasets.""")
    parser.add_argument("dataset_list", help="""Path to file containing
                        list of the datasets to be used for training.
                        The datasets named in this file need to contain file
                        called train.txt inside its patches_dataset_all
                        directory containing list of all pairs for
                        training.""")
    parser.add_argument("val_dataset_list", help="""Path to file containing
                        list of the datasets to be used for validation.
                        The datasets named in this file need to contain file
                        called val.txt inside its patches_dataset_all
                        directory containing list of all pairs for
                        validation.""")
    parser.add_argument("cache_dir", help="""Absolute path to the directory
                        which will be used for the dataset cache to speedup
                        training. Please make sure the storage is large
                        enough (several tens of GB). The faster the better,
                        ideally SSD.""")
    parser.add_argument("-l", "--log_dir", help="""Directory used for saving
                        training progress logs and models. Defaults to cwd
                        when not specified.""", default=os.getcwd())
    parser.add_argument("-r", "--restore", help="""Restores the training from
                        the snapshot with given name. If no model is specified
                        using -m, loads latest model.""")
    parser.add_argument("-m", "--model_name", help="""Specify exact model name
                        to be restored using -r option.""")
    parser.add_argument("--architecture", help="Specify name of the \
                        network architecture to be used. \
                        (Default: MultimodalPatchNet5lShared2l)",
                        default="MultimodalPatchNet5lShared2l")
    parser.add_argument("-v", "--validate", help="Only run validation, \
                        do not train.", action="store_true")
    parser.add_argument("-sf", "--save_figs", type=str, nargs=1,
                        help="Can be combined with -v option. If set, the \
                        validation figures will be saved onto the specified \
                        directory.")
    parser.add_argument("-s", "--sift", action="store_true", help="To be \
                        combined with -v option. If this flag is used, \
                        the validation will run using OpenCV SIFT descriptor, \
                        and neural model won't be used.")
    parser.add_argument("-mi", "--match_images", nargs=3, help=" \
                        Matches two input images with each other. The \
                        third argument defines output path.")
    parser.add_argument("-a", "--autoselect-gpu", action='store_true',
                        help="Automatically detect empty GPU and use it for \
                        training. Usable when used on cluster.")
    parser.add_argument("-c", "--cuda", help="If set, cuda is used.",
                        action="store_true")
    parser.add_argument("-lr", "--learning_rate", default=0.0001, type=float,
                        help="Learning rate used for training with SGD.")
    parser.add_argument("-n", "--num_workers", default=8, type=int,
                        help="Number of workers used for sampling patches \
                        into the cache.")
    parser.add_argument("--num_workers_loader", default=1, type=int,
                        help="Number of workers used for loading the \
                        cached items. Usually 0 should be fine.")
    parser.add_argument("-rr", "--rejection_radius", type=int, default=10,
                        help="Query radius used for density rejection \
                        sampling in meters.")
    parser.add_argument("--rejection_radius_position", type=int, default=-1,
                        help="Rejection radius in meters used for positional \
                        dataset rejection sampling in each epoch. If zero or \
                        a negative number is specified, the positional \
                        rejection sampling will be turned off (default)")
    parser.add_argument("-pt", "--positive_threshold", type=float,
                        help="Distance in 3D in meters to define positive \
                        and negative pairs. Patch with 3D distance below \
                        positive threshold is considered positive, and \
                        closest patches exceeding the threshold are considered\
                        negative.", default=500)
    parser.add_argument("-d", "--distance_type", default="3D", help="Distance \
                        to be used for positive/negative pair selection. \
                        Valid options are `2D` and `3D`. Default: 3D.")
    parser.add_argument("-no", "--normalize_output", action="store_true",
                        help="If set, the descriptors are normalized to unit \
                        hypersphere.")
    parser.add_argument("-nr", "--no_reject", action="store_true",
                        help="Do not use 3D density rejection sampling during \
                        training. It will be used ANYWAYS during validation.")
    parser.add_argument("-hn", "--hardnet", action="store_true",
                        help="If set to true, the hardnet loss is used for \
                        training the model. Default False.", default=False)
    parser.add_argument('--margin', type=float, default=1.0, metavar='MARGIN',
                        help='the margin value for the triplet loss function \
                        (default: 1.0')
    parser.add_argument('--hardnet_anchorswap', type=bool, default=False,
                        help='turns on anchor swap for hardnet loss.')
    parser.add_argument('--hardnet_anchorave', type=bool, default=False,
                        help='anchorave for hardnet loss')
    parser.add_argument('--hardnet_batch_reduce', default='min', help='Batch reduce\
                        for hardnet loss. Other options: average, random')
    parser.add_argument('--hardnet_loss', default='triplet_margin',
                        help='Loss type inside hardnet loss. \
                        Other options: softmax, contrastive')
    parser.add_argument('--hardnet_filter_dist', action='store_true',
                        help='If this flag is used, the negatives in hardnet\
                        loss will be filtered not to contain too close and \
                        too far negatives to improve training stability.')
    parser.add_argument('--hardnet_filter_dist_values', nargs=2,
                        default=[100.0, 300.0], help='Minimum and maximum \
                        allowable distance in meters for a 3D point to be \
                        considered as a negative. Everything below the first \
                        number and everything above the second number \
                        is excluded from the negative selection. Default \
                        100.0 and 300.0. Effective only in combination with \
                        --hardnet_filter_dist flag.', type=float)
    parser.add_argument('--hardest-mining', action='store_true', help=' \
                        Use hardest negative during hardmining (as original) \
                        HardNet does. If not set, semi-hard mining is used.')
    parser.add_argument("--batch_size", type=int, default=600, help="Final \
                        batch size assembled by the loadbalancer, which is \
                        used to training.")
    parser.add_argument("--disable_tqdm", default=False, action='store_true',
                        help="Disables the tqdm progress bar during training.")
    parser.add_argument("--use_depth", action='store_true', help="If this \
                        flag is used, the rendered patches will concatenated \
                        with a depth channel.")
    parser.add_argument("--use_normals", action='store_true', help="If this \
                        flag is used, the rendered patches will be \
                        concatenated with 3 channels encoding normals.")
    parser.add_argument("--use_silhouettes", action='store_true', help="If this \
                        flag is used, the rendered patches will be \
                        concatenated with 1 channel encoding silhouettes.")
    parser.add_argument("--color_jitter", action='store_true', help='Enable \
                        color jittering on rendered images.')
    parser.add_argument("--no_symmetric", action='store_true',
                        help="If this flag is set, anchor will be always \
                        photo, and positive/negative samples will always \
                        be renders. If this flag is NOT used, loss is \
                        calculated symmetrically, with anchor patch as \
                        photo, and pos neg as render, and then flipped: \
                        anchor as render and pos neg as photo.")
    parser.add_argument("--greyscale", action='store_true', help="If set, \
                        the patches will be converted to greyscale.")
    parser.add_argument("--patch_radius", default=None, type=int, help="\
                        defines the patch size which will be used as input.")
    parser.add_argument("--do_plot", action='store_true', help="Visualize \
                        the patches generated using hard negative mining. \
                        Can be used only jointly with the --hardnet option.")
    parser.add_argument("--plot_frequency", default=200, help="The \
                        visualization will be updated each Nth step.",
                        type=int)
    parser.add_argument("--hardnet_orig", action='store_true', help="Original \
                        hardnet will be used instead of \
                        our as-hard-as-possible approach.")
    parser.add_argument("--adaptive_mining", action='store_true',
                        help="Usable only when --hardest-mining is NOT used. \
                        Gradually adapts the semi-hard negative mining to be \
                        harder and harder until it reaches the hardest \
                        negative mining.")
    parser.add_argument("--adaptive_mining_step", type=float, default=0.05,
                        help="Step which is added after each validation to \
                        the semihard negative mining.")
    parser.add_argument("--adaptive_mining_coeff", type=float, default=0.05,
                        help="Start value for adaptive semihard mining \
                        coefficient.")
    parser.add_argument("--maxres", default=4096, help="Resolution for \
                        recalculating width according to the FOV of the image.\
                        180 degrees corresponds to the maxres. Default=4096, \
                        might be subject of change.", type=int)
    parser.add_argument("--uniform_negatives", action='store_true', help="If \
                        used, uniformly sampled negatives will be added \
                        as an additional triplet loss (possibly with \
                        hardnegative mining if --hardnet is used as well.)")
    parser.add_argument("--scale_jitter", action='store_true', help="\
                        Jitter the scale of the patches.")
    parser.add_argument("--photo_jitter", action='store_true', help="Jitter \
                        the color, rotation, shear, translation of \
                        each photo patch. Also applies random gamma to photo.")
    parser.add_argument("--needles", default=0, type=int, help="If number \
                        greater than zero is used, then instead of a single \
                        patch a whole needle of patches will be extracted. Our\
                        network then takes several patches in a form of a \
                        needle encoded to channels of the input. This \
                        approach is described here: Lotan and Irani: \
                        Needle-Match: Reliable Patch Matching under \
                        High Uncertainty, CVPR 2016")
    parser.add_argument("--render_only", action='store_true',
                        help="Trains only using rendered images.")
    parser.add_argument("--num_epochs", default=10, help="Number of epochs \
                        to train.", type=int)
    parser.add_argument("--loss_aux", action='store_true', help="Adds loss \
                        term for photo-photo and render-render matching.")
    parser.add_argument("--cache-once", type=int, metavar='LIMIT',
                        help="Creates a cache of limited size at the start of\
                        the training and uses only this cache for whole \
                        training. LIMIT sets the number of items in the cache.\
                        Intended mainly for debugging purposes to be able to \
                        train fast.")
    parser.add_argument("--name", type=str, help="Name of the run instead of \
                        automatically generated timestamp.")
    return parser


def computeRepresentations(net, dataset, d_idx, device, batchsize, loss_func):
    orig_reduction = loss_func.reduction
    loss_func.reduction = 'none'
    anchor = []
    pos = []
    neg = []
    coords1 = []
    coords2 = []
    anchor_fea = []
    pos_fea = []
    neg_fea = []
    losses = []

    for idx in range(d_idx, d_idx + dataset.numpatches):
        if idx % batchsize == 0 and len(anchor) > 0:
            anchor_b = torch.stack(anchor).to(device)
            pos_b = torch.stack(pos).to(device)
            neg_b = torch.stack(neg)[:,0,:].to(device)

            anchor = []
            pos = []
            neg = []

            anchor_out, pos_out, neg_out = net(anchor_b, pos_b, neg_b)
            loss = loss_func(anchor_out, pos_out, neg_out).cpu().detach().numpy()
            losses.append(loss)
            anchor_fea.append(anchor_out.cpu())
            pos_fea.append(pos_out.cpu())
            neg_fea.append(neg_out.cpu())
            anchor_out = 0
            pos_out = 0
            neg_out = 0

        p1, p2, n, c1, c2 = dataset[idx]
        anchor.append(p1)
        pos.append(p2)
        neg.append(n)
        coords1.append(c1)
        coords2.append(c2)


    if (len(anchor) > 0):
        # compute the rest
        anchor_b = torch.stack(anchor).to(device)
        pos_b = torch.stack(pos).to(device)
        neg_b = torch.stack(neg)[:, 0, :].to(device)

        anchor = []
        pos = []
        neg = []

        anchor_out, pos_out, neg_out = net(anchor_b, pos_b, neg_b)
        loss = loss_func(anchor_out, pos_out, neg_out).cpu().detach().numpy()
        losses.append(loss)

        anchor_fea.append(anchor_out.cpu())
        pos_fea.append(pos_out.cpu())
        neg_fea.append(neg_out.cpu())
        anchor_out = 0
        pos_out = 0
        neg_out = 0

    anchor_fea = torch.stack(anchor_fea).detach().numpy()
    pos_fea = torch.stack(pos_fea).detach().numpy()
    neg_fea = torch.stack(neg_fea).detach().numpy()

    anchor_fea_r = anchor_fea.reshape(dataset.numpatches, anchor_fea.shape[2])
    pos_fea_r = pos_fea.reshape(dataset.numpatches, pos_fea.shape[2])
    neg_fea_r = neg_fea.reshape(dataset.numpatches, neg_fea.shape[2])
    coords1 = np.array(coords1)
    coords2 = np.array(coords2)
    losses = np.array(losses).reshape(dataset.numpatches)

    loss_func.reduction = orig_reduction

    return anchor_fea_r, pos_fea_r, neg_fea_r, coords1, coords2, losses


def calculate2DDistances(anchor_fea, pos_fea, coords2, num_neigh=10):
    kdt = NearestNeighbors(n_neighbors=num_neigh, algorithm='kd_tree').fit(pos_fea)

    dists, nn_idxs = kdt.kneighbors(anchor_fea)
    nn_coords = coords2[nn_idxs]
    diff = nn_coords - np.expand_dims(coords2, 1)
    dist_2d = np.linalg.norm(diff, axis=2)
    return dist_2d, dists, nn_idxs


def saveValidationFigs(nn_idxs, dist_2d, fea_dist, dataset, d_idx, name,
                       output_dir, num_neigh=10):
    if dataset.mode != 'eval':
        raise Warning("To save figs the dataset needs to be in eval mode!")
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    print("Saving validation figs:")
    idxs = np.arange(0, nn_idxs.shape[0], 50)
    for idx in tqdm(idxs):
        p1, p2, n, p1r, p2p, npp, c1, c2, c3, color, sel, id = dataset[d_idx + idx]
        p1 = p1.numpy()
        p1 = np.moveaxis(p1, 0, -1)
        if p1.shape[2] == 1:
            p1 = p1.reshape(p1.shape[0], p1.shape[1])
        nn_idx = nn_idxs[idx]

        fig = plt.figure()
        ax = plt.subplot(3, 10, 1)
        ax.axis("off")
        plt.imshow(p1)
        # top num_neigh
        for i in range(0, num_neigh):
            p1, p2, n, p1r, p2p, npp, c1, c2, c3, color, sel, id = dataset[d_idx + nn_idx[i]] # nn_idx[0, i]
            p1 = p1.numpy()
            p1 = np.moveaxis(p1, 0, -1)
            p2 = p2.numpy()
            p2 = np.moveaxis(p2, 0, -1)
            if p2.shape[2] == 1:
                p2 = p2.reshape(p2.shape[0], p2.shape[1])
            ax = plt.subplot(3, 10, 11 + i)
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            ax.set_title("{0:.2f}".format(fea_dist[idx,i]))
            ax.set_xlabel("{0:.2f}".format(dist_2d[idx,i]))
            plt.imshow(p2)
            # before, we were showing also positive patch corresponding to
            # the retrieved one
            # plt.subplot(3, 10, 21 + i)
            # plt.imshow(p1)

        # bottom num_neigh
        plt_idx = 0
        for i in range(nn_idxs.shape[1]-1, nn_idxs.shape[1]-1-num_neigh, -1):
            p1, p2, n, p1r, p2p, npp, c1, c2, c3, color, sel, id = dataset[d_idx + nn_idx[i]] #nn_idx[0, i]
            p2 = p2.numpy()
            p2 = np.moveaxis(p2, 0, -1)
            if p2.shape[2] == 1:
                p2 = p2.reshape(p2.shape[0], p2.shape[1])
            ax = plt.subplot(3, 10, 21 + plt_idx)
            ax.set_title("{0:.2f}".format(fea_dist[idx,i]))
            ax.set_xlabel("{0:.2f}".format(dist_2d[idx,i]))
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            plt.imshow(p2)
            plt_idx += 1

        #plt.show()

        plt.savefig(os.path.join(output_dir, name) + "_" + str(idx) + ".png")
        plt.close(fig)


def describe_sift(patches):
    sift = cv2.xfeatures2d.SIFT_create()
    kp = [cv2.KeyPoint(32, 32, 32)]
    descs = []
    for p in range(0, patches.shape[0]):
        patch = np.moveaxis(patches[p], 0, -1)
        patch = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        patch = (patch * 255).astype(np.uint8)
        kp, desc = sift.compute(patch, kp)
        descs.append(desc)

    return np.concatenate(descs)


def validate_all_sift(net, dataset, val_list, device, loss_func, val_step,
                      num_workers, output_figs_dir,
                      batchsize=300, savefigs=False):
    net.eval()
    dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=False, num_workers=num_workers, collate_fn=my_collate)

    # numpatches per image must be divisible by batch size without remainder
    assert (int(dataset.numpatches) % int(batchsize)) == 0

    batches = dataset.numpatches // batchsize

    anchor_fea_sift = []
    pos_fea_sift = []
    coords1 = []
    coords2 = []

    dists_sift = []

    idx = 0
    for i_batch, sample_batched in enumerate(tqdm(dataloader)):
        (
            anchor, pos, neg, ar, pp, np, c1, c2, cneg, id
        ) = MultimodalPatchesCache.loadBatch(
            sample_batched, dataset.mode, device
        )
        if anchor.shape[0] == 0:
            continue

        coords1.append(c1)
        coords2.append(c2)

        anchor_sift = describe_sift(anchor.cpu().numpy())
        pos_sift = describe_sift(pos.cpu().numpy())
        anchor_fea_sift.append(anchor_sift)
        pos_fea_sift.append(pos_sift)
        # calculate 2D distances
        if (i_batch+1) % batches == 0 and i_batch > 0:

            anchor_fea_sift = np.concatenate(anchor_fea_sift)
            pos_fea_sift = np.concatenate(pos_fea_sift)
            coords1 = np.concatenate(coords1)
            coords2 = np.concatenate(coords2)

            current_batch_size = anchor.shape[0]
            dist_2d_sift, fea_dist, nn_idxs_sift = calculate2DDistances(anchor_fea_sift,
                                                              pos_fea_sift,
                                                              coords2,
                                                              num_neigh=current_batch_size)
            dists_sift.append(dist_2d_sift)
            d_idx = idx * dataset.numpatches

            if savefigs:
                saveValidationFigs(nn_idxs_sift, dist_2d_sift, fea_dist,
                                   dataset, d_idx, val_list[idx],
                                   output_figs_dir, 10)

            anchor_fea_sift = []
            pos_fea_sift = []
            coords1 = []
            coords2 = []

            idx = idx + 1

    dists_sift = np.concatenate(dists_sift)
    np.save(os.path.join(writer.log_dir, "dists_sift_val_step_" + str(val_step)), dists_sift)
    writer.add_histogram('dist2d_sift_first', dists_sift[:, 0], val_step)
    writer.add_histogram('dist2d_sift_mean_top10', np.mean(dists_sift, axis=1), val_step)


def my_collate(batch):
    batch = list(filter(lambda x : x is not None, batch))
    return default_collate(batch)


def validate_all(net, dataset, val_list, device, loss_func, val_step,
                 num_workers, output_figs_dir, batchsize=300,
                 savefigs=False, onlyloss=False):
    net.eval()
    train_keypoints = False
    if isinstance(net, MultimodalKeypointPatchNet5lShared2l):
        train_keypoints = True

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False,
                            num_workers=num_workers, collate_fn=my_collate)

    # numpatches per image must be divisible by batch size without remainder
    # assert (int(dataset.numpatches) % int(batchsize)) == 0

    batches = max(1, dataset.numpatches // batchsize)

    anchor_fea = []
    pos_fea = []
    neg_fea = []
    coords1 = []
    coords2 = []

    losses = []
    dists = []

    patches_1 = []
    patches_2 = []
    patches_proba = []
    idx = 0
    for i_batch, data in enumerate(tqdm(dataloader)):
        anchor = data[0][0].to(device, non_blocking=True)
        pos = data[1][0].to(device, non_blocking=True)
        neg = data[2][0].reshape(-1, data[2][0].shape[2], data[2][0].shape[3], data[2][0].shape[4]).to(device, non_blocking=True)
        anchor_r = data[3][0].to(device, non_blocking=True)
        pos_p = data[4][0].to(device, non_blocking=True)
        neg_p = data[5][0].reshape(-1, data[5][0].shape[2], data[5][0].shape[3], data[5][0].shape[4]).to(device, non_blocking=True)
        c1 = data[6][0] #.to(device, non_blocking=True) #coords per patch in 3D
        c2 = data[7][0] #.to(device, non_blocking=True)
        cneg = data[8][0].to(device, non_blocking=True)
        id = data[9][0].to(device, non_blocking=True)

        current_batch_size = anchor.shape[0]
        if current_batch_size == 0:
            continue

        coords1.append(c1.numpy())
        coords2.append(c2.numpy())

        if train_keypoints:
            anchor_out, pos_out, neg_out, _sa, _sp, _sn = net(anchor, pos, neg)
        else:
            anchor_out, pos_out, neg_out = net(anchor, pos, neg)
        loss = torch.mean(triplet_loss(anchor_out, pos_out, neg_out)).cpu().detach().numpy()
        losses.append(loss)
        anchor_fea.append(anchor_out.detach().cpu().numpy())
        pos_fea.append(pos_out.detach().cpu().numpy())
        neg_fea.append(neg_out.detach().cpu().numpy())
        anchor_out = 0
        pos_out = 0
        neg_out = 0
        patches_1.append(data[0][0])
        patches_2.append(data[1][0])
        patches_proba.append(data[9][0])

        if not onlyloss or savefigs:
            #calculate 2D distances
            if (i_batch) % batches == 0:
                anchor = 0
                pos = 0
                neg = 0
                anchor_fea = np.concatenate(anchor_fea)
                pos_fea = np.concatenate(pos_fea)
                neg_fea = np.concatenate(neg_fea)
                coords1 = np.concatenate(coords1)
                coords2 = np.concatenate(coords2)

                dist_2d, fea_dist, nn_idxs = calculate2DDistances(anchor_fea, pos_fea, coords2, num_neigh=current_batch_size)
                dists.append(dist_2d[:, :10].copy())
                d_idx = idx * dataset.numpatches

                if savefigs:
                    saveValidationFigs(nn_idxs, dist_2d, fea_dist, dataset,
                                       d_idx, val_list[idx],
                                       output_figs_dir, 10)

                    # Save visualization of probability that patch will be rejected
                    if True: # i_batch == 0:
                        pp = torch.cat(patches_proba)
                        heat = torchvision.utils.make_grid(pp, nrow=30)
                        alpha = torch.ones([1, heat.shape[1], heat.shape[2]]) * 0.5
                        heat = torch.cat([heat, alpha]).numpy()
                        heat = np.transpose(heat, (1,2,0))
                        fig1 = plt.figure(dpi=300, figsize=[12, 6])
                        plt.clf()
                        ax1 = plt.subplot(1, 2, 1)
                        show(torchvision.utils.make_grid(torch.cat(patches_1), nrow=30))
                        h1 = plt.imshow(heat)
                        divider = make_axes_locatable(ax1)
                        cax = divider.append_axes('right', size='5%', pad=0.05)
                        #fig1.colorbar(h1, cax=cax, orientation='vertical')
                        mpl.colorbar.ColorbarBase(cax, cmap=plt.cm.jet)

                        ax2 = plt.subplot(1, 2, 2)
                        show(torchvision.utils.make_grid(torch.cat(patches_2), nrow=30))
                        h2 = plt.imshow(heat)
                        divider = make_axes_locatable(ax2)
                        cax = divider.append_axes('right', size='5%', pad=0.05)
                        mpl.colorbar.ColorbarBase(cax, cmap=plt.cm.jet)

                        plt.savefig(os.path.join(output_figs_dir, val_list[idx]) + "_proba_reject.png")
                        plt.close(fig1)

                nn_idxs = []
                anchor_fea = []
                pos_fea = []
                neg_fea = []
                coords1 = []
                coords2 = []
                patches_1 = []
                patches_2 = []
                patches_proba = []

                idx = idx + 1
        else:
            anchor_fea = []
            pos_fea = []
            neg_fea = []
            coords1 = []
            coords2 = []
            patches_1 = []
            patches_2 = []
            patches_proba = []

    losses = np.array(losses).reshape(-1)
    writer.add_scalar('data/val_loss', np.mean(losses), val_step)

    if not onlyloss:
        dists = np.concatenate(dists)
        writer.add_histogram('dist2d_first', dists[:, 0], val_step)
        writer.add_histogram('dist2d_mean_top10', np.mean(dists, axis=1), val_step)


def loadImageAndSift(img_name, contrastThreshold=0.04, edgeThreshold=10,
                     sigma=1.6, nfeatures=5000, shape=None):
    img = cv2.imread(img_name)

    if shape is not None:
        print("shape", img_name, shape)
        img = cv2.resize(img, shape)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create(nfeatures=nfeatures,
                                       contrastThreshold=contrastThreshold,
                                       edgeThreshold=edgeThreshold,
                                       sigma=sigma)
    kp, ds_sift = sift.detectAndCompute(gray, None)

    patchsize = 32
    coords = getCoordsAtSiftKeypoints(kp)
    sel = np.logical_and(coords > patchsize, coords < (np.array(img.shape[:2]).reshape(1, 2) - patchsize))
    sel = np.logical_and(sel[:, 0], sel[:, 1])
    nkp = []
    for idx in range(0, sel.shape[0]):
        if sel[idx]:
            nkp.append(kp[idx])
    kp = nkp
    ds_sift = ds_sift[sel]

    img = np.flip(img, 2)  # flip channels as opencv treats them as BGR

    return img, kp, ds_sift


def getCoordsAtSiftKeypoints(kp):
    coords = []
    for key in kp:
        coords.append(np.array([key.pt[1], key.pt[0]]))
    coords = np.array(coords)
    return coords


def getPatchesFromImg(img, coords, fov, maxres=2048, patchsize=64):
    patches = generatePatchesImg(img, img.shape, coords, fov, maxres=maxres, radius=int(patchsize / 2))
    patches = np.array(patches)
    patches = torch.from_numpy(patches).float().permute(0, 3, 1, 2)
    return patches


def show(img):
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')


def detectSiftAndDescribe(img1_name, img2_name, fov1, fov2, net, device,
                          contrastThreshold1=0.04, edgeThreshold1=10, sigma1=1.6,
                          contrastThreshold2=0.04, edgeThreshold2=10, sigma2=1.6):

    img1, kp1, ds1_sift = loadImageAndSift(img1_name, contrastThreshold1, edgeThreshold1, sigma1)
    img2, kp2, ds2_sift = loadImageAndSift(img2_name, contrastThreshold2, edgeThreshold2, sigma2)

    coords1 = getCoordsAtSiftKeypoints(kp1)
    coords2 = getCoordsAtSiftKeypoints(kp2)

    patches1 = getPatchesFromImg(img1, coords1, fov1)
    patches2 = getPatchesFromImg(img2, coords2, fov2)

    batchsize = 900
    p1 = []
    p2 = []
    for idx in range(0, patches1.shape[0], batchsize):
        p1_fea = net.forward_photo(patches1[idx:idx + batchsize].to(device))
        p1_fea = p1_fea.detach().cpu().numpy()
        p1.append(p1_fea)
    for idx in range(0, patches2.shape[0], batchsize):
        p2_fea = net.forward_render(patches2[idx:idx + batchsize].to(device))
        p2_fea = p2_fea.detach().cpu().numpy()
        p2.append(p2_fea)

    p1 = np.concatenate(p1)
    p2 = np.concatenate(p2)

    p1 = p1.reshape(p1.shape[0], p1.shape[1])
    p2 = p2.reshape(p2.shape[0], p2.shape[1])

    return img1, img2, kp1, kp2, ds1_sift, ds2_sift, p1, p2, patches1, patches2


def plotMatches(img1, kp1, img2, kp2, good, matches_outfile,
                mask=None, show=False):
    fig2 = plt.figure(dpi=600)
    ax = plt.gca()
    if mask is not None:
        mask = mask.ravel().tolist()
    distances = []
    for m in good:
        distances.append(m.distance)
    distances = np.array(distances)
    norm = mpl.colors.Normalize()
    colors = np.floor(plt.cm.jet(norm(distances)) * 255)

    img_matches = drawMatches(img1, kp1, img2, kp2, good, mask, colors)

    plt.imshow(img_matches, 'gray')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    mpl.colorbar.ColorbarBase(cax, cmap=plt.cm.jet, norm=norm)

    plt.title('')
    if show:
        plt.show()
    if matches_outfile is not None:
        plt.savefig(matches_outfile)
        plt.close(fig2)


def estimatePoseAndPlot(img1, img2, desc1, desc2, kp1, kp2, patches1, patches2, P1, P2, RT1, RT2, img2_depth, patches_outfile, matches_outfile, lowe_ratio=0.8, save_patches=False, fundam_filter=False, best_buddy=False, show=False):
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    print(desc1.shape[0], desc2.shape[0])
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desc1, desc2, k=2)
    matches = sorted(matches, key = lambda x:x[0].distance, reverse=True) #lambda x:x[0].distance/x[1].distance)

    if best_buddy:
        matches2 = flann.knnMatch(desc2, desc1, k=1)

    count_matches_raw = len(matches)
    # store all the good matches as per Lowe's ratio test.
    good = []
    query_idxs = []
    train_idxs = []
    count = 0

    for m, n in matches:
        if m.distance < lowe_ratio*n.distance:  # and m.distance > 25: 0.999
            query_idxs.append(m.queryIdx)
            train_idxs.append(m.trainIdx)
            count += 1
            good.append(m)

    if best_buddy:
        #filter ambiguous
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
    query_idxs = np.array(query_idxs)
    train_idxs = np.array(train_idxs)

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

        res, R, t, mask = poseFrom2D3D(src_pts, P1, img1.shape,
                                       dst_pts, P2, RT2, img2_depth)
        count_matches_pose = int(np.sum(mask))
        mparts = os.path.splitext(matches_outfile)
        matches_outfile_good = mparts[0] + "_good" + mparts[1]
        plotMatches(img1, kp1, img2, kp2, good, matches_outfile_good, show=show)
        plotMatches(img1, kp1, img2, kp2, good, matches_outfile, mask, show=show)

    return res, R, t, count_matches_raw, count_matches_good, count_matches_pose


def calculateErrors(R_gt, t_gt, R_est, t_est):
    trace = (np.trace(np.dot(np.linalg.inv(R_gt), R_est)) - 1.0) / 2.0
    trace = min(trace, 1.0)
    orient_err = (np.arccos(trace) * 180.0) / np.pi
    t_err = np.linalg.norm(t_gt - t_est)
    return orient_err, t_err


def matchImages(img1_name, img2_name, net, device, output_dir):
    net.eval()

    img1_base, img1_ext = os.path.splitext(img1_name)
    img2_base, img2_ext = os.path.splitext(img2_name)

    img1_base = re.sub("_texture", "", img1_base)
    img2_base = re.sub("_texture", "", img2_base)

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    img1_base_f = os.path.basename(img1_base)
    img2_base_f = os.path.basename(img2_base)
    matches_outfile = os.path.join(output_dir, img1_base_f + "-" + img2_base_f + "_matches.png")
    patches_outfile = os.path.join(output_dir, img1_base_f + "-" + img2_base_f + "_patches.png")

    if os.path.isfile(matches_outfile) and os.path.isfile(patches_outfile):
        print("Skipping pair: " + img1_base_f + "-"
              + img2_base_f + " because it already exists.")
        return

    P1_name = img1_base + "_projection.txt"
    P2_name = img2_base + "_projection.txt"

    if os.path.isfile(P1_name) and os.path.isfile(P2_name):
        P1 = FUtil.loadMatrixFromFile(P1_name)
        P2 = FUtil.loadMatrixFromFile(P2_name)

        fov1, fovy1 = FUtil.projectiveToFOV(P1)
        fov2, fovy2 = FUtil.projectiveToFOV(P2)
    else:
        print("Warning: Unable to find FOV, using default 60 degrees.")
        fov1 = (60/180) * np.pi
        fov2 = (60/180) * np.pi

    img1, img2, kp1, kp2, desc1_sift, desc2_sift, desc1, desc2, patches1, patches2 = detectSiftAndDescribe(img1_name, img2_name, fov1, fov2, net, device) #, contrastThreshold1=0.06, edgeThreshold1=5, sigma1=2.0

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    print(desc1.shape[0], desc2.shape[0])
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desc1, desc2, k=2)
    matches = sorted(matches, key = lambda x:x[0].distance) #lambda x:x[0].distance/x[1].distance)

    # store all the good matches as per Lowe's ratio test.
    good = []
    query_idxs = []
    train_idxs = []
    count = 0
    for m, n in matches:
        print(m.distance, n.distance, m.distance / n.distance)
        if m.distance < 0.99*n.distance:
            query_idxs.append(m.queryIdx)
            train_idxs.append(m.trainIdx)
            count += 1
            good.append(m)
    print("number of good matches: ", len(good))

    query_idxs = np.array(query_idxs)
    train_idxs = np.array(train_idxs)
    fig1 = plt.figure(dpi=600)
    plt.subplot(1, 2, 1)
    number_rows = int(np.floor(np.sqrt(query_idxs.shape[0])))
    show(torchvision.utils.make_grid(patches1[query_idxs], nrow=number_rows))
    plt.subplot(1, 2, 2)
    show(torchvision.utils.make_grid(patches2[train_idxs], nrow=number_rows))
    plt.savefig(patches_outfile)
    plt.close(fig1)

    fig2 = plt.figure(dpi=600)
    if len(good) > 10:
        src_pts = np.int32([kp1[m.queryIdx].pt for m in good])
        dst_pts = np.int32([kp2[m.trainIdx].pt for m in good])
        M, mask = cv2.findFundamentalMat(src_pts, dst_pts,
                                         cv2.FM_RANSAC, 3.0)
        mask = mask.ravel().tolist()
        print(M, mask)
        draw_params = dict(matchColor=None,  # draw matches in green color
                           singlePointColor=None,
                           matchesMask=mask,  # draw only inliers
                           flags=2)

        img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
    else:
        draw_params = dict(matchColor=None,  # draw matches in green color
                       singlePointColor=None,  # matchesMask = matchesMask, # draw only inliers
                       flags=2)

        img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
    plt.imshow(img_matches, 'gray')
    plt.savefig(matches_outfile)
    plt.close(fig2)


def matchTest(img1_name, img2_name, net, device, output_dir):
    net.eval()
    img1_base, img1_ext = os.path.splitext(img1_name)
    img2_base, img2_ext = os.path.splitext(img2_name)

    img1_base = re.sub("_texture", "", img1_base)
    img2_base = re.sub("_texture", "", img2_base)

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    img1_base_f = os.path.basename(img1_base)
    img2_base_f = os.path.basename(img2_base)
    matches_outfile = os.path.join(output_dir, img1_base_f + "-" + img2_base_f + "_matches.png")
    patches_outfile = os.path.join(output_dir, img1_base_f + "-" + img2_base_f + "_patches.png")
    matches_outfile_sift = os.path.join(output_dir, img1_base_f + "-" + img2_base_f + "_matches_sift.png")
    patches_outfile_sift = os.path.join(output_dir, img1_base_f + "-" + img2_base_f + "_patches_sift.png")
    matches_outfile_sift_real = os.path.join(output_dir, img1_base_f + "-" + img2_base_f + "_matches_sift_real.png")
    patches_outfile_sift_real = os.path.join(output_dir, img1_base_f + "-" + img2_base_f + "_patches_sift_real.png")
    errors_outfile = os.path.join(output_dir, "errors.txt")

    if os.path.isfile(matches_outfile):  # and os.path.isfile(patches_outfile):
        print("Skipping pair: " + img1_base_f + "-"
              + img2_base_f + " because it already exists.")
        return

    P1_name = img1_base + "_projection.txt"
    P2_name = img2_base + "_projection.txt"

    MV1_name = img1_base + "_modelview.txt"
    MV2_name = img2_base + "_modelview.txt"

    img1_depth_name = img1_base + "_texture_depth.txt.gz"
    img2_depth_name = img2_base + "_texture_depth.txt.gz"

    img1_tex_name = img1_base + "_texture.png"
    img2_tex_name = img2_base + "_texture.png"

    P1 = FUtil.loadMatrixFromFile(P1_name)
    P2 = FUtil.loadMatrixFromFile(P2_name)

    RT1 = (FUtil.loadMatrixFromFile(MV1_name))
    RT2 = (FUtil.loadMatrixFromFile(MV2_name))

    fov1, fovy1 = FUtil.projectiveToFOV(P1)
    fov2, fovy2 = FUtil.projectiveToFOV(P2)

    # detect features for real images
    start = time.time()
    img1, img2, kp1, kp2, desc1_sift, desc2_sift, desc1, desc2, patches1, patches2 = detectSiftAndDescribe(img1_name, img2_name, fov1, fov2, net, device, contrastThreshold1=0.01, edgeThreshold1=5, sigma1=1.6, contrastThreshold2=0.01, edgeThreshold2=5, sigma2=1.6)
    img1t, img2t, kp1t, kp2t, desc1_siftt, desc2_siftt, desc1t, desc2t, patches1t, patches2t = detectSiftAndDescribe(img1_tex_name, img2_tex_name, fov1, fov2, net, device) #, contrastThreshold=0.02, edgeThreshold=20, sigma=1.0
    print("Detecting and describing took: ", time.time() - start)

    start = time.time()
    img2_depth = loadDepth(img2_depth_name)
    img2_depth = cv2.resize(img2_depth, (img2t.shape[1], img2t.shape[0]))
    print("Loading depth took: ", time.time() - start)

    start = time.time()
    res_real = estimatePoseAndPlot(img1, img2, desc1_sift, desc2_sift, kp1, kp2, patches1, patches2, P1, P2, RT1, RT2, img2_depth, patches_outfile_sift_real, matches_outfile_sift_real)
    res_sift_real, R_sift_real, t_sift_real, cr_raw, cr_good, cr_pose = res_real

    res_tex = estimatePoseAndPlot(img1t, img2t, desc1_siftt, desc2_siftt, kp1t, kp2t, patches1t, patches2t, P1, P2, RT1, RT2, img2_depth, patches_outfile_sift, matches_outfile_sift)
    res_sift_tex, R_sift_tex, t_sift_tex, ct_raw, ct_good, ct_pose = res_tex

    res_net = estimatePoseAndPlot(img1, img2t, desc1, desc2t, kp1, kp2t, patches1, patches2t, P1, P2, RT1, RT2, img2_depth, patches_outfile, matches_outfile, lowe_ratio=2.0, best_buddy=False)
    res_net, R_net, t_net, cn_raw, cn_good, cn_pose = res_net
    print("Estimate pose and plot took: ", time.time() - start)

    print("Matching pair", img1_base_f, img2_base_f)
    print("SIFT on rendered (baseline):")
    print(R_sift_tex,  t_sift_tex)
    print("Neural on cross domain:")
    print(R_net, t_net)
    print("Ground truth", RT1)

    R_gt = RT1[:3, :3]
    t_gt = RT1[:3, 3]

    R2_gt = RT2[:3, :3]
    t2_gt = RT2[:3, 3]

    res_sift_t = [-1, -1]
    res_sift_t_gt = [-1, -1]
    res_sift_rea_gt = [-1, -1]
    res_gt = [-1, -1]
    if res_sift_tex and res_net:
        res_sift_t = calculateErrors(R_sift_tex, t_sift_tex, R_net, t_net)
        res_sift_t_gt = calculateErrors(R_gt, t_gt, R_sift_tex, t_sift_tex)
        res_gt = calculateErrors(R_gt, t_gt, R_net, t_net)
    if res_sift_real:
        res_sift_rea_gt = calculateErrors(R_gt, t_gt, R_sift_real, t_sift_real)

    pair_dist = calculateErrors(R_gt, t_gt, R2_gt, t2_gt)

    with open(errors_outfile, "a") as errfile:
        pair = img1_base_f + "-" + img2_base_f
        sift_err_t = str(res_sift_t[0]) + " " + str(res_sift_t[1])
        sift_rea_cnt = str(cr_raw) + " " + str(cr_good) + " " + str(cr_pose)
        sift_err_rea_gt = sift_rea_cnt + " " + str(res_sift_rea_gt[0]) + " " + str(res_sift_rea_gt[1])
        sift_t_gt_cnt = str(ct_raw) + " " + str(ct_good) + " " + str(ct_pose)
        sift_err_t_gt = sift_t_gt_cnt + " " + str(res_sift_t_gt[0]) + " " + str(res_sift_t_gt[1])
        gt_err = str(res_gt[0]) + " " + str(res_gt[1])

        all_pair_dist = str(pair_dist[0]) + " " + str(pair_dist[1])
        all_sift_gt = all_pair_dist + " " + sift_err_rea_gt + " " + sift_err_t_gt

        net_gt_cnt = str(cn_raw) + " " + str(cn_good) + " " + str(cn_pose)
        all_our_gt = net_gt_cnt + " " + sift_err_t + " " + gt_err
        errfile.write(pair + " " + all_sift_gt + " " + all_our_gt + "\n")


def showBatch(anchor, pos, neg):
    fig1 = plt.figure(dpi=300, figsize=[18, 6])
    ax1 = plt.subplot(1, 3, 1)
    show(torchvision.utils.make_grid(anchor, nrow=30))

    ax2 = plt.subplot(1, 3, 2)
    show(torchvision.utils.make_grid(pos, nrow=30))

    ax3 = plt.subplot(1, 3, 3)
    show(torchvision.utils.make_grid(neg, nrow=30))
    plt.show()


def visualizeHardPatchesAndDist(anchor, pos, neg, c1, c2, cneg, id,
                                anchor_out, pos_out, neg_out,
                                hard_negs_idx, random_hard_negs_idx, keep, random_keep, an_neg_dist,
                                dist_type):

    img_an = torchvision.utils.make_grid(anchor[keep], nrow=15).detach().cpu().numpy()
    img_pos = torchvision.utils.make_grid(pos[keep], nrow=15).detach().cpu().numpy()
    img_neg = torchvision.utils.make_grid(pos[hard_negs_idx], nrow=15).detach().cpu().numpy()

    plt.figure(1)
    plt.clf()
    plt.subplot(1, 3, 1)
    plt.imshow(np.transpose(img_an, (1,2,0)), interpolation='nearest')
    plt.subplot(1, 3, 2)
    plt.imshow(np.transpose(img_pos, (1,2,0)), interpolation='nearest')
    plt.subplot(1, 3, 3)
    plt.imshow(np.transpose(img_neg, (1,2,0)), interpolation='nearest')

    if dist_type == '2D':
        #print(an_neg_dist.shape)
        norm = mpl.colors.Normalize()
        colors = plt.cm.jet(norm(an_neg_dist.detach().cpu().numpy()))

        c1c = c1[keep].detach().cpu().numpy()
        c2c = c2[hard_negs_idx].detach().cpu().numpy()
        idk = id[keep]
        unique_id = torch.unique(id)
        cnt_plot = 0
        for idx in range(0, unique_id.shape[0]):
            #plot patches
            sel = idk == unique_id[idx]
            if torch.sum(sel) == 0:
                continue
            if cnt_plot < 3:
                cnt_plot += 1
                plt.figure(3 + cnt_plot)
                plt.clf()
                hard_negs_idx_b = hard_negs_idx[sel]
                c1cb = c1c[sel.cpu().numpy().astype(np.bool)]
                c2cb = c2c[sel.cpu().numpy().astype(np.bool)]
                ymax = np.max(c2c[:, 1])
                ax = plt.axes([0,0,1,1], frameon=False)
                ax.set_axis_off()
                ax.set_xlim(0,np.max(c2c[:, 0]))
                ax.set_ylim(0, ymax)

                pos_patches = np.transpose(pos[keep[sel]].cpu().numpy(), [0, 2, 3, 1])
                neg_patches = np.transpose(pos[hard_negs_idx_b].cpu().numpy(), [0, 2, 3, 1])
                for idx in range(0, pos_patches.shape[0]):
                    plt.imshow(pos_patches[idx], extent=(c1cb[idx, 0] - 32, c1cb[idx, 0] + 32, ymax - (c1cb[idx, 1] + 32), ymax - (c1cb[idx, 1] - 32)))
                    plt.imshow(neg_patches[idx], extent=(c2cb[idx, 0] - 32, c2cb[idx, 0] + 32, ymax - (c2cb[idx, 1] + 32), ymax - (c2cb[idx, 1] - 32)))

                plt.scatter(c1cb[:, 0], ymax - c1cb[:, 1])
                plt.scatter(c2cb[:, 0], ymax - c2cb[:, 1], c='green')
                for i in range(0, c1cb.shape[0]):
                    plt.plot([c1cb[i, 0], c2cb[i, 0]], [ymax - c1cb[i, 1], ymax - c2cb[i, 1]]) #, color=colors[keep][sel][i]

        plt.draw()
        plt.pause(0.0001)

    print(' '.join(['{:.3f}'.format(x) for x in anchor_out[0, :16]]))
    print(' '.join(['{:.3f}'.format(x) for x in pos_out[0, :16]]))
    print(' '.join(['{:.3f}'.format(x) for x in neg_out[hard_negs_idx[0], :16]]))
    print('=========================')


def visualizeHardPatchesAndDistRandomNeg(anchor, pos, neg, c1, c2, cneg, id,
                                anchor_out, pos_out, neg_out,
                                hard_negs_idx, random_hard_negs_idx, keep, random_keep, an_neg_dist,
                                dist_type):

    img_an_for_neg = torchvision.utils.make_grid(anchor[random_keep], nrow=15).detach().cpu().numpy()
    img_pos_for_neg = torchvision.utils.make_grid(pos[random_keep], nrow=15).detach().cpu().numpy()
    img_neg_ran = torchvision.utils.make_grid(neg[random_hard_negs_idx], nrow=15).detach().cpu().numpy()

    plt.figure(1)
    plt.clf()
    plt.subplot(1, 3, 1)
    plt.imshow(np.transpose(img_an_for_neg, (1,2,0)), interpolation='nearest')
    plt.subplot(1, 3, 2)
    plt.imshow(np.transpose(img_pos_for_neg, (1,2,0)), interpolation='nearest')
    plt.subplot(1, 3, 3)
    plt.imshow(np.transpose(img_neg_ran, (1,2,0)), interpolation='nearest')
    plt.draw()

    if dist_type == '2D':
        norm = mpl.colors.Normalize()
        colors = plt.cm.jet(norm(an_neg_dist.detach().cpu().numpy()))

        c1c = c1[random_keep].detach().cpu().numpy()
        cnegc = cneg[random_hard_negs_idx].detach().cpu().numpy()
        idk = id[random_keep]
        unique_id = torch.unique(id)
        cnt_plot = 0
        for idx in range(0, unique_id.shape[0]):
            # plot patches
            sel = idk == unique_id[idx]
            if torch.sum(sel) == 0:
                continue
            if cnt_plot < 3:
                cnt_plot += 1
                plt.figure(3 + cnt_plot)
                plt.clf()
                random_keep_b = random_keep[sel]
                random_hard_negs_idx_b = random_hard_negs_idx[sel]
                c1cb = c1c[sel.cpu().numpy().astype(np.bool)]
                cnegcb = cnegc[sel.cpu().numpy().astype(np.bool)]
                ymax = np.max(cnegc[:, 1])
                ax = plt.axes([0,0,1,1], frameon=False)
                ax.set_axis_off()
                ax.set_xlim(0,np.max(cnegc[:, 0]))
                ax.set_ylim(0, ymax)

                pos_patches = np.transpose(pos[random_keep_b].cpu().numpy(), [0, 2, 3, 1])
                neg_patches = np.transpose(neg[random_hard_negs_idx_b].cpu().numpy(), [0, 2, 3, 1])
                for idx in range(0, pos_patches.shape[0]):
                    plt.imshow(pos_patches[idx], extent=(c1cb[idx, 0] - 32, c1cb[idx, 0] + 32, ymax - (c1cb[idx, 1] + 32), ymax - (c1cb[idx, 1] - 32)))
                    plt.imshow(neg_patches[idx], extent=(cnegcb[idx, 0] - 32, cnegcb[idx, 0] + 32, ymax - (cnegcb[idx, 1] + 32), ymax - (cnegcb[idx, 1] - 32)))

                plt.scatter(c1cb[:, 0], ymax - c1cb[:, 1])
                plt.scatter(cnegcb[:, 0], ymax - cnegcb[:, 1], c='red')

        plt.draw()
        plt.pause(0.0001)

    print(' '.join(['{:.3f}'.format(x) for x in anchor_out[0, :16]]))
    print(' '.join(['{:.3f}'.format(x) for x in pos_out[0, :16]]))
    print(' '.join(['{:.3f}'.format(x) for x in neg_out[hard_negs_idx[0], :16]]))
    print('=========================')


def getGPU():
    freeGpu = subprocess.check_output('nvidia-smi -q | grep "Minor\|Processes" | grep "None" -B1 | tr -d " " | cut -d ":" -f2 | sed -n "1p"', shell=True)
    if len(freeGpu) == 0:
        raise RuntimeError('No free GPU available!')
    os.environ['CUDA_VISIBLE_DEVICES'] = freeGpu.decode().strip()
    print("Selected GPU:", os.environ['CUDA_VISIBLE_DEVICES'])


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')

    parser = buildArgumentParser()
    args = parser.parse_args()

    if args.autoselect_gpu:
        getGPU()

    dataset_list = loadListFile(args.dataset_list)
    val_dataset_list = loadListFile(args.val_dataset_list)

    output_figs_dir = ""
    save_figs = False
    if args.save_figs:
        output_figs_dir = args.save_figs[0]
        save_figs = True
    pos_thr = args.positive_threshold
    reject = True
    if args.no_reject:
        reject = False

    gpu_num = 0

    device = torch.device("cuda:" + str(gpu_num) if args.cuda else "cpu")

    timestamp = strftime("%Y-%m-%d_%H:%M:%S", gmtime())
    if args.restore:
        timestamp = args.restore
    elif args.name:
        timestamp = args.name

    log_dir = os.path.join(args.log_dir, "runs", timestamp)
    writer = SummaryWriter(log_dir=log_dir)

    model_name = args.architecture
    if args.restore:
        models_path = os.path.join(log_dir, "models")
        if args.model_name:
            model_path = os.path.join(models_path, args.model_name)
        else:
            model_path = findLatestModelPath(models_path)
        model_name = os.path.basename(model_path).split("_epoch_")[0]

    module = __import__("training").Architectures
    net_class = getattr(module, model_name)
    if args.needles > 0:
        net = net_class(
            normalize_output=args.normalize_output, needles=args.needles
        )
    else:
        net = net_class(
            normalize_output=args.normalize_output
        )

    if args.cuda:
        net = net.to(device)

    n_epochs = 10
    step = 0
    loss = 0
    start_ep = 0

    triplet_loss = nn.TripletMarginLoss(margin=args.margin, reduction='none')
    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)

    start_step = 0
    # when this is zero, semi-hard mining using randomly chosen negative
    # is used
    hard_dist_coeff = 0
    if args.restore:
        checkpoint = torch.load(model_path, map_location=device)
        updated_weights = False
        #if 'conv2.weight' in checkpoint['model_state_dict']:
        #    x = checkpoint['model_state_dict']['conv2.weight']
        #    if x.dim() > 2:
                #update to correct shape
        #        checkpoint['model_state_dict']['conv2.weight'] = x.reshape(x.size(0), -1)
        #        updated_weights = True
        net.load_state_dict(checkpoint['model_state_dict'])
        if not updated_weights:
            # we can use the old optimizer, otherwise start from scratch
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_ep = checkpoint['epoch']
        step = checkpoint['step']
        start_step = step
        loss = checkpoint['loss']
        if args.adaptive_mining and 'hard_dist_coeff' in checkpoint:
            hard_dist_coeff = checkpoint['hard_dist_coeff']
        print("Restored model: " + model_path)

    if args.adaptive_mining:
        # prefer semihard mining coeff set from command line
        hard_dist_coeff = args.adaptive_mining_coeff

    train_keypoints = False
    if isinstance(net, MultimodalKeypointPatchNet5lShared2l):
        if not args.hardnet:
            raise RuntimeError("Training keypoints without \
                                hardnet is not yet implemented.")
        train_keypoints = True

    dset = args.dataset_path

    cache_dir_proc = os.path.join(args.cache_dir, timestamp)
    cache_dir_val = os.path.join(cache_dir_proc, "val")
    val_dataset_cache = MultimodalPatchesCache(
        cache_dir_val, dset, val_dataset_list, args.cuda,
        batch_size=args.batch_size,
        num_workers=args.num_workers, renew_frequency=5,
        numneg=1, pos_thr=pos_thr,
        reject=True, mode='eval',
        rejection_radius=args.rejection_radius,
        dist_type=args.distance_type,
        use_depth=args.use_depth,
        use_normals=args.use_normals,
        use_silhouettes=args.use_silhouettes,
        color_jitter=args.color_jitter,
        patch_radius=args.patch_radius,
        greyscale=args.greyscale,
        maxres=args.maxres,
        needles=args.needles,
        render_only=args.render_only)
    val_dataset = CachedMultimodalPatchesDataset(val_dataset_cache)
    val = MultimodalPatchesDatasetAll.getListAll(
        dset, val_dataset_list, 'eval', args.rejection_radius_position
    )
    if args.validate:
        if args.sift:
            validate_all_sift(net, val_dataset, val, device, triplet_loss,
                              step, args.num_workers_loader, output_figs_dir,
                              batchsize=args.batch_size, savefigs=save_figs)
            exit(0)
        else:
            validate_all(net, val_dataset, val, device, triplet_loss,
                         step, args.num_workers_loader, output_figs_dir,
                         batchsize=args.batch_size, savefigs=save_figs)
            writer.close()
            exit(0)

        exit(0)
    if args.match_images:
        matchTest(args.match_images[0], args.match_images[1], net, device, args.match_images[2])
        exit(0)

    numneg = 1

    cache_dir_train = os.path.join(cache_dir_proc, "train")

    for ep in range(start_ep, args.num_epochs):
        net.train()
        cache_once = False
        cache_limit = 200
        if args.cache_once:
            cache_once = True
            cache_limit = args.cache_once
        dataset_cache = MultimodalPatchesCache(
            cache_dir_train, dset, dataset_list, args.cuda, batch_size=args.batch_size,
            num_workers=args.num_workers, renew_frequency=5,
            numneg=numneg,
            pos_thr=pos_thr, reject=reject,
            rejection_radius=args.rejection_radius,
            dist_type=args.distance_type,
            use_depth=args.use_depth,
            use_normals=args.use_normals,
            use_silhouettes=args.use_silhouettes,
            color_jitter=args.color_jitter,
            patch_radius=args.patch_radius,
            greyscale=args.greyscale,
            scale_jitter=args.scale_jitter,
            uniform_negatives=args.uniform_negatives,
            maxres=args.maxres,
            needles=args.needles,
            render_only=args.render_only,
            cache_once=cache_once,
            maxitems=cache_limit
            )
        print("Main: creating dataset...")
        dataset = CachedMultimodalPatchesDataset(dataset_cache)
        dataloader = DataLoader(
            dataset, batch_size=1, shuffle=False,
            num_workers=args.num_workers_loader, pin_memory=args.cuda
        )
        print("Running training loop...")
        while True:
            for i_batch, data in enumerate(tqdm(dataloader)):
                optimizer.zero_grad()
                validation_step = 10000
                if step % validation_step == 0: # and step != start_step:
                    # update hard_dist_coeff
                    if args.adaptive_mining:
                        # enlarge hard dist coeff by step%
                        hard_dist_coeff = min(hard_dist_coeff + args.adaptive_mining_step, 1.0)
                    print("Saving model...")
                    saveModel(log_dir, net, ep, step, optimizer, loss, device, hard_dist_coeff)
                    print("Validation with dist 2D...")
                    validate_all(net, val_dataset, val, device, triplet_loss,
                                 step, 0, output_figs_dir,
                                 batchsize=args.batch_size, savefigs=save_figs,
                                 onlyloss=False)
                    net.train()

                anchor = data[0][0].to(device, non_blocking=True)
                pos = data[1][0].to(device, non_blocking=True)
                neg = data[2][0].reshape(-1, data[2][0].shape[2], data[2][0].shape[3], data[2][0].shape[4]).to(device, non_blocking=True)
                anchor_r = data[3][0].to(device, non_blocking=True)
                pos_p = data[4][0].to(device, non_blocking=True)
                neg_p = data[5][0].reshape(-1, data[5][0].shape[2], data[5][0].shape[3], data[5][0].shape[4]).to(device, non_blocking=True)
                # coords per patch in 3D
                c1 = data[6][0].to(device, non_blocking=True)
                c2 = data[7][0].to(device, non_blocking=True)
                cneg = data[8][0].to(device, non_blocking=True)
                id = data[9][0].to(device, non_blocking=True)

                anchor_out = net.forward_photo(anchor)
                pos_out = net.forward_render(pos)

                do_plot = args.do_plot and (step % args.plot_frequency == 0)

                if args.hardnet:
                    if train_keypoints:
                        anchor_score = anchor_out[1]
                        pos_score = pos_out[1]
                        anchor_out = anchor_out[0]
                        pos_out = pos_out[0]
                    # we need just anchor and pos, we dont need neg
                    # print('----batch----')
                    # print(' '.join(['{:.3f}'.format(x) for x in anchor_out[0, :16]]))
                    # print(' '.join(['{:.3f}'.format(x) for x in pos_out[0, :16]]))
                    # print('-------------')

                    loss_1p = loss_HardNetWithDist(anchor_out, pos_out, c1, c2,
                                        margin=args.margin,
                                        anchor_swap=args.hardnet_anchorswap,
                                        anchor_ave=args.hardnet_anchorave,
                                        batch_reduce=args.hardnet_batch_reduce,
                                        loss_type=args.hardnet_loss,
                                        show=do_plot, an_img=anchor, pos_img=pos,
                                        filter_dist=args.hardnet_filter_dist,
                                        min_dist=args.hardnet_filter_dist_values[0],
                                        max_dist=args.hardnet_filter_dist_values[1],
                                        device=device,
                                        hardest=args.hardest_mining,
                                        hard_dist_coeff=hard_dist_coeff)
                    loss1 = loss_1p.clone()

                    if args.loss_aux:
                        pos_p_out = net.forward_photo(pos_p)
                        anchor_r_out = net.forward_render(anchor_r)
                        if train_keypoints:
                            anchor_r_score = anchor_r_out[1]
                            anchor_r_out = anchor_r_out[0]
                            pos_p_score = pos_p_out[1]
                            pos_p_out = pos_p_out[0]
                        loss_aux_photo = loss_HardNetWithDist(anchor_out, pos_p_out, c1, c2,
                                            margin=args.margin,
                                            anchor_swap=args.hardnet_anchorswap,
                                            anchor_ave=args.hardnet_anchorave,
                                            batch_reduce=args.hardnet_batch_reduce,
                                            loss_type=args.hardnet_loss,
                                            show=do_plot, an_img=anchor, pos_img=pos,
                                            filter_dist=args.hardnet_filter_dist,
                                            min_dist=args.hardnet_filter_dist_values[0],
                                            max_dist=args.hardnet_filter_dist_values[1],
                                            device=device,
                                            hardest=args.hardest_mining,
                                            hard_dist_coeff=hard_dist_coeff)

                        loss_aux_render = loss_HardNetWithDist(anchor_r_out, pos_out, c1, c2,
                                            margin=args.margin,
                                            anchor_swap=args.hardnet_anchorswap,
                                            anchor_ave=args.hardnet_anchorave,
                                            batch_reduce=args.hardnet_batch_reduce,
                                            loss_type=args.hardnet_loss,
                                            show=do_plot, an_img=anchor, pos_img=pos,
                                            filter_dist=args.hardnet_filter_dist,
                                            min_dist=args.hardnet_filter_dist_values[0],
                                            max_dist=args.hardnet_filter_dist_values[1],
                                            device=device,
                                            hardest=args.hardest_mining,
                                            hard_dist_coeff=hard_dist_coeff)

                        if train_keypoints:
                            loss_aux_photo = (anchor_score * pos_p_score * loss_aux_photo) + loss_aux_photo
                            loss_aux_render = (anchor_r_score * pos_score * loss_aux_render) + loss_aux_render

                        loss_aux_photo = torch.mean(loss_aux_photo)
                        loss_aux_render = torch.mean(loss_aux_render)
                        loss_aux = loss_aux_photo + loss_aux_render

                    if args.uniform_negatives:
                        random_neg_out = net.forward_render(neg)
                        if train_keypoints:
                            random_neg_out = random_neg_out[0]
                        anchor_out_for_neg = anchor_out.clone()
                        pos_out_for_neg = pos_out.clone()
                        random_hard_negs_idx, random_keep, random_an_neg_dist = select_HardNetMultimodal(anchor_out_for_neg, random_neg_out, c1, cneg, id,
                                                             min_dist=args.hardnet_filter_dist_values[0],
                                                             max_dist=args.hardnet_filter_dist_values[1],
                                                             device=device,
                                                             dist_type=args.distance_type,
                                                             plot=do_plot,
                                                             as_hard_as_possible=(not args.hardnet_orig),
                                                             hardest=args.hardest_mining,
                                                             margin=args.margin,
                                                             hard_dist_coeff=hard_dist_coeff)

                        anchor_out_for_neg = anchor_out_for_neg[random_keep]
                        pos_out_for_neg = pos_out_for_neg[random_keep]
                        random_hard_negs_idx = random_hard_negs_idx[random_keep]
                        random_neg_out = random_neg_out[random_hard_negs_idx]

                        loss_1n = triplet_loss(anchor_out_for_neg, pos_out_for_neg, random_neg_out)
                        loss1 += loss_1n
                        loss_1n = torch.mean(loss_1n)

                    if train_keypoints:
                        loss1 = (anchor_score * pos_score * loss1) + loss1
                    loss1 = torch.mean(loss1)
                    loss_1p = torch.mean(loss_1p)

                    if not args.no_symmetric:
                        # apply symmetric loss to swapped render-photo modality
                        anchor_r_out = net.forward_render(anchor_r)
                        pos_p_out = net.forward_photo(pos_p)

                        if train_keypoints:
                            anchor_r_score = anchor_r_out[1]
                            pos_p_score = pos_p_out[1]
                            anchor_r_out = anchor_r_out[0]
                            pos_p_out = pos_p_out[0]

                        loss_2p = loss_HardNetWithDist(anchor_r_out, pos_p_out, c1, c2,
                                            margin=args.margin,
                                            anchor_swap=args.hardnet_anchorswap,
                                            anchor_ave=args.hardnet_anchorave,
                                            batch_reduce=args.hardnet_batch_reduce,
                                            loss_type=args.hardnet_loss,
                                            show=do_plot, an_img=anchor_r, pos_img=pos_p,
                                            filter_dist=args.hardnet_filter_dist,
                                            min_dist=args.hardnet_filter_dist_values[0],
                                            max_dist=args.hardnet_filter_dist_values[1],
                                            device=device,
                                            hardest=args.hardest_mining,
                                            hard_dist_coeff=hard_dist_coeff)
                        loss2 = loss_2p.clone()

                        if args.uniform_negatives:
                            random_neg_p_out = net.forward_photo(neg_p)
                            if train_keypoints:
                                random_neg_p_out = random_neg_p_out[0]
                            anchor_r_out_for_neg = anchor_r_out.clone()
                            pos_p_out_for_neg = pos_p_out.clone()
                            random_hard_negs_idx, random_keep, random_an_neg_dist = select_HardNetMultimodal(anchor_r_out_for_neg, random_neg_p_out, c1, cneg, id,
                                                                 min_dist=args.hardnet_filter_dist_values[0],
                                                                 max_dist=args.hardnet_filter_dist_values[1],
                                                                 device=device,
                                                                 dist_type=args.distance_type,
                                                                 plot=do_plot,
                                                                 as_hard_as_possible=(not args.hardnet_orig),
                                                                 hardest=args.hardest_mining,
                                                                 margin=args.margin,
                                                                 hard_dist_coeff=hard_dist_coeff)

                            anchor_r_out_for_neg = anchor_r_out_for_neg[random_keep]
                            pos_p_out_for_neg = pos_p_out_for_neg[random_keep]
                            random_hard_negs_idx = random_hard_negs_idx[random_keep]
                            random_neg_p_out = random_neg_p_out[random_hard_negs_idx]

                            loss_2n = triplet_loss(anchor_r_out_for_neg, pos_p_out_for_neg, random_neg_p_out)
                            loss2 += loss_2n
                            loss_2n = torch.mean(loss_2n)

                        if train_keypoints:
                            loss2 = (anchor_r_score * pos_p_score * loss2) + loss2
                        loss2 = torch.mean(loss2)
                        loss_2p = torch.mean(loss_2p)

                        loss = loss1 + loss2
                    else:
                        loss = loss1
                    if args.loss_aux:
                        loss = loss + loss_aux

                else:
                    anchor_out, pos_out, neg_out = net(anchor, pos, neg)
                    loss1 = triplet_loss(anchor_out, pos_out, neg_out)
                    if not args.no_symmetric:
                        anchor_r_out = net.forward_render(anchor_r)
                        pos_p_out = net.forward_photo(pos_p)
                        neg_p_out = net.forward_photo(neg_p)
                        loss2 = triplet_loss(anchor_r_out, pos_p_out, neg_p_out)
                        loss = loss1 + loss2
                    else:
                        loss = loss1

                writer.add_scalar('data/loss', loss.detach().cpu(), step)
                writer.add_scalar('data/loss_1p', loss_1p.detach().cpu(), step)
                if args.loss_aux:
                    writer.add_scalar('data/loss_aux', loss_aux.detach().cpu(), step)
                    writer.add_scalar('data/loss_aux_photo', loss_aux_photo.detach().cpu(), step)
                    writer.add_scalar('data/loss_aux_render', loss_aux_render.detach().cpu(), step)
                if args.uniform_negatives:
                    writer.add_scalar('data/loss_1n', loss_1n.detach().cpu(), step)
                if not args.no_symmetric:
                    writer.add_scalar('data/loss_2p', loss_2p.detach().cpu(), step)
                    if args.uniform_negatives:
                        writer.add_scalar('data/loss_2n', loss_2n.detach().cpu(), step)
                writer.add_scalar('data/batchsize', anchor.shape[0], step)
                writer.add_scalar('data/hard_dist_coef', hard_dist_coeff, step)
                loss.backward()
                optimizer.step()
                step = step + 1

            cache_changed, cache_done_all = dataset_cache.restart()
            if cache_changed:
                del dataloader
                dataloader = None
                del dataset
                dataset = None
                dataset = CachedMultimodalPatchesDataset(dataset_cache)
                dataloader = DataLoader(
                    dataset, batch_size=1, shuffle=False,
                    num_workers=args.num_workers_loader, pin_memory=False
                )
            if cache_done_all:
                break

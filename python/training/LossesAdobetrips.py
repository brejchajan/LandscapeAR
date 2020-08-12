# @Author: Jan Brejcha <ibrejcha>
# @Date:   2020-08-11T17:50:05+02:00
# @Email:  ibrejcha@fit.vutbr.cz; brejchaja@gmail.com
# @Project: LandscapeAR
# @Last modified by:   ibrejcha
# @Last modified time: 2020-08-12T11:45:40+02:00
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
import torch.nn as nn
import sys
import matplotlib.pyplot as plt
import torchvision
import numpy as np

def distance_matrix_vector(anchor, positive):
    """Given batch of anchor descriptors and positive descriptors calculate distance matrix"""
    #return 0.5*(1-torch.matmul(torch.t(anchor), positive))
    d1_sq = torch.sum(anchor * anchor, dim=1).unsqueeze(-1)
    d2_sq = torch.sum(positive * positive, dim=1).unsqueeze(-1)

    eps = 1e-6
    return torch.sqrt((d1_sq.repeat(1, positive.size(0)) + torch.t(d2_sq.repeat(1, anchor.size(0)))
                      - 2.0 * torch.bmm(anchor.unsqueeze(0), torch.t(positive).unsqueeze(0)).squeeze(0))+eps)

def distance_matrix_vector_negative(anchor, negative):
    numneg = negative.shape[0] // anchor.shape[0]
    #repeat each anchor numneg times
    anchor_rep = anchor.unsqueeze(1).repeat(1, numneg, 1).reshape(negative.shape[0], anchor.shape[1])
    #calculate the norm
    dist = torch.norm(anchor_rep - negative, dim=1).reshape(anchor.shape[0], numneg)
    return dist


def distance_vectors_pairwise(anchor, positive, negative = None):
    """Given batch of anchor descriptors and positive descriptors calculate distance matrix"""

    a_sq = torch.sum(anchor * anchor, dim=1)
    p_sq = torch.sum(positive * positive, dim=1)

    eps = 1e-8
    d_a_p = torch.sqrt(a_sq + p_sq - 2*torch.sum(anchor * positive, dim = 1) + eps)
    if negative is not None:
        n_sq = torch.sum(negative * negative, dim=1)
        d_a_n = torch.sqrt(a_sq + n_sq - 2*torch.sum(anchor * negative, dim = 1) + eps)
        d_p_n = torch.sqrt(p_sq + n_sq - 2*torch.sum(positive * negative, dim = 1) + eps)
        return d_a_p, d_a_n, d_p_n
    return d_a_p
def loss_random_sampling(anchor, positive, negative, anchor_swap = False, margin = 1.0, loss_type = "triplet_margin"):
    """Loss with random sampling (no hard in batch).
    """

    assert anchor.size() == positive.size(), "Input sizes between positive and negative must be equal."
    assert anchor.size() == negative.size(), "Input sizes between positive and negative must be equal."
    assert anchor.dim() == 2, "Inputd must be a 2D matrix."
    eps = 1e-8
    (pos, d_a_n, d_p_n) = distance_vectors_pairwise(anchor, positive, negative)
    if anchor_swap:
       min_neg = torch.min(d_a_n, d_p_n)
    else:
       min_neg = d_a_n

    if loss_type == "triplet_margin":
        loss = torch.clamp(margin + pos - min_neg, min=0.0)
    elif loss_type == 'softmax':
        exp_pos = torch.exp(2.0 - pos);
        exp_den = exp_pos + torch.exp(2.0 - min_neg) + eps;
        loss = - torch.log( exp_pos / exp_den )
    elif loss_type == 'contrastive':
        loss = torch.clamp(margin - min_neg, min=0.0) + pos;
    else:
        print ('Unknown loss type. Try triplet_margin, softmax or contrastive')
        sys.exit(1)
    loss = torch.mean(loss)
    return loss

def loss_L2Net(anchor, positive, anchor_swap = False,  margin = 1.0, loss_type = "triplet_margin"):
    """L2Net losses: using whole batch as negatives, not only hardest.
    """

    assert anchor.size() == positive.size(), "Input sizes between positive and negative must be equal."
    assert anchor.dim() == 2, "Inputd must be a 2D matrix."
    eps = 1e-8
    dist_matrix = distance_matrix_vector(anchor, positive)
    eye = torch.autograd.Variable(torch.eye(dist_matrix.size(1))).cuda()

    # steps to filter out same patches that occur in distance matrix as negatives
    pos1 = torch.diag(dist_matrix)
    dist_without_min_on_diag = dist_matrix+eye*10
    mask = (dist_without_min_on_diag.ge(0.008)-1)*-1
    mask = mask.type_as(dist_without_min_on_diag)*10
    dist_without_min_on_diag = dist_without_min_on_diag+mask

    if loss_type == 'softmax':
        exp_pos = torch.exp(2.0 - pos1);
        exp_den = torch.sum(torch.exp(2.0 - dist_matrix),1) + eps;
        loss = -torch.log( exp_pos / exp_den )
        if anchor_swap:
            exp_den1 = torch.sum(torch.exp(2.0 - dist_matrix),0) + eps;
            loss += -torch.log( exp_pos / exp_den1 )
    else:
        print ('Only softmax loss works with L2Net sampling')
        sys.exit(1)
    loss = torch.mean(loss)
    return loss


def loss_HardNetNeg(anchor, positive, neg, margin=1.0):
    pos_dist = torch.norm(anchor - positive, dim=1)
    eps = 1e-8
    dist_neg = distance_matrix_vector_negative(anchor, neg) + eps
    min_neg, _ = torch.min(dist_neg, 1)
    loss = torch.clamp(margin + pos_dist - min_neg, min=0.0)
    return torch.mean(loss)


def select_HardNetMultimodal(anchor, positive, c1, c2, id, min_dist, max_dist, device, dist_type, plot=False, as_hard_as_possible=True, min_in_batch=10, hardest=False, margin=1.0, hard_dist_coeff=0.0):
    assert anchor.size() == positive.size(), "Input sizes between positive and negative must be equal."
    assert anchor.dim() == 2, "Inputd must be a 2D matrix."
    eps = 1e-8

    dist_matrix = distance_matrix_vector(anchor, positive) + eps
    pos1 = torch.diag(dist_matrix)

    # our implementation to filter out too close and too far negatives
    #c2 = torch.flip(c2, [1])
    dist_matrix_3D = distance_matrix_vector(c1, c2)
    dist_matrix_3D_mask = (dist_matrix_3D.le(min_dist) + dist_matrix_3D.ge(max_dist)).float()
    dm = torch.max(dist_matrix) * 1000
    dist_without_min_on_diag = dist_matrix + (dist_matrix_3D_mask * dm)

    if plot:
        #print(dist_matrix_3D)
        plt.figure(2)
        plt.clf()
        plt.subplot(1, 3, 1)
        plt.imshow(dist_matrix_3D.detach().cpu().numpy())
        plt.colorbar()
        plt.subplot(1, 3, 2)
        plt.imshow(dist_matrix_3D_mask.detach().cpu().numpy())
        plt.colorbar()
        plt.subplot(1, 3, 3)
        plt.imshow(dist_matrix.detach().cpu().numpy())
        plt.colorbar()
        plt.draw()
        plt.pause(0.001)

    keep = []
    # as hard as possible
    if dist_type == '2D':
        # we need to make sure that we select negative patches from
        # corresponding view
        neg_idx = torch.zeros(anchor.shape[0]).long().to(device)
        idx_map = torch.arange(0, dist_without_min_on_diag.shape[0]).to(device)
        for idx in range(0, dist_without_min_on_diag.shape[0]):
            an_id = id[idx]
            sel = id == an_id
            if torch.sum(sel) < min_in_batch:
                continue
            idx_map_sel = idx_map[sel]
            # hard mining
            min_neg_i, neg_idx_i = torch.min(dist_without_min_on_diag[idx][sel].reshape(1, -1), 1)
            if not hardest:
                # semi-hard mining
                pos_dist = pos1[idx] + margin
                #adaptive semi hard mining the greater hard_dist_coeff is, the harder negatives are selected
                hard_dist = (pos_dist - (min_neg_i + eps)) * hard_dist_coeff
                hard_idx = (dist_without_min_on_diag[idx, sel] < (pos_dist - hard_dist)).nonzero()
                if hard_idx.shape[0] > 0:
                    hard_idx = hard_idx.squeeze(1)
                    sel_neg = torch.randperm(hard_idx.shape[0])[0]
                    neg_idx_i = hard_idx[sel_neg]
                    min_neg_i = dist_without_min_on_diag[idx, neg_idx_i]
            neg_idx_i = idx_map_sel[neg_idx_i]
            neg_idx[idx] = neg_idx_i
            if as_hard_as_possible:
                # set selected negative column to max value, so that it cannot be
                # picked again. Otherwise this will be the same as the original
                # hardnet, just taking into account mixed pairs in a single batch
                dist_without_min_on_diag[:, neg_idx_i] = dm
            if min_neg_i < dm:
                keep.append(neg_idx_i)
    else:
        neg_idx = torch.zeros(anchor.shape[0]).long().to(device)
        for idx in range(0, dist_without_min_on_diag.shape[0]):
            min_neg_i, neg_idx_i = torch.min(dist_without_min_on_diag[idx].reshape(1, -1), 1)
            if not hardest:
                # semi-hard mining
                pos_dist = pos1[idx] + margin
                #adaptive semi hard mining the greater hard_dist_coeff is, the harder negatives are selected
                hard_dist = (pos_dist - (min_neg_i + eps)) * hard_dist_coeff
                hard_idx = (dist_without_min_on_diag[idx, :] < (pos_dist - hard_dist)).nonzero()
                if hard_idx.shape[0] > 0:
                    hard_idx = hard_idx.squeeze(1)
                    sel_neg = torch.randperm(hard_idx.shape[0])[0]
                    neg_idx_i = hard_idx[sel_neg]
                    min_neg_i = dist_without_min_on_diag[idx, neg_idx_i]
            neg_idx[idx] = neg_idx_i
            if as_hard_as_possible:
                # set selected negative column to max value, so that it cannot be
                # picked again. Otherwise this will be the same as the original
                # hardnet.
                dist_without_min_on_diag[:, neg_idx_i] = dm
            if min_neg_i < dm:
                keep.append(neg_idx_i)
    keep = torch.tensor(keep).long().to(device)

    return neg_idx, keep, dist_matrix_3D[torch.arange(0, dist_matrix_3D.shape[0]), neg_idx][keep]


def loss_HardNetWithDist(anchor, positive, c1, c2, anchor_swap = False, anchor_ave = False,\
        margin = 1.0, batch_reduce = 'min', loss_type = "triplet_margin", show=False, an_img=None, pos_img=None, filter_dist=True, device=torch.device('cpu'), min_dist=100.0, max_dist=300.0, hardest=False, hard_dist_coeff=0.0):
    """HardNet margin loss - calculates loss based on distance matrix based on positive distance and closest negative distance.
    """
    assert anchor.size() == positive.size(), "Input sizes between positive and negative must be equal."
    assert anchor.dim() == 2, "Inputd must be a 2D matrix."
    eps = 1e-8

    dist_matrix = distance_matrix_vector(anchor, positive) + eps
    eye = torch.autograd.Variable(torch.eye(dist_matrix.size(1))).to(device)

    # steps to filter out same patches that occur in distance matrix as negatives
    pos1 = torch.diag(dist_matrix)
    pos_dist_l = margin + pos1
    # hard_mask = torch.ones(dist_matrix.shape).bool()

    if batch_reduce == 'random':
        dist_without_min_on_diag = dist_matrix
    else:
        if not filter_dist:
            # original hardnet implementation
            dist_without_min_on_diag = dist_matrix+eye*10
            mask = (dist_without_min_on_diag.ge(0.008).float()-1.0)*(-1)
            mask = mask.type_as(dist_without_min_on_diag)*10
            dist_without_min_on_diag = dist_without_min_on_diag+mask
        else:
            # our implementation to filter out too close and too far negatives
            dist_matrix_3D = distance_matrix_vector(c1, c2)
            dist_matrix_3D_mask = (dist_matrix_3D.le(min_dist) + dist_matrix_3D.ge(max_dist)).float()
            #if not hardest:
            #    dist_matrix_3D_mask = 1.0 - dist_matrix_3D_mask
            #    dist_without_min_on_diag = dist_matrix * dist_matrix_3D_mask
            #else:
            dm = torch.max(dist_matrix) * 1000
            dist_without_min_on_diag = dist_matrix + (dist_matrix_3D_mask * dm) + eye * 1000

    if batch_reduce == 'min':
        min_neg, _ = torch.min(dist_without_min_on_diag, 1)
        # hard_mask = torch.zeros(dist_without_min_on_diag.shape)
        # hard_mask[torch.arange(0, dist_without_min_on_diag.shape[0]), _] = 1
        # hard_mask = hard_mask.bool()
        show_idx = _.clone().detach()
        if not hardest:
            ### using all triplets is not good for training, training
            ### is distracted by easy triplets
            #min_neg = dist_without_min_on_diag
            #pos_dist_l = pos_dist_l.reshape(-1, 1)
            #print("pos_dist", (pos_dist - min_neg))

            ### using random violating triplets is a way to go, but is slow
            ### because of the for loop
            # select random hard negative, not only the hardest
            for idx in range(dist_without_min_on_diag.shape[0]):
                #randomneg: any random patch which does not violate the positive threshold: dist_without_min_on_diag[idx, :] < dm
                #randomhardneg: any hard patch which does not violate the positive threshold: dist_without_min_on_diag[idx, :] < pos1[idx] + margin
                pos_dist = pos1[idx] + margin
                hard_dist = (pos_dist - (min_neg[idx] + eps)) * hard_dist_coeff  #adaptive semi hard mining the greater hard_dist_coeff is, the harder negatives are selected
                hard_idx = (dist_without_min_on_diag[idx, :] < (pos_dist - hard_dist)).nonzero()
                if hard_idx.shape[0] > 0:
                    hard_idx = hard_idx.squeeze(1)
                    sel = torch.randperm(hard_idx.shape[0])[0]
                    min_neg[idx] = dist_without_min_on_diag[idx, hard_idx[sel]]
                    show_idx[idx] = hard_idx[sel.detach()].detach() #update the used index so that the plots are correct

            # ### take all violating triplets without the need of the for loop
            # WARNING THIS VERSION DOES NOT TRAIN WELL, SEE 2019-11-13_09:16:47
            # # column vector
            # pos_dist_l = pos_dist_l.reshape(-1, 1)
            # # adaptive semi hard mining the greater hard_dist_coeff is,
            # # the harder negatives are selected
            # hard_dist = (pos_dist_l - (min_neg + eps)) * hard_dist_coeff
            # # create a mask to select only hard (violating negatives)
            # hard_mask = (dist_without_min_on_diag < (pos_dist_l - hard_dist))

        #print("pos shape", positive.shape, positive[_].shape)
        dist_pos_neg = torch.norm(positive - positive[_], dim=1)
        if show:
            min_neg_img = pos_img[show_idx]
            min_neg_sidxs = np.sort(show_idx.detach().cpu().numpy())
        if anchor_swap:
            min_neg2, _ = torch.min(dist_without_min_on_diag,0)
            min_neg = torch.min(min_neg,min_neg2)
        if show:

            img_neg = torchvision.utils.make_grid(min_neg_img[:, :3], nrow=15).detach().cpu().numpy()
            img_pos = torchvision.utils.make_grid(pos_img[:, :3], nrow=15).detach().cpu().numpy()
            img_an = torchvision.utils.make_grid(an_img[:, :3], nrow=15).detach().cpu().numpy()

            plt.clf()
            plt.subplot(1, 5, 1)
            plt.imshow(np.transpose(img_an, (1,2,0)), interpolation='nearest')
            plt.subplot(1, 5, 2)
            plt.imshow(np.transpose(img_pos, (1,2,0)), interpolation='nearest')
            plt.subplot(1, 5, 3)
            plt.imshow(np.transpose(img_neg, (1,2,0)), interpolation='nearest')
            plt.subplot(1, 5, 4)
            plt.imshow(dist_without_min_on_diag.detach().cpu().numpy())
            plt.colorbar()
            plt.subplot(1, 5, 5)
            plt.imshow(dist_matrix.detach().cpu().numpy())
            plt.colorbar()
            #plt.hist(min_neg_sidxs, bins=min_neg_sidxs.shape[0])
            #plt.show()
            plt.draw()
            plt.pause(0.001)

        pos = pos1
    elif batch_reduce == 'average':
        pos = pos1.repeat(anchor.size(0)).view(-1,1).squeeze(0)
        min_neg = dist_without_min_on_diag.view(-1,1)
        if anchor_swap:
            min_neg2 = torch.t(dist_without_min_on_diag).contiguous().view(-1,1)
            min_neg = torch.min(min_neg,min_neg2)
        min_neg = min_neg.squeeze(0)
    elif batch_reduce == 'random':
        idxs = torch.autograd.Variable(torch.randperm(anchor.size()[0]).long()).to(device)
        min_neg = dist_without_min_on_diag.gather(1,idxs.view(-1,1))
        if anchor_swap:
            min_neg2 = torch.t(dist_without_min_on_diag).gather(1,idxs.view(-1,1))
            min_neg = torch.min(min_neg,min_neg2)
        min_neg = torch.t(min_neg).squeeze(0)
        pos = pos1
    else:
        print ('Unknown batch reduce mode. Try min, average or random')
        sys.exit(1)
    if loss_type == "triplet_margin":
        #print("hard mask sum", torch.sum(hard_mask), "num elements", hard_mask.numel())
        # loss = torch.clamp(pos_dist_l - dist_without_min_on_diag, min=0.0)[hard_mask]
        #print("loss shapes", pos_dist_l.shape, min_neg.shape)
        loss = torch.clamp(pos_dist_l - min_neg, min=0.0)
    elif loss_type == 'softmax':
        exp_pos = torch.exp(2.0 - pos);
        exp_den = exp_pos + torch.exp(2.0 - min_neg) + eps;
        loss = - torch.log( exp_pos / exp_den )
    elif loss_type == 'contrastive':
        loss = torch.clamp(margin - min_neg, min=0.0) + pos;
    else:
        print ('Unknown loss type. Try triplet_margin, softmax or contrastive')
        sys.exit(1)
    #loss = torch.mean(loss)
    return loss


def loss_HardNet(anchor, positive, anchor_swap = False, anchor_ave = False,\
        margin = 1.0, batch_reduce = 'min', loss_type = "triplet_margin", show=False, an_img=None, pos_img=None):
    """HardNet margin loss - calculates loss based on distance matrix based on positive distance and closest negative distance.
    """

    assert anchor.size() == positive.size(), "Input sizes between positive and negative must be equal."
    assert anchor.dim() == 2, "Inputd must be a 2D matrix."
    eps = 1e-8
    dist_matrix = distance_matrix_vector(anchor, positive) + eps
    eye = torch.autograd.Variable(torch.eye(dist_matrix.size(1))).cuda()

    # steps to filter out same patches that occur in distance matrix as negatives
    pos1 = torch.diag(dist_matrix)
    dist_without_min_on_diag = dist_matrix+eye*10
    mask = (dist_without_min_on_diag.ge(0.008).float()-1.0)*(-1)
    mask = mask.type_as(dist_without_min_on_diag)*10
    dist_without_min_on_diag = dist_without_min_on_diag+mask
    if batch_reduce == 'min':
        min_neg, _ = torch.min(dist_without_min_on_diag,1)
        if show:
            min_neg_img = pos_img[_]
            min_neg_sidxs = np.sort(_.detach().cpu().numpy())
        if anchor_swap:
            min_neg2, _ = torch.min(dist_without_min_on_diag,0)
            min_neg = torch.min(min_neg,min_neg2)
        if show:
            print(dist_without_min_on_diag)
            img_neg = torchvision.utils.make_grid(min_neg_img, nrow=15).detach().cpu().numpy()
            img_pos = torchvision.utils.make_grid(pos_img, nrow=15).detach().cpu().numpy()
            img_an = torchvision.utils.make_grid(an_img, nrow=15).detach().cpu().numpy()

            plt.clf()
            plt.subplot(1, 5, 1)
            plt.imshow(np.transpose(img_an, (1,2,0)), interpolation='nearest')
            plt.subplot(1, 5, 2)
            plt.imshow(np.transpose(img_pos, (1,2,0)), interpolation='nearest')
            plt.subplot(1, 5, 3)
            plt.imshow(np.transpose(img_neg, (1,2,0)), interpolation='nearest')
            plt.subplot(1, 5, 4)
            plt.imshow(dist_matrix.detach().cpu().numpy())
            plt.subplot(1, 5, 5)
            plt.hist(min_neg_sidxs, bins=min_neg_sidxs.shape[0])
            #plt.show()
            plt.draw()
            plt.pause(0.001)

        pos = pos1
    elif batch_reduce == 'average':
        pos = pos1.repeat(anchor.size(0)).view(-1,1).squeeze(0)
        min_neg = dist_without_min_on_diag.view(-1,1)
        if anchor_swap:
            min_neg2 = torch.t(dist_without_min_on_diag).contiguous().view(-1,1)
            min_neg = torch.min(min_neg,min_neg2)
        min_neg = min_neg.squeeze(0)
    elif batch_reduce == 'random':
        idxs = torch.autograd.Variable(torch.randperm(anchor.size()[0]).long()).cuda()
        min_neg = dist_without_min_on_diag.gather(1,idxs.view(-1,1))
        if anchor_swap:
            min_neg2 = torch.t(dist_without_min_on_diag).gather(1,idxs.view(-1,1))
            min_neg = torch.min(min_neg,min_neg2)
        min_neg = torch.t(min_neg).squeeze(0)
        pos = pos1
    else:
        print ('Unknown batch reduce mode. Try min, average or random')
        sys.exit(1)
    if loss_type == "triplet_margin":
        loss = torch.clamp(margin + pos - min_neg, min=0.0)
    elif loss_type == 'softmax':
        exp_pos = torch.exp(2.0 - pos);
        exp_den = exp_pos + torch.exp(2.0 - min_neg) + eps;
        loss = - torch.log( exp_pos / exp_den )
    elif loss_type == 'contrastive':
        loss = torch.clamp(margin - min_neg, min=0.0) + pos;
    else:
        print ('Unknown loss type. Try triplet_margin, softmax or contrastive')
        sys.exit(1)
    loss = torch.mean(loss)
    return loss


def loss_HardNetNegShow(anchor, positive, neg, anchor_swap = False, anchor_ave = False,\
        margin = 1.0, batch_reduce = 'min', loss_type = "triplet_margin", show=False, an_img=None, pos_img=None, neg_img=None):
    """HardNet margin loss - calculates loss based on distance matrix based on positive distance and closest negative distance.
    """
    assert anchor.size() == positive.size(), "Input sizes between positive and negative must be equal."
    assert anchor.dim() == 2, "Inputd must be a 2D matrix."
    eps = 1e-8
    dist_matrix = distance_matrix_vector(anchor, positive) + eps
    eye = torch.autograd.Variable(torch.eye(dist_matrix.size(1))).cuda()

    # steps to filter out same patches that occur in distance matrix as negatives
    pos1 = torch.diag(dist_matrix)
    dist_without_min_on_diag = distance_matrix_vector_negative(anchor, neg) + eps
    #max_mat = torch.max(dist_matrix) + 1.0
    #dist_without_min_on_diag = dist_matrix + eye * max_mat
    #dist_without_min_on_diag = dist_matrix+eye*10
    #mask = (dist_without_min_on_diag.ge(0.008).float()-1.0)*(-1)
    #mask = mask.type_as(dist_without_min_on_diag)*10
    #dist_without_min_on_diag = dist_without_min_on_diag+mask
    if batch_reduce == 'min':
        min_neg, _ = torch.min(dist_without_min_on_diag, 1)
        numneg = neg.shape[0] // anchor.shape[0]
        if show:
            min_neg_img = []
            for i in range(_.shape[0]):
                #for element in batch
                neg_idx = i * numneg + _[i]
                min_neg_img.append(neg_img[neg_idx][None, :])
            min_neg_img = torch.cat(min_neg_img)

            #print("min neg idx: ", (torch.arange(0, min_neg.shape[0]) == _.detach().cpu()))
        min_neg_sidxs = np.sort(_.detach().cpu().numpy())
        if anchor_swap:
            min_neg2, _ = torch.min(dist_without_min_on_diag, 0)
            if show:
                min_neg_img = []
                for i in range(0, min_neg.shape[0]):
                    if min_neg[i] < min_neg2[i]:
                        print("pos shape, ", pos_img[i, :].shape, i)
                        min_neg_img.append(pos_img[i, :][None, :])
                    else:
                        print("an shape, ", an_img[i, :].shape, i)
                        min_neg_img.append(an_img[i, :][None, :])
                #print("min neg img len", len(min_neg_img), "shape", min_neg_img[0].shape, pos_img.shape, an_img.shape, min_neg.shape[0])
                min_neg_img = torch.cat(min_neg_img)
            min_neg = torch.min(min_neg, min_neg2)
        if show:
            print(dist_without_min_on_diag)
            img_neg = torchvision.utils.make_grid(min_neg_img, nrow=15).detach().cpu().numpy()
            img_pos = torchvision.utils.make_grid(pos_img, nrow=15).detach().cpu().numpy()
            img_an = torchvision.utils.make_grid(an_img, nrow=15).detach().cpu().numpy()

            plt.clf()
            plt.subplot(1, 5, 1)
            plt.imshow(np.transpose(img_an, (1,2,0)), interpolation='nearest')
            plt.subplot(1, 5, 2)
            plt.imshow(np.transpose(img_pos, (1,2,0)), interpolation='nearest')
            plt.subplot(1, 5, 3)
            plt.imshow(np.transpose(img_neg, (1,2,0)), interpolation='nearest')
            plt.subplot(1, 5, 4)
            plt.imshow(dist_without_min_on_diag.detach().cpu().numpy())
            plt.subplot(1, 5, 5)
            plt.hist(min_neg_sidxs, bins=min_neg_sidxs.shape[0])
            #plt.show()
            plt.draw()
            plt.pause(0.001)

        pos = pos1
    elif batch_reduce == 'average':
        pos = pos1.repeat(anchor.size(0)).view(-1,1).squeeze(0)
        min_neg = dist_without_min_on_diag.view(-1,1)
        if anchor_swap:
            min_neg2 = torch.t(dist_without_min_on_diag).contiguous().view(-1,1)
            min_neg = torch.min(min_neg,min_neg2)
        min_neg = min_neg.squeeze(0)
    elif batch_reduce == 'random':
        idxs = torch.autograd.Variable(torch.randperm(anchor.size()[0]).long()).cuda()
        #min_neg = dist_without_min_on_diag.gather(1,idxs.view(-1,1))
        min_neg = dist_without_min_on_diag[:, 0]
        if anchor_swap:
            min_neg2 = torch.t(dist_without_min_on_diag).gather(1,idxs.view(-1,1))
            min_neg = torch.min(min_neg,min_neg2)
        #min_neg = torch.t(min_neg).squeeze(0)
        pos = pos1
    else:
        print ('Unknown batch reduce mode. Try min, average or random')
        sys.exit(1)
    if loss_type == "triplet_margin":
        if show:
            print("pos ", pos[:10])
            print("min_neg ", min_neg[:10])
        loss = torch.clamp(margin + pos - min_neg, min=0.0)
    elif loss_type == 'softmax':
        exp_pos = torch.exp(2.0 - pos);
        exp_den = exp_pos + torch.exp(2.0 - min_neg) + eps;
        loss = - torch.log( exp_pos / exp_den )
    elif loss_type == 'contrastive':
        loss = torch.clamp(margin - min_neg, min=0.0) + pos;
    else:
        print ('Unknown loss type. Try triplet_margin, softmax or contrastive')
        sys.exit(1)
    loss = torch.mean(loss)
    return loss

def global_orthogonal_regularization(anchor, negative):

    neg_dis = torch.sum(torch.mul(anchor,negative),1)
    dim = anchor.size(1)
    gor = torch.pow(torch.mean(neg_dis),2) + torch.clamp(torch.mean(torch.pow(neg_dis,2))-1.0/dim, min=0.0)

    return gor

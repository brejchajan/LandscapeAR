# @Date:   2020-08-06T16:45:26+02:00
# @Email:  ibrejcha@fit.vutbr.cz; brejchaja@gmail.com
# @Project: LandscapeAR
# @Last modified time: 2020-08-11T17:51:53+02:00
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

import sys
import argparse as ap
import torch
import torch.nn.init
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import time
import os
import sys
import cv2
import math
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import random
import time
import numpy as np
import glob
import os
import inspect

# our code
import training.Architectures as Architectures
from trainPatchesDescriptors import findLatestModelPath

# D2net
from thirdparty.d2net.lib.model_test import D2Net

# all types of patches
tps = ['ref','e1','e2','e3','e4','e5','h1','h2','h3','h4','h5',\
       't1','t2','t3','t4','t5']

class hpatches_sequence:
    """Class for loading an HPatches sequence from a sequence folder"""
    itr = tps
    def __init__(self,base):
        name = base.split('/')
        self.name = name[-1]
        self.base = base
        for t in self.itr:
            im_path = os.path.join(base, t+'.png')
            im = cv2.imread(im_path,0)
            self.N = int(im.shape[0]/65)
            setattr(self, t, np.split(im, self.N))

class L2Norm(nn.Module):
    def __init__(self):
        super(L2Norm,self).__init__()
        self.eps = 1e-10
    def forward(self, x):
        norm = torch.sqrt(torch.sum(x * x, dim = 1) + self.eps)
        x= x / norm.unsqueeze(-1).expand_as(x)
        return x

class L1Norm(nn.Module):
    def __init__(self):
        super(L1Norm,self).__init__()
        self.eps = 1e-10
    def forward(self, x):
        norm = torch.sum(torch.abs(x), dim = 1) + self.eps
        x= x / norm.expand_as(x)
        return x

def buildArgumentParser():
    parser = ap.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("output_dir", default="trained_models")
    parser.add_argument("snapshot", help="""Restores model from
                        the snapshot with given name. If no model is specified
                        using -m, loads latest model.""")
    parser.add_argument("-l", "--log_dir", help="""Directory used for saving
                         training progress logs and models. Defaults to cwd
                         when not specified.""", default=os.getcwd())
    parser.add_argument("-m", "--model_name", help="""Specify exact model name
                        to be restored using -r option.""")
    parser.add_argument("-c", "--cuda", action="store_true", help="If this \
                        flag is used, cuda will be used for neural network\
                        processing.")
    parser.add_argument("--branch", help="Which branch to use for descriptor \
                        extraction. [photo|render] Default=photo.",
                        default="photo")
    parser.add_argument("--d2net", action='store_true', help=" \
                        User original D2Net keypoints and descriptors for \
                        matching.")
    return parser


class HPatchesExtractor(object):
    def __init__(self, args):
        super(HPatchesExtractor, self).__init__()
        self.normalize_output = True
        self.cuda = args.cuda
        self.device = torch.device("cuda:0" if args.cuda else "cpu")

        self.use_d2net = args.d2net
        self.snapshot = args.snapshot
        self.branch = args.branch
        self.fcn_keypoints = False
        self.needles = False
        if not args.model_name:
            log_dir = os.path.join(args.log_dir, args.snapshot)
            models_path = os.path.join(log_dir, "models")
            model_path = findLatestModelPath(models_path)
            args.model_name = os.path.basename(model_path)
        self.full_model_name = args.model_name
        try:
            self.input_dir = args.input_dir
            self.output_dir = args.output_dir
            seqs = glob.glob(self.input_dir+'/*')
            self.seqs = [os.path.abspath(p) for p in seqs]
            if len(self.seqs) <= 0:
                print("Unable to find input HPatches sequences. Aborting.")
                sys.exit(1)
        except:
            print('Wrong input format. Try python hpatches_extract_HardNet.py /home/ubuntu/dev/hpatches/hpatches-benchmark/data/hpatches-release /home/old-ufo/dev/hpatches/hpatches-benchmark/data/descriptors')
            sys.exit(1)

        self.w = 65

        if self.use_d2net:
            self.d2net_name = "d2_tf"
            self.d2net = D2Net(
                model_file='pretrained/d2_net/' + self.d2net_name + '.pth',
                use_relu=True,
                use_cuda=args.cuda
            )
            # so that we get single descriptor
            self.net_w = 65
        else:
            self.net_w = 64

        self.loadModel(args)

    def loadModel(self, args):
        log_dir = os.path.join(args.log_dir, args.snapshot)

        models_path = os.path.join(log_dir, "models")
        if args.model_name:
            model_path = os.path.join(models_path, args.model_name)
            model_name = os.path.basename(model_path).split("_epoch_")[0]
        else:
            model_path = findLatestModelPath(models_path)
            model_name = os.path.basename(model_path).split("_epoch_")[0]
        self.model_name = model_name

        # Our detector
        module = __import__("training").Architectures
        net_class = getattr(module, model_name)
        self.net_class = net_class

        if self.fcn_keypoints:
            if  net_class.__name__ != "MultimodalKeypointPatchNet5lShared2lFCN":
                self.net = Architectures.MultimodalPatchNet5lShared2lFCN(self.normalize_output)
                self.net_keypoints = Architectures.MultimodalPatchNet5lShared2lFCN(self.normalize_output,
                strides=[1], strides_branch=[2, 1, 1, 1],
                dilations=[2], dilations_branch=[1, 1, 2, 2])
            else:
                self.net = Architectures.MultimodalKeypointPatchNet5lShared2lFCN(self.normalize_output, pretrained=False)
                self.net_keypoints = Architectures.MultimodalKeypointPatchNet5lShared2lFCN(
                    self.normalize_output,
                    strides=[1], strides_branch=[2, 1, 1, 1],
                    dilations=[2], dilations_branch=[1, 1, 2, 2],
                    pretrained=False
                )
        else:
            cls_args = inspect.getfullargspec(net_class.__init__).args
            if 'needles' not in cls_args:
                self.needles = 0
                self.net = net_class(self.normalize_output)
            else:
                self.net = net_class(
                    self.normalize_output, needles=self.needles
                )

        # load the actual model
        checkpoint = torch.load(model_path, map_location=self.device)
        if self.fcn_keypoints:
            x = checkpoint['model_state_dict']['conv2.weight']
            checkpoint['model_state_dict']['conv2.weight'] = x.reshape(x.shape[0], x.shape[1], 1, 1)
            self.net_keypoints.load_state_dict(checkpoint['model_state_dict'])
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.net.to(self.device)
        self.net.eval()

    def extract(self):
        branch = ""
        if self.branch != "photo":
            branch = "_render"
        if self.use_d2net:
            curr_desc_name = self.d2net_name + "_center"
        else:
            curr_desc_name = self.snapshot + "_" + self.full_model_name + branch
        for seq_path in self.seqs:
            seq = hpatches_sequence(seq_path)
            path = os.path.join(self.output_dir, os.path.join(curr_desc_name, seq.name))
            if not os.path.exists(path):
                os.makedirs(path)
            descr = np.zeros((seq.N, 128)) # trivial (mi,sigma) descriptor
            for tp in tps:
                print(seq.name+'/'+tp)
                if os.path.isfile(os.path.join(path,tp+'.csv')):
                    continue
                n_patches = 0
                for i,patch in enumerate(getattr(seq, tp)):
                    n_patches+=1
                t = time.time()
                patches_for_net = np.zeros((n_patches, 1, self.net_w, self.net_w))
                uuu = 0
                for i,patch in enumerate(getattr(seq, tp)):
                    patches_for_net[i, 0, :, :] = cv2.resize(patch[0:self.w, 0:self.w], (self.net_w, self.net_w))
                patches_for_net = np.concatenate([patches_for_net, patches_for_net, patches_for_net], axis=1)
                ###
                self.net.eval()
                outs = []
                bs = 128
                n_batches = int(n_patches / bs) + 1
                for batch_idx in range(n_batches):
                    st = batch_idx * bs
                    if batch_idx == n_batches - 1:
                        if (batch_idx + 1) * bs > n_patches:
                            end = n_patches
                        else:
                            end = (batch_idx + 1) * bs
                    else:
                        end = (batch_idx + 1) * bs
                    if st >= end:
                        continue
                    data_a = patches_for_net[st: end, :, :, :].astype(np.float32)
                    data_a = torch.from_numpy(data_a)
                    if self.cuda:
                        data_a = data_a.cuda()
                    data_a = Variable(data_a, volatile=True)
                    # compute output
                    if (self.use_d2net):
                        out_a = self.d2net.dense_feature_extraction(data_a)[:, :, 7, 7].reshape(-1, 512)
                    else:
                        if self.branch == 'photo':
                            out_a = self.net.forward_photo(data_a).reshape(-1, 128)
                        else:
                            out_a = self.net.forward_render(data_a).reshape(-1, 128)
                    outs.append(out_a.data.cpu().numpy())
                res_desc = np.concatenate(outs)
                print(res_desc.shape, n_patches)
                res_desc = np.reshape(res_desc, (n_patches, -1))
                out = np.reshape(res_desc, (n_patches,-1))
                np.savetxt(os.path.join(path,tp+'.csv'), out, delimiter=',', fmt='%10.5f')   # X is an array

if __name__ == "__main__":
    parser = buildArgumentParser()
    args = parser.parse_args()
    extractor = HPatchesExtractor(args)
    extractor.extract()

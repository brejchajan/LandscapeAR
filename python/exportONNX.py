# @Author: Jan Brejcha <ibrejcha>
# @Date:   2020-08-11T17:51:13+02:00
# @Email:  ibrejcha@fit.vutbr.cz; brejchaja@gmail.com
# @Project: LandscapeAR
# @Last modified by:   ibrejcha
# @Last modified time: 2020-08-12T11:46:19+02:00
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
import torchvision
import argparse as ap
import os
import inspect
import numpy as np

import onnx
import onnxruntime

import training.Architectures
from export.ModelExporter import ModelExporter
from trainPatchesDescriptors import findLatestModelPath


def buildArgumentParser():
    parser = ap.ArgumentParser()
    parser.add_argument("snapshot", help="""Restores model from
                        the snapshot with given name. If no model is specified
                        using -m, loads latest model.""")
    parser.add_argument("-l", "--log_dir", help="""Directory used for saving
                         training progress logs and models. Defaults to cwd
                         when not specified.""", default=os.getcwd())
    parser.add_argument("-m", "--model_name", help="""Specify exact model name
                        to be restored using -r option.""")
    # FCN Keypoints not supported yet
    #parser.add_argument("--fcn-keypoints", action='store_true', help="Use \
    #                    fully convolutional variant of our keypoint \
    #                    two-branch net. This allows dense extraction of \
    #                    keypoints from the whole image at one step.")
    parser.add_argument("--needles", default=0, type=int, help="If number \
                        greater than zero is used, then instead of a single \
                        patch a whole needle of patches will be extracted. Our\
                        network then takes several patches in a form of a \
                        needle encoded to channels of the input. This \
                        approach is described here: Lotan and Irani: \
                        Needle-Match: Reliable Patch Matching under \
                        High Uncertainty, CVPR 2016")
    parser.add_argument("-c", "--cuda", action="store_true", help="If this \
                        flag is used, cuda will be used for neural network\
                        processing.")
    parser.add_argument("--output_dir", default="trained_models")

    return parser


if __name__ == "__main__":
    parser = buildArgumentParser()
    args = parser.parse_args()

    exporter = ModelExporter(args)
    exporter.exportONNX(args.output_dir)

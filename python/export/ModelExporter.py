# @Author: Jan Brejcha <ibrejcha>
# @Date:   2020-08-12T11:40:28+02:00
# @Email:  ibrejcha@fit.vutbr.cz; brejchaja@gmail.com
# @Project: LandscapeAR
# @Last modified by:   ibrejcha
# @Last modified time: 2020-08-12T11:40:50+02:00
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
import os
import inspect
import numpy as np

import onnx
import onnxruntime

from trainPatchesDescriptors import findLatestModelPath


class ModelExporter(object):
    def __init__(self, args):
        super(ModelExporter, self).__init__()

        self.normalize_output = True
        self.cuda = False
        self.fcn_keypoints = False  # args.fcn_keypoints
        self.needles = args.needles
        self.device = torch.device("cuda:0" if args.cuda else "cpu")

        self.loadModel(args)

    def exportONNX(self, output_dir):
        dummy_input_1 = torch.randn(1, 3, 64, 64)
        input_names = ["input"]
        output_names = ["output"]

        # use custom forward function which takes just one argument
        # export separate model for photo and separate model for render

        output_name = self.net_photo.__class__.__name__.replace("PhotoONNX", "") + "_" + self.model_name + "_photo.onnx"
        output_path = os.path.join(output_dir, output_name)
        torch.onnx.export(self.net_photo, dummy_input_1, output_path, verbose=True, input_names=input_names, output_names=output_names)

        self.checkONNXModel(output_path, dummy_input_1, self.net_photo)

        output_name = self.net_render.__class__.__name__.replace("RenderONNX", "") + "_" + self.model_name + "_render.onnx"
        output_path = os.path.join(output_dir, output_name)
        torch.onnx.export(self.net_render, dummy_input_1, output_path, verbose=True, input_names=input_names, output_names=output_names)

        self.checkONNXModel(output_path, dummy_input_1, self.net_render)

    def checkONNXModel(self, output_path, dummy_input_1, net):
        # model checking
        print("Checking model", output_path)
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)

        ort_session = onnxruntime.InferenceSession(output_path)
        def to_numpy(tensor):
            return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

        # compute ONNX Runtime output prediction
        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input_1)}
        ort_outs = ort_session.run(None, ort_inputs)

        # compare ONNX Runtime and PyTorch results
        torch_out = net(dummy_input_1)
        np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
        print("Exported model has been tested with ONNXRuntime, and the result looks good!")


    def loadModel(self, args):
        log_dir = os.path.join(args.log_dir, "runs", args.snapshot)

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
        try:
            net_class_photo = getattr(module, model_name + "PhotoONNX")
        except AttributeError as ae:
            print(
                """
                Unable to find a wrapper class for ONNX export.
                Please, provide a wrapper class encapsulating a photo branch of
                a two branch network with PhotoONNX suffix in the class name.
                """
            )
            exit()

        try:
            net_class_render = getattr(module, model_name + "RenderONNX")
        except AttributeError as ae:
            print(
                """
                Unable to find a wrapper class for ONNX export.
                Please, provide a wrapper class encapsulating a render branch of
                a two branch network with RenderONNX suffix in the class name.
                """
            )
            exit()

        self.net_class_photo = net_class_photo
        self.net_class_render = net_class_render

        # FCN Keypoints not supported yet.
        # if self.fcn_keypoints:
        #     if  net_class.__name__ != "MultimodalKeypointPatchNet5lShared2lFCN":
        #         self.net = training.Architectures.MultimodalPatchNet5lShared2lFCN(self.normalize_output)
        #         self.net_keypoints = training.Architectures.MultimodalPatchNet5lShared2lFCN(self.normalize_output,
        #         strides=[1], strides_branch=[2, 1, 1, 1],
        #         dilations=[2], dilations_branch=[1, 1, 2, 2])
        #     else:
        #         self.net = training.Architectures.MultimodalKeypointPatchNet5lShared2lFCN(self.normalize_output, pretrained=False)
        #         self.net_keypoints = training.Architectures.MultimodalKeypointPatchNet5lShared2lFCN(
        #             self.normalize_output,
        #             strides=[1], strides_branch=[2, 1, 1, 1],
        #             dilations=[2], dilations_branch=[1, 1, 2, 2],
        #             pretrained=False
        #         )
        # else:
        cls_args = inspect.getfullargspec(net_class.__init__).args
        if 'needles' not in cls_args:
            self.needles = 0
            self.net_photo = net_class_photo(self.normalize_output)
            self.net_render = net_class_render(self.normalize_output)
        else:
            self.net_photo = net_class_photo(
                self.normalize_output, needles=self.needles
            )
            self.net_render = net_class_render(
                self.normalize_output, needles=self.needles
            )

        # load the actual model
        checkpoint = torch.load(model_path, map_location=self.device)
        if self.fcn_keypoints:
            x = checkpoint['model_state_dict']['conv2.weight']
            checkpoint['model_state_dict']['conv2.weight'] = x.reshape(x.shape[0], x.shape[1], 1, 1)
            self.net_keypoints.load_state_dict(checkpoint['model_state_dict'])
        self.net_photo.load_state_dict(checkpoint['model_state_dict'])
        self.net_photo.eval()
        self.net_render.load_state_dict(checkpoint['model_state_dict'])
        self.net_render.eval()

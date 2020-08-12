# @Date:   2020-08-06T16:28:44+02:00
# @Email:  ibrejcha@fit.vutbr.cz; brejchaja@gmail.com
# @Project: LandscapeAR
# @Last modified time: 2020-08-11T17:49:59+02:00
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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models.resnet import BasicBlock, conv1x1
from torchvision.models.vgg import vgg16
import os

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.orthogonal(m.weight.data, gain=0.6)
        try:
            nn.init.constant(m.bias.data, 0.01)
        except:
            pass
    return

class L2Norm(nn.Module):
    def __init__(self):
        super(L2Norm,self).__init__()
        self.eps = 1e-10
    def forward(self, x):
        norm = torch.sqrt(torch.sum(x * x, dim = 1) + self.eps)
        x= x / norm.unsqueeze(-1).expand_as(x)
        return x

class HardNet(nn.Module):
    """HardNet model definition
    """
    def __init__(self, affine_bn=False):
        super(HardNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(32, affine=affine_bn),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(32, affine=affine_bn),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias = False),
            nn.BatchNorm2d(64, affine=affine_bn),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(64, affine=affine_bn),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias = False),
            nn.BatchNorm2d(128, affine=affine_bn),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(128, affine=affine_bn),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(128, 128, kernel_size=8, bias = False),
            nn.BatchNorm2d(128, affine=affine_bn),
        )
        self.features.apply(weights_init)
        return

    def input_norm(self,x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)

    def forward(self, input):
        x_features = self.features(self.input_norm(input))
        return x_features
        #x = x_features.view(x_features.size(0), -1)
        #return L2Norm()(x)


class OriginalHardNet(nn.Module):
    def __init__(self, normalize_output=False):
        super(OriginalHardNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias = False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2,padding=1, bias = False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(128, 128, kernel_size=8, bias = False),
            nn.BatchNorm2d(128, affine=False),
        )
        self.features.apply(weights_init)

        print("not loading weights")
        # load pretrained hardnet weights
        #model_path ="/mnt/matylda1/ibrejcha/devel/adobetrips/python/pretrained/hardnet/HardNet++.pth"
        #checkpoint = torch.load(model_path)
        #self.load_state_dict(checkpoint['state_dict'])

        return

    def input_norm(self, x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)

    def forward_shared(self, input):
        x_features = self.features(self.input_norm(input))
        x = x_features.view(x_features.size(0), -1)
        return L2Norm()(x)

    def forward_photo(self, x):
        return self.forward_shared(x)

    def forward_render(self, x):
        return self.forward_shared(x)

    def forward(self, anchor, pos, neg):
        anchor = self.forward_shared(anchor)
        pos = self.forward_shared(pos)
        neg = self.forward_shared(neg)
        return anchor, pos, neg


class SimplePatchNet(nn.Module):

    def __init__(self, needles=0):
        super(SimplePatchNet, self).__init__()
        self.conv1 = nn.Conv2d(3 * (needles + 1), 32, 3)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.conv3 = nn.Conv2d(32, 32, 3)

    def input_norm(self, x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)

    def forward(self, x):
        # x = self.input_norm(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        return x


class SimplePatchNet5l8CH(nn.Module):

    def __init__(self):
        super(SimplePatchNet5l8CH, self).__init__()
        self.conv1 = nn.Conv2d(8, 16, 3, stride=2)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=2)
        self.conv4 = nn.Conv2d(64, 128, 3, stride=2)

    def forward(self, x):
        #no input norm since this is for processing rendered data
        x = x - 0.5  # so that we have data centered around zero
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        return x


class SimplePatchNet5l(nn.Module):

    def __init__(self, strides=[2, 2, 2, 2],
                 dilations=[1, 1, 1, 1], needles=0):
        super(SimplePatchNet5l, self).__init__()
        self.conv1 = nn.Conv2d(3 * (needles + 1), 16, 3, stride=strides[0], dilation=dilations[0])
        self.conv2 = nn.Conv2d(16, 32, 3, stride=strides[1], dilation=dilations[1])
        self.conv3 = nn.Conv2d(32, 64, 3, stride=strides[2], dilation=dilations[2])
        self.conv4 = nn.Conv2d(64, 128, 3, stride=strides[3], dilation=dilations[3])


    def input_norm(self, x):
        ## alternatively can be written as
        #mp = torch.mean(x, dim=[1,2,3])
        #sp = torch.std(x, dim=[1,2,3]) + 1e-7
        #return x - mp.detach().reshape(-1, 1, 1, 1).expand_as(x) / sp.detach().reshape(-1, 1, 1, 1).expand_as(x)

        flat = x.contiguous().view(x.size(0), -1)  # for export to ONNX remove change x.size(0) to 1
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)

    def forward(self, x):
        x = self.input_norm(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        return x


class PatchNetBN5l(nn.Module):

    def __init__(self):
        super(PatchNetBN5l, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=2)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, 3, stride=2)
        self.bn4 = nn.BatchNorm2d(128)


    def input_norm(self, x):
        flat = x.contiguous().view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)

    def forward(self, x):
        x = self.input_norm(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        return x


class MultimodalPatchNet5lShared2l8CH(nn.Module):
    def __init__(self, normalize_output=False):
        super(MultimodalPatchNet5lShared2l8CH, self).__init__()

        self.normalize_output = normalize_output
        self.branch1 = SimplePatchNet5l()
        self.branch2 = SimplePatchNet5l8CH()

        self.conv1 = nn.Conv2d(128, 128, 3, stride=2)
        self.conv2 = nn.Linear(128, 128)

        self.eps = 1e-10

    def forward_shared(self, x):
        x = F.relu(self.conv1(x))
        x = x.reshape(x.size(0), -1)
        x = self.conv2(x)

        if self.normalize_output:
            x = x / (torch.norm(x, dim=1).reshape(-1, 1) + self.eps)

        return x

    def forward_photo(self, x):
        return self.forward_shared(self.branch1(x))

    def forward_render(self, x):
        return self.forward_shared(self.branch2(x))

    def forward(self, anchor, pos, neg):
        anchor = self.forward_shared(self.branch1(anchor))
        pos = self.forward_shared(self.branch2(pos))
        neg = self.forward_shared(self.branch2(neg))

        return anchor, pos, neg


class SimpleResNet(nn.Module):
    def __init__(self, normalize_output=False):
        super(SimpleResNet, self).__init__()

        self.normalize_output = normalize_output
        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(2048, 128)

    def forward_photo(self, x):
        return self.model(x)

    def forward_render(self, x):
        return self.model(x)

    def forward(self, anchor, pos, neg):
        anchor = self.model(anchor)
        pos = self.model(pos)
        neg = self.model(neg)

        if self.normalize_output:
            anchor = L2Norm()(anchor)
            pos = L2Norm()(pos)
            neg = L2Norm()(neg)
        return anchor, pos, neg


class MultimodalResNet50(nn.Module):
    def __init__(self, normalize_output=False):
        super(MultimodalResNet50, self).__init__()

        self.normalize_output = normalize_output
        self.branch1 = models.resnet50(pretrained=True)
        self.branch1.fc = nn.Linear(2048, 128)

        self.branch2 = models.resnet50(pretrained=True)
        self.branch2.fc = nn.Linear(2048, 128)

    def input_norm(self, x):
        flat = x.contiguous().view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)

    def forward_shared(self, x):
        if self.normalize_output:
            x = L2Norm()(x)
        return x

    def forward_photo(self, x):
        x = self.input_norm(x)
        return self.forward_shared(self.branch1(x))

    def forward_render(self, x):
        x = self.input_norm(x)
        return self.forward_shared(self.branch2(x))

    def forward(self, anchor, pos, neg):
        anchor = self.forward_photo(anchor)
        pos = self.forward_render(pos)
        neg = self.forward_render(neg)

        return anchor, pos, neg


class SinglemodalHardNet3l(nn.Module):
    def __init__(self, normalize_output=False, affine_bn=True):
        super(SinglemodalHardNet3l, self).__init__()

        self.normalize_output = normalize_output
        self.branch1 = HardNet(affine_bn=False)

        # load pretrained hardnet weights
        model_path ="/mnt/matylda1/ibrejcha/devel/adobetrips/python/pretrained/hardnet/HardNet++_3ch.pth"
        checkpoint = torch.load(model_path)
        self.branch1.load_state_dict(checkpoint['state_dict'])

        self.conv1 = nn.Conv2d(128, 128, 3, stride=2, bias=False)
        self.bn1 = nn.BatchNorm2d(128, affine=affine_bn)
        self.conv2 = nn.Conv2d(128, 128, 3, stride=2, bias=False)
        self.bn2 = nn.BatchNorm2d(128, affine=affine_bn)
        self.conv3 = nn.Linear(128, 128)

        self.eps = 1e-10

    def forward_shared(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.reshape(x.size(0), -1)
        x = self.conv3(x)

        if self.normalize_output:
            return L2Norm()(x)

        return x

    def forward_photo(self, x):
        return self.forward_shared(self.branch1(x))

    def forward_render(self, x):
        return self.forward_shared(self.branch1(x))

    def forward(self, anchor, pos, neg):
        anchor = self.forward_shared(self.branch1(anchor))
        pos = self.forward_shared(self.branch1(pos))
        neg = self.forward_shared(self.branch1(neg))

        return anchor, pos, neg

class LockedSinglemodalVGG16FinetunedShared3l(nn.Module):
    def __init__(self, normalize_output=False):
        super(LockedSinglemodalVGG16FinetunedShared3l, self).__init__()

        self.normalize_output = normalize_output
        self.branch1 = vgg16(pretrained=True)
        vgg16_layers = [
            'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2',
            'pool1',
            'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2',
            'pool2',
            'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3',
            'pool3',
            'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3',
            'pool4',
            'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3',
            'pool5'
        ]
        conv4_3_idx = vgg16_layers.index('conv4_3')
        self.branch1 = nn.Sequential(
            *list(self.branch1.features.children())[: conv4_3_idx + 1]
        )

        model_file = "/mnt/matylda1/ibrejcha/devel/adobetrips/python/d2_net/models/d2_tf.pth"
        checkpoint = torch.load(model_file)
        d2net_dict = {}
        for key in checkpoint['model'].keys():
            if 'dense_feature_extraction' in key:
                nk = key.replace('dense_feature_extraction.model.', '')
                d2net_dict.update({nk: checkpoint['model'][key]})

        self.branch1.load_state_dict(d2net_dict)
        print("Loaded model", model_file)

        # lock weights in both branches to finetune only the shared part
        for param in self.branch1.parameters():
            param.requires_grad = False

        # unlock the conv 4_3
        self.branch1[-1].weight.requires_grad = True

        self.conv_vgg_1 = nn.Conv2d(512, 128, 1)
        self.bn_vgg_1 = nn.BatchNorm2d(128, affine=False)
        self.conv1 = nn.Conv2d(128, 128, 3, stride=2)
        self.bn1 = nn.BatchNorm2d(128, affine=False)
        self.conv2 = nn.Conv2d(128, 128, 3, stride=2)
        self.bn2 = nn.BatchNorm2d(128, affine=False)
        self.conv3 = nn.Linear(128, 128)

        # FIXME: just hack loading pretrained weights for shared layers from
        # net pretrained on renders
        # model_dict = checkpoint['model_state_dict']
        # self.conv1.weight.data = model_dict['conv1.weight']
        # self.conv2.weight.data = model_dict['conv2.weight']
        # self.conv3.weight.data = model_dict['conv3.weight']
        # self.conv3.bias.data = model_dict['conv3.bias']
        #
        # self.bn1.weight = torch.nn.Parameter(model_dict['bn1.weight'])
        # self.bn1.bias = torch.nn.Parameter(model_dict['bn1.bias'])
        # self.bn1.running_mean = torch.nn.Parameter(model_dict['bn1.running_mean'], requires_grad=False)
        # self.bn1.running_var = torch.nn.Parameter(model_dict['bn1.running_var'], requires_grad=False)
        # self.bn1.num_batches_tracked = torch.nn.Parameter(model_dict['bn1.num_batches_tracked'], requires_grad=False)
        #
        # self.bn2.weight = torch.nn.Parameter(model_dict['bn2.weight'])
        # self.bn2.bias = torch.nn.Parameter(model_dict['bn2.bias'])
        # self.bn2.running_mean = torch.nn.Parameter(model_dict['bn2.running_mean'], requires_grad=False)
        # self.bn2.running_var = torch.nn.Parameter(model_dict['bn2.running_var'], requires_grad=False)
        # self.bn2.num_batches_tracked = torch.nn.Parameter(model_dict['bn2.num_batches_tracked'], requires_grad=False)

        self.eps = 1e-10

    def forward_shared(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.reshape(x.size(0), -1)
        x = self.conv3(x)

        if self.normalize_output:
            return L2Norm()(x)

        return x

    def forward_photo(self, x):
        x = F.relu(self.branch1(x))
        x = F.relu(self.bn_vgg_1(self.conv_vgg_1(x)))
        return self.forward_shared(x)

    def forward_render(self, x):
        x = F.relu(self.branch1(x))
        x = F.relu(self.bn_vgg_1(self.conv_vgg_1(x)))
        return self.forward_shared(x)

    def forward(self, anchor, pos, neg):
        anchor = self.forward_photo(anchor)
        pos = self.forward_render(pos)
        neg = self.forward_render(neg)

        return anchor, pos, neg

class MultimodalVGG16HardNetShared3l(nn.Module):
    def __init__(self, normalize_output=False):
        super(MultimodalVGG16HardNetShared3l, self).__init__()

        self.normalize_output = normalize_output
        self.branch1 = vgg16(pretrained=True)
        vgg16_layers = [
            'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2',
            'pool1',
            'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2',
            'pool2',
            'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3',
            'pool3',
            'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3',
            'pool4',
            'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3',
            'pool5'
        ]
        conv4_3_idx = vgg16_layers.index('conv4_3')
        self.branch1 = nn.Sequential(
            *list(self.branch1.features.children())[: conv4_3_idx + 1]
        )

        self.branch2 = HardNet()

        # load pretrained features on renders
        render_model_path = "/mnt/matylda1/ibrejcha/devel/adobetrips/python/runs/2019-08-30_11:57:24/models/SinglemodalHardNet3l_epoch_1_step_1030000"
        checkpoint = torch.load(render_model_path)

        branch1_dict = {}
        for key in checkpoint['model_state_dict'].keys():
            if 'branch1' in key:
                nk = key.replace('branch1.', '')
                branch1_dict.update({nk: checkpoint['model_state_dict'][key]})

        self.branch2.load_state_dict(branch1_dict)

        # lock weights in both branches to finetune only the shared part
        for param in self.branch1.parameters():
            param.requires_grad = False
        for param in self.branch2.parameters():
            param.requires_grad = False

        self.conv_vgg_1 = nn.Conv2d(512, 128, 1)
        self.bn_vgg_1 = nn.BatchNorm2d(128, affine=False)
        self.conv1 = nn.Conv2d(128, 128, 3, stride=2)
        self.bn1 = nn.BatchNorm2d(128, affine=False)
        self.conv2 = nn.Conv2d(128, 128, 3, stride=2)
        self.bn2 = nn.BatchNorm2d(128, affine=False)
        self.conv3 = nn.Linear(128, 128)

        # FIXME: just hack loading pretrained weights for shared layers from
        # net pretrained on renders
        # model_dict = checkpoint['model_state_dict']
        # self.conv1.weight.data = model_dict['conv1.weight']
        # self.conv2.weight.data = model_dict['conv2.weight']
        # self.conv3.weight.data = model_dict['conv3.weight']
        # self.conv3.bias.data = model_dict['conv3.bias']
        #
        # self.bn1.weight = torch.nn.Parameter(model_dict['bn1.weight'])
        # self.bn1.bias = torch.nn.Parameter(model_dict['bn1.bias'])
        # self.bn1.running_mean = torch.nn.Parameter(model_dict['bn1.running_mean'], requires_grad=False)
        # self.bn1.running_var = torch.nn.Parameter(model_dict['bn1.running_var'], requires_grad=False)
        # self.bn1.num_batches_tracked = torch.nn.Parameter(model_dict['bn1.num_batches_tracked'], requires_grad=False)
        #
        # self.bn2.weight = torch.nn.Parameter(model_dict['bn2.weight'])
        # self.bn2.bias = torch.nn.Parameter(model_dict['bn2.bias'])
        # self.bn2.running_mean = torch.nn.Parameter(model_dict['bn2.running_mean'], requires_grad=False)
        # self.bn2.running_var = torch.nn.Parameter(model_dict['bn2.running_var'], requires_grad=False)
        # self.bn2.num_batches_tracked = torch.nn.Parameter(model_dict['bn2.num_batches_tracked'], requires_grad=False)

        self.eps = 1e-10

    def forward_shared(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.reshape(x.size(0), -1)
        x = self.conv3(x)

        if self.normalize_output:
            return L2Norm()(x)

        return x

    def forward_photo(self, x):
        x = F.relu(self.branch1(x))
        x = F.relu(self.bn_vgg_1(self.conv_vgg_1(x)))
        return self.forward_shared(x)

    def forward_render(self, x):
        return self.forward_shared(F.relu(self.branch2(x)))

    def forward(self, anchor, pos, neg):
        anchor = self.forward_photo(anchor)
        pos = self.forward_render(pos)
        neg = self.forward_render(neg)

        return anchor, pos, neg

class MultimodalHardNetShared3l(nn.Module):
    def __init__(self, normalize_output=False):
        super(MultimodalHardNetShared3l, self).__init__()

        self.normalize_output = normalize_output
        self.branch1 = HardNet()
        self.branch2 = HardNet()

        model_path ="/mnt/matylda1/ibrejcha/devel/adobetrips/python/pretrained/hardnet/HardNet++_3ch.pth"
        checkpoint = torch.load(model_path)
        self.branch1.load_state_dict(checkpoint['state_dict'])

        # load pretrained features on renders
        render_model_path = "/mnt/matylda1/ibrejcha/devel/adobetrips/python/runs/2019-08-30_11:57:24/models/SinglemodalHardNet3l_epoch_1_step_1030000"
        checkpoint = torch.load(render_model_path)

        branch1_dict = {}
        for key in checkpoint['model_state_dict'].keys():
            if 'branch1' in key:
                nk = key.replace('branch1.', '')
                branch1_dict.update({nk: checkpoint['model_state_dict'][key]})

        self.branch2.load_state_dict(branch1_dict)

        # lock weights in both branches to finetune only the shared part
        for param in self.branch1.parameters():
            param.requires_grad = False
        for param in self.branch2.parameters():
            param.requires_grad = False

        self.conv1 = nn.Conv2d(128, 128, 3, stride=2)
        self.bn1 = nn.BatchNorm2d(128, affine=False)
        self.conv2 = nn.Conv2d(128, 128, 3, stride=2)
        self.bn2 = nn.BatchNorm2d(128, affine=False)
        self.conv3 = nn.Linear(128, 128)

        # FIXME: just hack loading pretrained weights for shared layers from
        # net pretrained on renders
        # model_dict = checkpoint['model_state_dict']
        # self.conv1.weight.data = model_dict['conv1.weight']
        # self.conv2.weight.data = model_dict['conv2.weight']
        # self.conv3.weight.data = model_dict['conv3.weight']
        # self.conv3.bias.data = model_dict['conv3.bias']
        #
        # self.bn1.weight = torch.nn.Parameter(model_dict['bn1.weight'])
        # self.bn1.bias = torch.nn.Parameter(model_dict['bn1.bias'])
        # self.bn1.running_mean = torch.nn.Parameter(model_dict['bn1.running_mean'], requires_grad=False)
        # self.bn1.running_var = torch.nn.Parameter(model_dict['bn1.running_var'], requires_grad=False)
        # self.bn1.num_batches_tracked = torch.nn.Parameter(model_dict['bn1.num_batches_tracked'], requires_grad=False)
        #
        # self.bn2.weight = torch.nn.Parameter(model_dict['bn2.weight'])
        # self.bn2.bias = torch.nn.Parameter(model_dict['bn2.bias'])
        # self.bn2.running_mean = torch.nn.Parameter(model_dict['bn2.running_mean'], requires_grad=False)
        # self.bn2.running_var = torch.nn.Parameter(model_dict['bn2.running_var'], requires_grad=False)
        # self.bn2.num_batches_tracked = torch.nn.Parameter(model_dict['bn2.num_batches_tracked'], requires_grad=False)

        self.eps = 1e-10

    def forward_shared(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.reshape(x.size(0), -1)
        x = self.conv3(x)

        if self.normalize_output:
            return L2Norm()(x)

        return x

    def forward_photo(self, x):
        return self.forward_shared(self.branch1(x))

    def forward_render(self, x):
        return self.forward_shared(self.branch2(x))

    def forward(self, anchor, pos, neg):
        anchor = self.forward_shared(self.branch1(anchor))
        pos = self.forward_shared(self.branch2(pos))
        neg = self.forward_shared(self.branch2(neg))

        return anchor, pos, neg

class MultimodalPatchNet5lShared2l(nn.Module):
    def __init__(self, normalize_output=False, needles=0):
        super(MultimodalPatchNet5lShared2l, self).__init__()

        self.normalize_output = normalize_output
        self.branch1 = SimplePatchNet5l(needles=needles)
        self.branch2 = SimplePatchNet5l(needles=needles)

        self.conv1 = nn.Conv2d(128, 128, 3, stride=2)
        self.conv2 = nn.Linear(128, 128)
        self.bn2 = nn.BatchNorm1d(128, affine=False, track_running_stats=False)

        self.eps = 1e-10

    def forward_shared(self, x):
        x = F.relu(self.conv1(x))  # .squeeze(3).squeeze(2)
        x = x.reshape(x.size(0), -1)
        #x = x.reshape(1, -1)
        x = self.conv2(x)

        if self.normalize_output:
            #if x.shape[0] > 1:
                # we cannot use batch norm on single sample, since we don't
                # track running stats.
                #x = self.bn2(x)
            x = x / (torch.norm(x, dim=1).reshape(-1, 1) + self.eps)

        return x

    def forward_photo(self, x):
        return self.forward_shared(self.branch1(x))

    def forward_render(self, x):
        return self.forward_shared(self.branch2(x))

    def forward(self, anchor, pos, neg):
        anchor = self.forward_shared(self.branch1(anchor))
        pos = self.forward_shared(self.branch2(pos))
        neg = self.forward_shared(self.branch2(neg))

        return anchor, pos, neg

class MultimodalPatchNet5lShared2lPhotoONNX(MultimodalPatchNet5lShared2l):
    """
    class for wrapping photo branch of our two branch network for ONNX export.
    """
    def __init__(self, normalize_output=False, needles=0):
        super(MultimodalPatchNet5lShared2lPhotoONNX, self).__init__(normalize_output, needles)

    def forward(self, x):
        return super().forward_photo(x)

class MultimodalPatchNet5lShared2lRenderONNX(MultimodalPatchNet5lShared2l):
    """
    class for wrapping render branch of our two branch network for ONNX export.
    """
    def __init__(self, normalize_output=False, needles=0):
        super(MultimodalPatchNet5lShared2lRenderONNX, self).__init__(normalize_output, needles)

    def forward(self, x):
        return super().forward_render(x)

class MultimodalPatchNet5lShared2lFCN(nn.Module):
    def __init__(self, normalize_output=False, strides=[2],
                 strides_branch=[2, 2, 2, 2], dilations=[1],
                 dilations_branch=[1, 1, 1, 1], needles=0):
        super(MultimodalPatchNet5lShared2lFCN, self).__init__()

        self.normalize_output = normalize_output
        self.branch1 = SimplePatchNet5l(strides=strides_branch, dilations=dilations_branch, needles=needles)
        self.branch2 = SimplePatchNet5l(strides=strides_branch, dilations=dilations_branch, needles=needles)
        self.conv1 = nn.Conv2d(128, 128, 3, stride=strides[0], dilation=dilations[0])
        self.conv2 = nn.Conv2d(128, 128, 1)

        self.eps = 1e-10

    def forward_shared(self, x):
        x = F.relu(self.conv1(x))
        #x = x.reshape(x.size(0), -1)
        x = self.conv2(x)

        if self.normalize_output:
            if x.shape[2] == 1:
                x = x.reshape(x.shape[0], x.shape[1])
                x = x / (torch.norm(x, dim=1).reshape(-1, 1) + self.eps)
                x = x.reshape(x.shape[0], x.shape[1], 1, 1)
            else:
                x = x / (torch.norm(x, dim=1) + self.eps)

        return x

    def forward_photo(self, x):
        return self.forward_shared(self.branch1(x))

    def forward_render(self, x):
        return self.forward_shared(self.branch2(x))

    def forward(self, anchor, pos, neg):
        anchor = self.forward_shared(self.branch1(anchor))
        pos = self.forward_shared(self.branch2(pos))
        neg = self.forward_shared(self.branch2(neg))

        return anchor, pos, neg


class MultimodalKeypointPatchNet5lShared2l(MultimodalPatchNet5lShared2l):
    def __init__(self, normalize_output=False, pretrained=True, needles=0):
        super(MultimodalKeypointPatchNet5lShared2l, self).__init__(normalize_output, needles=needles)

        self.linear1 = nn.Linear(128, 1, bias=False)
        model_path ="/mnt/matylda1/ibrejcha/devel/adobetrips/python/runs/2019-06-28_21:13:01_hardnet_orig_impl_filter_50_1000000_randomhardneg_newimpl/models/MultimodalPatchNet5lShared2l_epoch_3_step_1310000"
        if pretrained:
            if os.path.isfile(model_path):
                # load pretrained weights
                checkpoint = torch.load(model_path)
                model_dict = self.state_dict()
                model_dict.update(checkpoint['model_state_dict'])
                self.load_state_dict(model_dict)

    def forward_photo(self, x):
        x = self.forward_shared(self.branch1(x))
        score = self.linear1(F.relu(x))
        if self.training:
            score = x.shape[0] * nn.functional.softmax(score, dim=0)
        else:
            score = torch.sigmoid(score)
        score = score.view(-1)
        return x, score

    def forward_render(self, x):
        x = self.forward_shared(self.branch2(x))
        score = self.linear1(F.relu(x))
        if self.training:
            score = x.shape[0] * nn.functional.softmax(score, dim=0)
        else:
            score = torch.sigmoid(score)
        score = score.view(-1)
        return x, score

    def forward(self, anchor, pos, neg):
        anchor, score_anchor = self.forward_photo(anchor)
        pos, score_pos = self.forward_render(pos)
        neg, score_neg = self.forward_render(neg)
        return anchor, pos, neg, score_anchor, score_pos, score_neg

class MultimodalKeypointPatchNet5lShared2lFCN(MultimodalPatchNet5lShared2lFCN):
    def __init__(self, normalize_output=False, pretrained=True,
                 strides=[2], strides_branch=[2, 2, 2, 2], dilations=[1],
                 dilations_branch=[1, 1, 1, 1], needles=0):
        super(MultimodalKeypointPatchNet5lShared2lFCN, self).__init__(normalize_output, strides=strides, strides_branch=strides_branch, dilations=dilations, dilations_branch=dilations_branch, needles=needles)

        self.linear1 = nn.Conv2d(128, 1, 1, bias=False)

        if pretrained:
            # load pretrained weights
            model_path ="/mnt/matylda1/ibrejcha/devel/adobetrips/python/runs/2019-06-28_21:13:01_hardnet_orig_impl_filter_50_1000000_randomhardneg_newimpl/models/MultimodalPatchNet5lShared2l_epoch_3_step_1310000"
            if os.path.isfile(model_path):
                checkpoint = torch.load(model_path)
                conv2_w = checkpoint['model_state_dict']['conv2.weight']
                checkpoint['model_state_dict']['conv2.weight'] = conv2_w.reshape(conv2_w.shape[0], conv2_w.shape[1], 1, 1)
                model_dict = self.state_dict()
                model_dict.update(checkpoint['model_state_dict'])
                self.load_state_dict(model_dict)
            else:
                print("WARNING: pretrained model not found: ", model_path)

    def forward_photo(self, x):
        x = self.forward_shared(self.branch1(x))
        score = self.linear1(F.relu(x))
        if self.training:
            score = x.shape[0] * nn.functional.softmax(score, dim=0)
        else:
            score = torch.sigmoid(score)
        if score.shape[2] == 1:
            score = score.view(-1)
        return x, score

    def forward_render(self, x):
        x = self.forward_shared(self.branch2(x))
        score = self.linear1(F.relu(x))
        if self.training:
            score = x.shape[0] * nn.functional.softmax(score, dim=0)
        else:
            score = torch.sigmoid(score)
        if score.shape[2] == 1:
            score = score.view(-1)
        return x, score

    def forward(self, anchor, pos, neg):
        anchor, score_anchor = self.forward_photo(anchor)
        pos, score_pos = self.forward_render(pos)
        neg, score_neg = self.forward_render(neg)
        return anchor, pos, neg, score_anchor, score_pos, score_neg

class MultimodalPatchNet5lShared2lBN(nn.Module):
    def __init__(self, normalize_output=False):
        super(MultimodalPatchNet5lShared2lBN, self).__init__()

        self.normalize_output = normalize_output
        self.branch1 = PatchNetBN5l()
        self.branch2 = PatchNetBN5l()

        self.conv1 = nn.Conv2d(128, 128, 3, stride=2)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Linear(128, 128, bias=False)

        self.eps = 1e-10

    def forward_shared(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = x.reshape(x.size(0), -1)
        x = self.conv2(x)

        if self.normalize_output:
            x = x / (torch.norm(x, dim=1).reshape(-1, 1) + self.eps)

        return x

    def forward_photo(self, x):
        return self.forward_shared(self.branch1(x))

    def forward_render(self, x):
        return self.forward_shared(self.branch2(x))

    def forward(self, anchor, pos, neg):
        anchor = self.forward_shared(self.branch1(anchor))
        pos = self.forward_shared(self.branch2(pos))
        neg = self.forward_shared(self.branch2(neg))


        return anchor, pos, neg


class MultimodalPatchNet(nn.Module):

    def __init__(self):
        super(MultimodalPatchNet, self).__init__()
        self.branch1 = SimplePatchNet()
        self.branch2 = SimplePatchNet()

    def forward(self, anchor, pos, neg):
        anchor = self.branch1(anchor)
        pos = self.branch2(pos)
        neg = self.branch2(neg)

        return anchor, pos, neg

import torch
from torch import nn
from torchvision.models.resnet import BasicBlock, conv1x1



class MultimodalResNetShared2l(nn.Module):
    def __init__(self, normalize_output=False, zero_init_residual=False):
        super(MultimodalResNetShared2l, self).__init__()
        self.normalize_output = normalize_output
        self.inplanes = 64

        # Define the neural network here
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


        # Left branch
        self.conv1_left = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.layer1_left = self._make_layer(BasicBlock, 64, 3)
        self.layer2_left = self._make_layer(BasicBlock, 128, 4, stride=2)

        # Right branch
        self.conv1_right = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.layer1_right = self._make_layer(BasicBlock, 64, 3)
        self.layer2_right = self._make_layer(BasicBlock, 128, 4, stride=2)

        # Common blocks
        #self.inplanes *= 2
        self.layer3 = self._make_layer(BasicBlock, 256, 6, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, 128) # Output 128

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                #if isinstance(m, Bottleneck):
                #    nn.init.constant_(m.bn3.weight, 0)
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def input_norm(self, x):
        flat = x.contiguous().view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)

    def forward_shared(self, x):
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        if self.normalize_output:
            x = L2Norm()(x)
        return x

    def forward_photo(self, x):
        x = self.input_norm(x)
        xl = self.conv1_left(x)
        xl = self.relu(xl)
        xl = self.maxpool(xl)
        xl = self.layer1_left(xl)
        xl = self.layer2_left(xl)

        xl = self.forward_shared(xl)
        return xl

    def forward_render(self, x):
        x = self.input_norm(x)
        xr = self.conv1_right(x)
        xr = self.relu(xr)
        xr = self.maxpool(xr)
        xr = self.layer1_right(xr)
        xr = self.layer2_right(xr)
        xr = self.forward_shared(xr)
        return xr

    def forward(self, anchor, pos, neg):

        anchor = self.forward_photo(anchor)
        pos = self.forward_render(pos)
        neg = self.forward_render(neg)
        return anchor, pos, neg


if __name__ == '__main__':
    #from torchsummary import summary
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #summary(MultimodalResNetShared2l().to(device), [(3, 64, 64), (3, 64, 64), (3, 64, 64)])
    print(LockedSinglemodalVGG16FinetunedShared3l().to(device), [(3, 64, 64), (3, 64, 64), (3, 64, 64)])

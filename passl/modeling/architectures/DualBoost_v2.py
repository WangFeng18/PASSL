# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn as nn

from ...modules.init import init_backbone_weight
from .builder import MODELS
from ..backbones import build_backbone
from ..necks import build_neck
from ..heads import build_head
from ..predictors import build_predictor


@MODELS.register()
class DualBoost(nn.Layer):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 predictor=None,
                 dim=256):
        """
        Args:
            backbone (dict): config of backbone.
            neck (dict): config of neck.
            head (dict): config of head.
            dim (int): feature dimension. Default: 256.
        """
        super(DualBoost, self).__init__()

        # create the encoders
        # num_classes is the output fc dimension
        self.towers = []

        #TODO try to see if the predictor is indispensable in the dualboost imagination
        # self.towers.append(nn.Sequential(build_backbone(backbone), build_neck(neck), build_predictor(predictor)))
        self.towers.append(nn.Sequential(build_backbone(backbone), build_neck(neck)))
        self.towers.append(nn.Sequential(build_backbone(backbone), build_neck(neck)))
        self.backbone = self.towers[0][0]

        self.head = build_head(head)
        self.register_buffer("id_main_tower", paddle.zeros([1], 'int64'))

    def train_iter(self, *inputs, **kwargs):
        id_main_tower = self.id_main_tower[0]
        id_target_tower = 1 - id_main_tower
        self.id_main_tower[0] = id_target_tower

        # stop gradients for target_tower
        # TODO ensure if the optimizer requires to be rearranged.
        self.recover_gradient(self.towers[id_main_tower])
        self.stop_gradient(self.towers[id_target_tower])

        img_a, img_b = inputs
        a1 = self.towers[id_main_tower](img_a)
        b1 = self.towers[id_target_tower](img_b)

        a1 = nn.functional.normalize(a1, axis=1)
        b1 = nn.functional.normalize(b1, axis=1)

        a2 = self.towers[id_main_tower](img_b)
        b2 = self.towers[id_target_tower](img_a)

        a2 = nn.functional.normalize(a2, axis=1)
        b2 = nn.functional.normalize(b2, axis=1)
        outputs = self.head(a1, b1, a2, b2)

        return outputs

    def stop_gradient(self, network):
        for param in network.parameters():
            param.stop_gradient = True

    def recover_gradient(self, network):
        for param in network.parameters():
            param.stop_gradient = False

    def forward(self, *inputs, mode='train', **kwargs):
        if mode == 'train':
            return self.train_iter(*inputs, **kwargs)
        elif mode == 'test':
            return self.test_iter(*inputs, **kwargs)
        elif mode == 'extract':
            return self.backbone(*inputs)
        else:
            raise Exception("No such mode: {}".format(mode))

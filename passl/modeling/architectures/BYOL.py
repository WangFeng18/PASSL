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

import math
import paddle
import paddle.nn as nn

from ...modules.init import init_backbone_weight
from .builder import MODELS
from ..backbones import build_backbone
from ..necks import build_neck
from ..heads import build_head
from ..predictors import build_predictor


@MODELS.register()
class BYOL(nn.Layer):
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
        super(BYOL, self).__init__()

        # create the encoders
        # num_classes is the output fc dimension
        self.towers = []
        self.base_m = 0.996

        #TODO try to see if the predictor is indispensable in the dualboost imagination
        # self.towers.append(nn.Sequential(build_backbone(backbone), build_neck(neck), build_predictor(predictor)))
        self.towers.append(nn.Sequential(build_backbone(backbone), build_neck(neck)))
        self.towers.append(nn.Sequential(build_backbone(backbone), build_neck(neck)))

        self.predictor = build_predictor(predictor)
        self.backbone = self.towers[0][0]

        for param_q, param_k in zip(self.towers[0].parameters(),self.towers[1].parameters()):
            param_k.set_value(param_q)  # initialize
        
        self.stop_gradient(self.towers[1])
        self.head = build_head(head)
        self.register_buffer("id_main_tower", paddle.zeros([1], 'int64'))
    
    @paddle.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.towers[0].parameters(),
                                    self.towers[1].parameters()):
            paddle.assign((param_k * self.m + param_q * (1. - self.m)), param_k)
            param_k.stop_gradient = True

    def train_iter(self, *inputs, **kwargs):
        
        current_iter = kwargs['current_iter']
        total_iters =  kwargs['total_iters']
        self.m = 1 - (1-self.base_m) * (1 + math.cos(math.pi*current_iter/total_iters))/2.0
        print(self.m)
        
        self._momentum_update_key_encoder()
        img_a, img_b = inputs
        a = self.predictor(self.towers[0](img_a))
        b = self.towers[1](img_b)

        a = nn.functional.normalize(a, axis=1)
        b = nn.functional.normalize(b, axis=1)
        outputs = self.head(a, b)

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

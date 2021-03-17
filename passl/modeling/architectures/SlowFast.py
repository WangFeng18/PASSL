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


@MODELS.register()
class SlowFast(nn.Layer):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 predictor=None,
                 dim=256,
                 target_decay_method='fixed',
                 target_decay_rate=0.996,
                 align_init_network=True,
                 use_synch_bn=True):
        """
        Args:
            backbone (dict): config of backbone.
            neck (dict): config of neck.
            head (dict): config of head.
            dim (int): feature dimension. Default: 256.
        """
        super(SlowFast, self).__init__()

        # create the encoders
        # num_classes is the output fc dimension
        self.towers = nn.LayerList()
        self.base_m = target_decay_rate
        self.target_decay_method = target_decay_method
        
        neck1 = build_neck(neck)
        neck2 = build_neck(neck)
        neck1.init_parameters()
        neck2.init_parameters()
        self.towers.append(nn.Sequential(build_backbone(backbone), neck1))
        self.towers.append(nn.Sequential(build_backbone(backbone), neck2))
        self.predictor = build_neck(predictor)

        # Convert BatchNorm*d to SyncBatchNorm*d
        if use_synch_bn:
            self.towers[0] = nn.SyncBatchNorm.convert_sync_batchnorm(self.towers[0])
            self.towers[1] = nn.SyncBatchNorm.convert_sync_batchnorm(self.towers[1])
            self.predictor = nn.SyncBatchNorm.convert_sync_batchnorm(self.predictor)

        self.backbone = self.towers[0][0]
        # self.neck1 = self.towers[0][1]

        # TODO IMPORTANT! Explore if the initialization requires to be synchronized
        if align_init_network:
            for param_q, param_k in zip(self.towers[0].parameters(),self.towers[1].parameters()):
                param_k.set_value(param_q)  # initialize
                
        self.head = build_head(head)


    def train_iter(self, *inputs, **kwargs):
        
        # current_iter = kwargs['current_iter']
        # total_iters =  kwargs['total_iters']

        img_a, img_b = inputs
        a1 = self.predictor(self.towers[0](img_a))
        b1 = self.towers[1](img_b)

        a1 = nn.functional.normalize(a1, axis=1)
        b1 = nn.functional.normalize(b1, axis=1)

        a2 = self.predictor(self.towers[0](img_b))
        b2 = self.towers[1](img_a)

        a2 = nn.functional.normalize(a2, axis=1)
        b2 = nn.functional.normalize(b2, axis=1)

        outputs = self.head(a1, b1, a2, b2)

        return outputs

    def forward(self, *inputs, mode='train', **kwargs):
        if mode == 'train':
            return self.train_iter(*inputs, **kwargs)
        elif mode == 'test':
            return self.test_iter(*inputs, **kwargs)
        elif mode == 'extract':
            return self.backbone(*inputs)
        else:
            raise Exception("No such mode: {}".format(mode))

    def separate_parameters(self):
        param_dict = {
            'online_backbone': self.towers[0][0].parameters(),
            'online_neck': self.towers[0][1].parameters(),
            'target_backbone': self.towers[1][0].parameters(),
            'target_neck': self.towers[1][1].parameters(),
            'predictor': self.predictor.parameters(),
        }
        return param_dict
    
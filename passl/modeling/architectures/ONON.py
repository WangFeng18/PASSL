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
import numpy as np

from ...modules.init import init_backbone_weight
from .builder import MODELS
from ..backbones import build_backbone
from ..necks import build_neck
from ..heads import build_head
from .moco import concat_all_gather

@MODELS.register()
class ONON(nn.Layer):
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
                 use_synch_bn=True,
                 K=32768):
        """
        Args:
            backbone (dict): config of backbone.
            neck (dict): config of neck.
            head (dict): config of head.
            dim (int): feature dimension. Default: 256.
        """
        super(ONON, self).__init__()

        self.K = K
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
        for param_q, param_k in zip(self.towers[0].parameters(),self.towers[1].parameters()):
            param_k.stop_gradient = True

        if align_init_network:
            for param_q, param_k in zip(self.towers[0].parameters(),self.towers[1].parameters()):
                param_k.set_value(param_q)  # initialize
                
        self.head = build_head(head)

        # create the queue
        self.register_buffer("queue", paddle.randn([dim, K]))
        self.queue = nn.functional.normalize(self.queue, axis=0)

        self.register_buffer("queue_ptr", paddle.zeros([1], 'int64'))

    @paddle.no_grad()
    def _dequeue_and_enqueue(self, keys):
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.transpose([1, 0])
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @paddle.no_grad()
    def update_target_network(self):
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

        if self.target_decay_method == 'cosine':
            self.m = 1 - (1-self.base_m) * (1 + math.cos(math.pi*current_iter/total_iters))/2.0   # 47.0
        elif self.target_decay_method == 'fixed':
            self.m = self.base_m   # 55.7
        else:
            raise NotImplementedError

        self.pos_prob = (0.8 - 0.0) * current_iter/total_iters
        use_other = self.sample()

        # self.update_target_network()
        img_a, img_b = inputs
        a1 = self.predictor(self.towers[0](img_a))
        a1 = nn.functional.normalize(a1, axis=1)
        b1 = self.towers[1](img_b)
        b1 = nn.functional.normalize(b1, axis=1)
        b1.stop_gradient = True
        if use_other:
            similarities = paddle.matmul(b1, self.queue)
            indices = paddle.argmax(similarities, axis=1)
            c = self.queue[indices]
            c.stop_gradient = True

        a2 = self.predictor(self.towers[0](img_b))
        a2 = nn.functional.normalize(a2, axis=1)
        if not use_other:
            b2 = self.towers[1](img_a)
            b2 = nn.functional.normalize(b2, axis=1)
            b2.stop_gradient = True

        if use_other:
            outputs = self.head(a1, c, a2, c)
        else:
            outputs = self.head(a1, b1, a2, b2)

        self._dequeue_and_enqueue(b1)

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

    def sample(self):
        t = np.random.rand()
        return t < self.pos_prob

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

from .builder import PREDICTORS
from ...modules.init import normal_init, kaiming_init, constant_


def _init_parameters(module, init_linear='normal', std=0.01, bias=0.):
    assert init_linear in ['normal', 'kaiming'], \
        "Undefined init_linear: {}".format(init_linear)
    for m in module.sublayers():
        if isinstance(m, nn.Linear):
            if init_linear == 'normal':
                normal_init(m, std=std, bias=bias)
            else:
                kaiming_init(m, mode='fan_in', nonlinearity='relu')
        elif isinstance(
                m,
            (nn.BatchNorm1D, nn.BatchNorm2D, nn.GroupNorm, nn.SyncBatchNorm)):
            if m.weight is not None:
                constant_(m.weight, 1)
            if m.bias is not None:
                constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2D):
            kaiming_init(m, mode='fan_in', nonlinearity='relu')



@PREDICTORS.register()
class Predictor(nn.Layer):
    """The non-linear neck in MoCo v2: fc-relu-fc.
    """

    def __init__(self,
                 in_channels,
                 hid_channels,
                 out_channels):
        super(Predictor, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hid_channels), 
            nn.BatchNorm1D(hid_channels),
            nn.ReLU(),
            nn.Linear(hid_channels, out_channels))

        self.init_parameters()

    def init_parameters(self, init_linear='normal'):
        _init_parameters(self, init_linear)

    def forward(self, x):
        return self.mlp(x)

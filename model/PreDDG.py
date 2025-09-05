#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project : PreDDG
@File    : PreDDG.py
@IDE     : PyCharm
@Author  : Henghui FAN
@Date    : 2025/3/26
"""
import torch
import torch.nn as nn
from .graph_unet import DDGGraphNet


class Preddg(torch.nn.Module):
    def __init__(self, node_dim=1280):
        super().__init__()
        self.fea_norm = nn.LayerNorm(node_dim)
        self.encoder = DDGGraphNet()

    def forward(self, wt, mut, mask=None):

        return self.encoder(wt['graph'], mut['graph'])


#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project : PreDDG
@File    : graph_unet.py
@IDE     : PyCharm 
@Author  : Henghui FAN
@Date    : 2025/7/8
"""
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool, TopKPooling
from torch_geometric.nn.aggr import Aggregation, MultiAggregation
from torch_geometric.typing import Adj, OptPairTensor, Size
from torch_geometric.utils import add_self_loops


class WeightedSAGEConv(MessagePassing):
    def __init__(self,
                 in_channels: Union[int, Tuple[int, int]],
                 out_channels: int,
                 aggr: Optional[Union[str, List[str], Aggregation]] = "mean",
                 bias: bool = True,
                 **kwargs, ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        # 若 in_channels 为单个整数，将其转换为元组 (in_channels, in_channels)
        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        # 调用父类 MessagePassing 的构造函数，传入聚合方式和其他参数
        super().__init__(aggr, **kwargs)
        if isinstance(self.aggr_module, MultiAggregation):
            aggr_out_channels = self.aggr_module.get_out_channels(
                in_channels[0])
        else:
            # 否则，聚合后的输出通道数与输入通道数相同
            aggr_out_channels = in_channels[0]
        # 初始化源节点和目标节点的线性变换层
        self.lin_l = nn.Linear(aggr_out_channels, out_channels, bias=bias)
        self.lin_r = nn.Linear(in_channels[1], out_channels, bias=bias)

    def forward(
            self,
            x: Union[Tensor, OptPairTensor],
            edge_index: Adj,
            edge_weight: Optional[Tensor] = None,
            size: Size = None):
        # 若输入 x 是单个张量，将其转换为 (x, x) 形式的张量对
        if isinstance(x, Tensor):
            x = (x, x)
        x = (self.lin_l(x[0]), self.lin_r(x[1]))
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=size)
        return out

    def message(self, x_j: Tensor, edge_weight: Tensor = None):
        # x_j: [E, out_channels]
        if edge_weight is None:
            return x_j
        return edge_weight.view(-1, 1) * x_j

    def update(self, aggr_out):
        return aggr_out


class DoubleGraphConv(nn.Module):
    """双卷积层 (带残差连接)"""

    def __init__(self, in_channels, out_channels, aggr='mean'):
        super().__init__()
        self.conv = nn.ModuleList()
        self.conv.append(WeightedSAGEConv(in_channels, out_channels, aggr=aggr))
        self.conv.append(ResGraphConv(out_channels, out_channels, aggr=aggr))

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor = None):
        for conv in self.conv:
            x = conv(x, edge_index, edge_weight)
        return x


class ResGraphConv(MessagePassing):
    """使用SAGEConv的动态图卷积层"""

    def __init__(self,
                 in_channels: Union[int, Tuple[int, int]],
                 out_channels: int,
                 aggr: Optional[Union[str, List[str], Aggregation]] = "mean",
                 ):
        super().__init__(aggr=None)
        # SAGEConv
        self.conv = WeightedSAGEConv(in_channels, out_channels, aggr=aggr)

        # 残差连接
        self.residual = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()

        # 层归一化
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor = None) -> Tensor:
        identity = self.residual(x)
        edge_index, edge_weight = add_self_loops(
            edge_index, edge_weight, fill_value=1.0, num_nodes=x.size(0)
        )
        # 图卷积
        out = self.conv(x, edge_index, edge_weight)

        # 残差连接和归一化
        out = self.norm(out + identity)

        return F.gelu(out)


class EncoderBlock(nn.Module):
    def __init__(self, conv1, conv2):
        super().__init__()
        self.conv1 = conv1
        self.conv2 = conv2

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight)
        x = self.conv2(x, edge_index, edge_weight)
        return x


class GraphDiffEncoder(nn.Module):
    """融合WT和MUT图特征"""

    def __init__(self, dim):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Dropout(),
        )
        # self.norm = nn.LayerNorm(dim)

    def forward(self, x_wt, x_mut):
        combined = torch.cat([x_wt, x_mut], dim=-1)
        # combined = self.norm(x_mut) - self.norm(x_wt)
        return self.fuse(combined)


class GraphUNet(nn.Module):
    def __init__(self, in_channels=1280, hidden_channels=256,
                 depth=3,
                 pool_ratios=0.5,
                 sum_res=False):
        super().__init__()
        self.depth = depth
        self.pool_ratios = [pool_ratios] * depth if isinstance(pool_ratios, float) else pool_ratios
        self.sum_res = sum_res

        # 下采样层
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.encoders.append(DoubleGraphConv(in_channels, hidden_channels))
        for i in range(depth):
            self.pools.append(TopKPooling(hidden_channels, self.pool_ratios[i]))
            # self.pools.append(AttentionPooling(hidden_channels, ratio=self.pool_ratios[i]))
            self.encoders.append(ResGraphConv(hidden_channels, hidden_channels))

        # 瓶颈层
        bottleneck_dim = hidden_channels
        self.bottleneck = EncoderBlock(
            ResGraphConv(bottleneck_dim, bottleneck_dim),
            ResGraphConv(bottleneck_dim, bottleneck_dim)
        )

        # 上采样层
        in_channels = hidden_channels if sum_res else 2 * hidden_channels
        # self.upsamplers = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(depth):
            # self.upsamplers.append()
            self.decoders.append(ResGraphConv(in_channels, hidden_channels))

        self.global_pool = lambda x, batch: torch.cat([
            global_mean_pool(x, batch),
            global_max_pool(x, batch)
        ], dim=1)
        self.final_dim = hidden_channels * 2

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        x = self.encoders[0](x, edge_index, edge_weight)  # 1280 -> 256
        # 存储每一层下采样后的节点特征
        xs = [x]
        # 存储每一层的边索引
        edge_indices = [edge_index]
        # 存储每一层的边权重
        edge_weights = [edge_weight]
        batch_list = [batch]
        # 存储每一层池化操作的索引
        perms = []
        # 编码路径
        for i in range(1, self.depth + 1):
            # edge_index, edge_weight = self.augment_adj(edge_index, edge_weight, x.size(0))
            x, edge_index, edge_weight, batch, indices, _ = self.pools[i - 1](x, edge_index, edge_weight, batch)  # 节点数量减半
            # x, edge_index, edge_weight, batch, indices = self.pools[i - 1](x, edge_index, edge_weight, batch)
            x = self.encoders[i](x, edge_index, edge_weight)  # 256 -> 256

            if i < self.depth:
                xs += [x]
                edge_indices += [edge_index]
                edge_weights += [edge_weight]
                batch_list += [batch]

            perms += [indices]

        # 瓶颈层
        x = self.bottleneck(x, edge_index, edge_weight)

        # 解码路径
        for i in range(self.depth):
            j = self.depth - 1 - i

            res = xs[j]
            edge_index = edge_indices[j]
            edge_weight = edge_weights[j]
            indices = perms[j]
            batch = batch_list[j]  # 恢复对应层的batch信息

            up = torch.zeros_like(res)
            up[indices] = x
            x = res + up if self.sum_res else torch.cat((res, up), dim=-1)
            x = self.decoders[i](x, edge_index, edge_weight)  # 256 -> 256，节点复原

        return self.global_pool(x, batch)  # [B, 2 * hidden_channels]


class DDGGraphNet(nn.Module):
    """支持突变前/后图输入的 ΔΔG 回归网络"""

    def __init__(self):
        super().__init__()
        self.encoder = GraphUNet()
        self.diff_encoder = GraphDiffEncoder(dim=self.encoder.final_dim)
        self.regressor = nn.Sequential(
            nn.Linear(self.encoder.final_dim // 2, 1)  # 输出 ΔΔG
        )
        # self.classifier = nn.Sequential(
        #     nn.Linear(self.encoder.final_dim // 2, 1)
        # )

    def forward(self, wt_data, mut_data):
        # 支持PyG Data对象格式输入，提取边权重
        wt_edge_weight = wt_data.edge_attr if hasattr(wt_data, 'edge_attr') else None
        wt_edge_index = wt_data.edge_index if hasattr(wt_data, 'edge_index') else None
        mut_edge_weight = mut_data.edge_attr if hasattr(mut_data, 'edge_attr') else None
        mut_edge_index = mut_data.edge_index if hasattr(mut_data, 'edge_index') else None

        # 分别编码WT和MUT图
        h_wt = self.encoder(
            wt_data.x,
            wt_edge_index,
            wt_edge_weight,
            wt_data.batch if hasattr(wt_data, 'batch') else None
        )

        h_mut = self.encoder(
            mut_data.x,
            mut_edge_index,
            mut_edge_weight,
            mut_data.batch if hasattr(mut_data, 'batch') else None
        )

        # 融合特征并预测ΔΔG
        h = self.diff_encoder(h_wt, h_mut)
        ddg = self.regressor(h).squeeze(-1)  # [B]
        return ddg, ddg

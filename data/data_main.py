#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project : PreDDG
@File    : data_main.py
@IDE     : PyCharm
@Author  : Henghui FAN
@Date    : 2025/2/26
"""
import sys
import os
import pandas as pd
import torch
from torch.nn.functional import pad
from functools import partial
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from .dataset_process import DatasetProcess

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


class DataMain(pl.LightningDataModule):
    def __init__(self,
                 train_name=None,
                 val_name=None,
                 test_name='cdna2,M28,M38,ptmul_nr',
                 fold=None,
                 **kwargs):
        """
        :type train_name: str
        :type test_name: str
        :type val_name: str
        :type fold: int
        :param train_name: 训练集名称
        :param test_name: 测试集名称
        :param fold: 交叉验证的折数
        """
        super().__init__()
        self.train_name = train_name
        self.val_name = val_name
        self.test_name = test_name
        self.fold = fold
        self.kwargs = kwargs
        if train_name is not None:
            self.batch_size = kwargs['batch_size']
        self.num_workers = 4
        self.cutoff = 0.1  # 接触图阈值


    def create_dataset(self, file, apply_thermodynamic_reversibility, fold_dir=None):
        """
        创建数据集
        :param file: 数据集文件路径
        :param apply_thermodynamic_reversibility: 是否应用反对称数据增强
        :param fold_dir: 交叉验证文件路径
        :return: 数据集对象
        """

        ism_folder = f'./data/dataset/{file}/ism/'
        # 检查ism_folder是否存在,以及ism_folder是否为空
        if not os.path.exists(ism_folder) or len(os.listdir(ism_folder)) == 0:
            data_dir = f'./data/dataset/{file}/mutations/{file}.csv'
            DatasetProcess(data_dir,
                           train=True if file == 'cdna' else False,
                           device=self.kwargs['device'],
                           )

        original_dataset = CustomDataset(ism_folder, apply_thermodynamic_reversibility, fold_dir)

        return original_dataset

    def create_dataloader(self, dataset, shuffle=False):
        """
        创建数据加载器
        :param dataset: 数据集对象
        :param shuffle: 是否打乱数据
        :return: 数据加载器对象
        """
        if shuffle:
            batch_size = self.batch_size
        else:
            batch_size = 64
        return DataLoader(dataset,
                          batch_size=batch_size,
                          num_workers=self.num_workers,
                          shuffle=shuffle,
                          pin_memory=False,
                          collate_fn=partial(pair_pad_collate_fn, contact_cutoff=self.cutoff),
                          persistent_workers=True if self.num_workers > 0 else False,
                          multiprocessing_context='spawn',
                          )

    def train_dataloader(self):
        if self.train_name is None and self.fold is None:
            return None
        train_data_dir = self.train_name
        fold_dir = None
        if self.fold is not None:
            fold_dir = (f'./data/dataset/'
                        f'{train_data_dir}/mutations/fold_data/fold_{self.fold}_train.csv')
        train_set = self.create_dataset(train_data_dir, apply_thermodynamic_reversibility=True, fold_dir=fold_dir)

        return self.create_dataloader(train_set, shuffle=True)

    def val_dataloader(self):
        if self.val_name is None and self.fold is None:
            return None

        fold_dir = None
        if self.fold is not None:
            val_dir = self.train_name
            fold_dir = (f'./data/dataset/'
                        f'{self.train_name}/mutations/fold_data/fold_{self.fold}_val.csv')
        else:
            val_dir = self.val_name

        if not val_dir:
            return None
        val_set = self.create_dataset(val_dir, apply_thermodynamic_reversibility=False,
                                      fold_dir=fold_dir)
        return self.create_dataloader(val_set, shuffle=False)

    def test_dataloader(self):
        test_datas = []
        for test_name in self.test_name.split(','):
            test_set = self.create_dataset(test_name, apply_thermodynamic_reversibility=False)
            test_datas.append(self.create_dataloader(test_set, shuffle=False))
        return test_datas


def pair_pad_collate_fn(batch, contact_cutoff=None):
    """
    batch: list of (wt_info, mut_info, ddg_label)
        - wt_info['node']: [L, D]
        - wt_info['seq']: [L]
    Returns:
        dict of batched tensors
    """
    wt_graphs, mut_graphs = [], []
    wt_nodes, mut_nodes, masks, ddgs = [], [], [], []
    # wt_seq, mut_seq = [], []

    max_len = max(max(item[0]['node'].size(0), item[1]['node'].size(0)) for item in batch)

    for wt_info, mut_info, ddg_label in batch:
        if contact_cutoff is not None:
            # 构建野生型图
            wt_graph = build_pyg_graph(wt_info['node'], wt_info['contact'], contact_cutoff)
            wt_graphs.append(wt_graph)
            # 构建突变型图
            mut_graph = build_pyg_graph(mut_info['node'], mut_info['contact'], contact_cutoff)
            mut_graphs.append(mut_graph)

            # ddgs.append(ddg_label)
        # else:
        L_wt = wt_info['node'].size(0)

        # 统一 pad 到 max_len
        def pad_node(node, target_len):
            return pad(node, (0, 0, 0, target_len - node.size(0)), value=0)

        # def pad_seq(seq, target_len):
        #     return pad(seq, (0, target_len - seq.size(0)), value=0)

        # wt_seq.append(pad_seq(wt_info['seq'], max_len))
        wt_nodes.append(pad_node(wt_info['node'], max_len))

        # mut_seq.append(pad_seq(mut_info['seq'], max_len))
        mut_nodes.append(pad_node(mut_info['node'], max_len))

        # mask 用 wild-type 长度为主
        mask = torch.zeros(max_len, dtype=torch.bool)
        mask[:L_wt] = True
        masks.append(mask)

        if ddg_label is None:
            # 预测阶段：无真实标签，用0占位（后续不参与计算）
            ddgs.append(torch.tensor(0.0, dtype=torch.float32))
        else:
            # 训练/验证阶段：有真实标签，直接保留
            ddgs.append(torch.tensor(ddg_label, dtype=torch.float32))  # 显式指定dtype，避免类型不一致
        # ddgs.append(ddg_label)

    # stack 成 batch 维度
    wt_data = {
        'node': torch.stack(wt_nodes),  # [B, L_max, D]
        # 'seq': torch.stack(wt_seq)  # [B, L_max]
    }
    mut_data = {
        'node': torch.stack(mut_nodes),  # [B, L_max, D]
        # 'seq': torch.stack(mut_seq)  # [B, L_max]
    }
    masks = torch.stack(masks)  # [B, L_max]
    ddgs = torch.tensor(ddgs)  # [B]
    if contact_cutoff is not None:
        # 构建 Batch 对象
        wt_data['graph'] = Batch.from_data_list(wt_graphs)
        mut_data['graph'] = Batch.from_data_list(mut_graphs)
        # return {'graph': Batch.from_data_list(wt_graphs)}, {'graph': Batch.from_data_list(mut_graphs)}
    return wt_data, mut_data, masks, ddgs


def build_pyg_graph(node_feat, contact_map, cutoff=0.1):
    """构建PyG格式的图"""
    # 接触图阈值处理
    adj_matrix = (contact_map >= cutoff).float()
    # 提取非零边（PyG的edge_index为[2, E]格式）
    u, v = torch.nonzero(adj_matrix, as_tuple=True)
    edge_index = torch.stack([u, v], dim=0)  # 转换为PyG要求的格式

    # 边权重（接触概率）
    edge_weight = contact_map[u, v].float()

    # 创建PyG的Data对象
    data = Data(
        x=node_feat.float(),  # 节点特征 [N, D]
        edge_index=edge_index,  # 边索引 [2, E]
        edge_attr=edge_weight.unsqueeze(1)  # 边权重 [E, 1]（PyG通常用edge_attr存储）
    )
    return data


class CustomDataset(Dataset):
    def __init__(self, feature_dir, asymmetric=True, fold_dir=None):
        self.feature_dir = feature_dir
        self.pairs = []
        self.cache = {}  # 文件缓存减少I/O
        self.asymmetric = asymmetric  # 反对称开关
        fold_pdb = None
        if fold_dir is not None:
            fold_data = pd.read_csv(fold_dir)
            fold_pdb = fold_data['pdb_id'].tolist()
        # 遍历目录，组织文件对
        files = os.listdir(feature_dir)
        wild_dict = {}
        mut_dict = {}

        for file in files:
            if not file.endswith(".pt"):
                continue

            parts = file.split('_')
            pdb_id = parts[0]

            if fold_pdb is not None and pdb_id not in fold_pdb:
                continue

            mut_info = '_'.join(parts[1:]).replace('.pt', '')

            full_path = os.path.join(feature_dir, file)
            # 如果_0在文件名中，则将该文件删除
            if '_0' in file:
                print("删除文件: " + full_path)
                os.remove(full_path)
            if '_wt' in file:
                wild_dict[pdb_id] = full_path
            else:
                mut_dict.setdefault(pdb_id, []).append((mut_info, full_path))

        # 组织成 (wild_path, mutant_path) 对
        for pdb_id in mut_dict:
            if pdb_id not in wild_dict:
                continue  # 跳过没有野生型的突变
            wild_path = wild_dict[pdb_id]
            for mut_info, mut_path in mut_dict[pdb_id]:
                self.pairs.append((wild_path, mut_path))

    def __len__(self):
        return len(self.pairs) * 2 if self.asymmetric else len(self.pairs)

    def _load_data(self, path):
        """带缓存的数据加载"""
        if path not in self.cache:
            data = torch.load(path, map_location='cpu')
            # seq_is = data.get('seq', None)
            ddg_label = data.get('ddg', None)
            contact_map = data.get('contacts', None)
            if ddg_label is None:
                self.cache[path] = {
                    # 'seq': data['seq'] if seq_is is not None else "0",
                    # 'contact': data['contacts'].squeeze(0).detach(),
                    'feature': data['residue_features'].squeeze(0).detach()
                }
                if contact_map is not None:
                    self.cache[path]['contact'] = contact_map.squeeze(0).detach()
                return self.cache[path]
            else:
                # 统一数据提取逻辑
                self.cache[path] = {
                    # 'seq': data['seq'] if seq_is is not None else "0",
                    # 'contact': data['contacts'].squeeze(0).detach(),
                    'feature': data['residue_features'].squeeze(0).detach(),
                    'ddg': float(data['ddg'].item() if isinstance(data['ddg'], torch.Tensor) else data['ddg'])
                }
                if contact_map is not None:
                    self.cache[path]['contact'] = contact_map.squeeze(0).detach()
                return self.cache[path]
        else:
            return self.cache[path]

    def __getitem__(self, idx):
        # 根据是否启用反对称确定索引处理方式
        if self.asymmetric:
            pair_idx = idx % len(self.pairs)  # 确定原始对索引
            reverse = idx >= len(self.pairs)  # 是否反转顺序
        else:
            pair_idx = idx
            reverse = False
        wt_path, mut_path = self.pairs[pair_idx]
        wt_data = self._load_data(wt_path)
        mut_data = self._load_data(mut_path)

        wt_contact_map = wt_data.get('contact', None)
        mut_contact_map = mut_data.get('contact', None)
        mut_ddg = mut_data.get('ddg', None)
        # 构建数据字典
        wt_dict = {'node': wt_data['feature']
                   }
        if wt_contact_map is not None:
            wt_dict['contact'] = wt_contact_map
        mut_dict = {'node': mut_data['feature']
                    }
        if mut_contact_map is not None:
            mut_dict['contact'] = mut_contact_map

        # 反对称处理
        if not reverse:
            return wt_dict, mut_dict, mut_ddg
        else:
            return mut_dict, wt_dict, -mut_ddg

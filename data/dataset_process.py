#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project : PreDDG
@File    : dataset_process.py
@IDE     : PyCharm
@Author  : Henghui FAN
@Date    : 2025/3/6
"""
import os
import torch
import pandas as pd
from pathlib import Path
from esm import pretrained
from tqdm import tqdm
from datetime import datetime
"""
ref:
[1]. Ouyang-Zhang J, Diaz D, Klivans A, Kraehenbuehl P. 
Predicting a Protein' s Stability under a Million Mutations.
Advances in Neural Information Processing Systems 2023; 36 76229-76247.
"""


class DatasetProcess:
    def __init__(self, data_dir: str, train: bool, del_file: bool = False, device: str = 'cuda:2') -> None:
        """
        处理数据集
        :param data_dir: 数据集路径
        :param train: 是否为训练集
        :param del_file: 是否删除处理后的文件
        :param device: 设备
        """
        self.dataset_name = Path(data_dir).stem  # 数据集名称  .stem表示路径中文件名的部分（不包含文件扩展名）
        self.folder_dir = Path(data_dir).parent.parent  # 数据文件夹
        self.train = train  # 是否为训练集

        self.ism_folder = self.folder_dir / 'ism'  # ism文件夹
        self.ism_folder.mkdir(exist_ok=True)
        self.processed_csv_dir = os.path.join(self.folder_dir,
                                              f'mutations/{self.dataset_name}_processed.csv')  # 处理后的csv文件路径

        if os.path.exists(self.processed_csv_dir):
            if del_file:
                os.remove(self.processed_csv_dir)
        if os.path.exists(self.processed_csv_dir):
            self.df = pd.read_csv(self.processed_csv_dir, low_memory=False)
            self.wt_info = self.df.groupby('pdb_id').head(1)
            self.mut_info = self.df['mut_info']
            print(f"Loaded {len(self.wt_info)} peptides ({len(self.df)} mutations) from '{self.processed_csv_dir}'")
        else:
            print(f"'{self.processed_csv_dir}' does not exist, processing csv file...")
            self.df = None
            self.wt_info = None
            self.mut_info = None
            self.csv_dir = data_dir  # csv文件路径
            self._process_csv()  # 处理csv文件

        ism, alphabet = pretrained.esm2_t33_650M_UR50D()
        ckpt = torch.load('./data/ism/ism_t33_650M_uc30pdb/ism_t33_650M_uc30pdb.pth',
                          map_location='cpu')
        ism.load_state_dict(ckpt)
        self.batch_converter = alphabet.get_batch_converter()
        self.device = torch.device(device)
        self.ism = ism.to(self.device)
        self.ism.eval()

        processed_pdbs = set()  # 跟踪已处理的PDB
        wt_seq_cache = {}  # 缓存野生型序列
        # 记录开始时间
        start_time = datetime.now()
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc=f"Processing {self.dataset_name}"):  # 直接遍历DataFrame
            pdb_id = row['pdb_id']

            # 处理野生型（每个PDB仅一次）
            if pdb_id not in processed_pdbs:
                wt_seq = row['wt_seq']
                wt_feat = self.extract_feature_from_ism(row['wt_seq'], pdb_id=pdb_id)
                torch.save(wt_feat, self.ism_folder / f'{pdb_id}_wt.pt')
                processed_pdbs.add(pdb_id)
                wt_seq_cache[pdb_id] = wt_seq
            # 验证野生型一致性
            elif row['wt_seq'] != wt_seq_cache[pdb_id]:
                raise ValueError(f"野生型序列不一致! PDB: {pdb_id}")
            # 处理突变（每条记录都必须处理）
            mut_info = row['mut_info'].replace(':', '_')  # 统一处理特殊字符
            mut_feat = self.extract_feature_from_ism(
                row['mut_seq'],
                ddg=row['ddg'] if 'ddg' in row else None,
                pdb_id=pdb_id
            )
            torch.save(mut_feat, self.ism_folder / f"{pdb_id}_{mut_info}.pt")
        # 记录结束时间
        end_time = datetime.now()
        # 计算并打印总耗时
        total_time = end_time - start_time
        print(f"处理 {self.dataset_name} 完成，总耗时: {total_time}")
        # 释放ISM模型占用的内存
        del self.ism
        torch.cuda.empty_cache()

    def _process_csv(self):
        # 读取csv文件
        df = pd.read_csv(self.csv_dir, low_memory=False)
        # 如果列名中包含"dms_score"，且不包含"ddg",则将其重命名为"ddg"
        if 'dms_score' in df.columns and 'ddg' not in df.columns:
            df = df.rename(columns={'dms_score': 'ddg'})
        # 如果列名中包含"unique_id"，且不包含"pdb_id"，则将其重命名为"pdb_id"
        if 'unique_id' in df.columns and 'pdb_id' not in df.columns:
            df = df.rename(columns={'unique_id': 'pdb_id'})
        # 检查是否存在wt列和mut_info列
        if 'wt_seq' not in df.columns or 'mut_info' not in df.columns:
            raise ValueError("CSV文件中不存在'wt'列或'mut_info'列")
        # 检查是否存在mut_seq列
        if 'mut_seq' not in df.columns:
            df['mut_seq'] = df.apply(lambda row: generate_mut_seq(row), axis=1)

        if 'ddg' not in df.columns:
            df['ddg'] = df.index

        # 过滤无效信息
        df = df[(~df.mut_info.isna()) & (~df.wt_seq.isna()) & (~df.ddg.isna())]

        # 只保留多突变
        df = df[df['mut_info'].str.count(':') >= 1]

        detdf = df.groupby('pdb_id', as_index=False).head(1)
        n_seq_per_pdb = df.groupby('pdb_id').wt_seq.nunique()  # 计算每个 pdb_id 对应的唯一序列 (wt_seq) 的数量
        if not (n_seq_per_pdb == 1).all():  # 检查n_seq_per_pdb中是否所有的值都等于1
            pdb_with_one_seq = n_seq_per_pdb[n_seq_per_pdb == 1].index
            pdb_with_few_seq = n_seq_per_pdb[n_seq_per_pdb != 1].index
            detdf = detdf[detdf.pdb_id.isin(pdb_with_one_seq)]
            df = df[df.pdb_id.isin(pdb_with_one_seq)]
            print(f'WARNING: Found multiple wt_seq, Removing {pdb_with_few_seq}')

        self.df = df  # 突变对应的全部数据
        self.wt_info = detdf
        self.mut_info = df['mut_info']
        if self.train:
            self.df = self.hierarchical_under_sampling(self.df)
        self.df.to_csv(self.processed_csv_dir, index=False)
        # 9.打印处理结果
        print(f"Saved {len(self.wt_info)} peptides ({len(self.df)} mutations) to '{self.processed_csv_dir}'")

    @staticmethod
    def hierarchical_under_sampling(df):
        """
        分层下采样
        :param df: 数据集
        :return: 下采样后的数据集
        """
        balanced_dfs = []  # 存储每个PDB平衡后的数据

        # 按PDB分组处理
        for pdb, group in df.groupby('pdb_id'):
            # 分离稳定突变 (ddg < 0) 和不稳定突变 (ddg >= 0)
            stable = group[group['ddg'] < 0]
            unstable = group[group['ddg'] >= 0]

            n_stable = len(stable)
            n_unstable = len(unstable)

            unstable_sampled = unstable.sample(n=min(n_stable, n_unstable),
                                               random_state=42)

            # 合并当前PDB的平衡数据
            balanced_group = pd.concat([stable, unstable_sampled])
            balanced_dfs.append(balanced_group)

        # 合并所有PDB的结果
        return pd.concat(balanced_dfs, ignore_index=True)

    def extract_feature_from_ism(self, seqs, ddg=None, pdb_id=None):
        """
        从ism模型中提取特征
        :param seqs: 蛋白质序列
        :param ddg: 突变ddg
        :param pdb_id: 蛋白质ID
        :return: 特征字典
        """
        # 确保输入格式正确
        assert isinstance(seqs, str), "Input must be a single protein sequence string"

        data = [("protein", seqs)]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)

        with torch.no_grad():
            results = self.ism(batch_tokens, repr_layers=[33], return_contacts=True)
        token_rep = results["representations"][33][0, 1:-1]
        contacts = results["contacts"][0]

        return {
            "seq": seqs,  # 原始序列
            "contacts": contacts,  # 残基级联系矩阵 [L, L]
            "residue_features": token_rep,  # 残基级特征 [L, latent_dim]
            "ddg": float(ddg) if ddg is not None else None
        }


# 定义一个函数来生成mut_seq
def generate_mut_seq(row):
    wt_seq = row['wt_seq']
    # 检查mut_info是否为空
    if pd.isna(row['mut_info']):
        raise ValueError("mut_info列中存在空值")
    # 检查mut_info是否只包含一个突变
    if ':' not in row['mut_info']:
        raise ValueError("请用','分隔多个突变")
    # 检查mut_info中存在几个突变，根据":"分割
    mut_info_list = row['mut_info'].split(':')
    for mut_info in mut_info_list:
        # 假设mut_info格式为 "A123T"，表示将A替换为T，位置为123（注意：位置从1开始）
        pos = int(mut_info[1:-1]) - 1  # 转换为0-based索引
        # 检查位置是否有效
        if pos < 0 or pos >= len(wt_seq):
            raise ValueError(f"突变位置 {pos+1} 超出序列长度 {len(wt_seq)}")
        # 检查是否存在相同的突变
        if wt_seq[pos] == mut_info[-1]:
            raise ValueError(f"突变位置 {pos+1} 上的突变 {mut_info} 与WT序列相同")

    mut_seq = wt_seq
    for mut_info in mut_info_list:
        # 检查突变位置的氨基酸是否与WT序列相同
        pos = int(mut_info[1:-1]) - 1  # 转换为0-based索引
        if wt_seq[pos] != mut_info[0]:
            raise ValueError(f"突变位置 {pos+1} 上的突变 {mut_info} 与WT序列不同")
        mut_seq = mut_seq[:pos] + mut_info[-1] + mut_seq[pos+1:]
    return mut_seq

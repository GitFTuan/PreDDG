#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project : PreDDG
@File    : predict.py
@IDE     : PyCharm 
@Author  : Henghui FAN
@Date    : 2025/7/7
"""
import torch
import pandas as pd
import os
import warnings
from argparse import ArgumentParser
import pytorch_lightning as pl
from model import load_model
from data import DataMain
from torch_geometric.data import Batch


torch.set_float32_matmul_precision('medium')


def get_original_file_paths(test_name):
    """获取测试集对应的原始数据文件路径"""
    # 根据实际数据文件结构调整路径生成逻辑
    test_names = test_name.split(',')
    file_paths = []
    for name in test_names:
        name = name.strip()
        file_path = f'./data/dataset/{name}/mutations/{name}_processed.csv'

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"原始数据文件不存在: {file_path}")
        file_paths.append(file_path)
    return file_paths


def write_predictions_to_file(original_file, predictions, indexes):
    """
    按原始索引将预测结果匹配到原始文件
    :param original_file: 原始CSV文件路径
    :param predictions: 模型预测结果（list/numpy，顺序与batch一致）
    :param indexes: 每个预测结果对应的原始文件索引（list/numpy，从batch中获取）
    """
    # 1. 读取原始数据
    df = pd.read_csv(original_file)
    original_len = len(df)
    # 2. 基础校验：预测结果数量与索引数量必须一致
    if len(predictions) != len(indexes):
        warnings.warn(f"⚠️ 预测结果数量({len(predictions)})与索引数量({len(indexes)})不匹配！未匹配索引将填充空值")
    # 3. 初始化PreDDG列为空值
    df['PreDDG'] = pd.NA

    # 4. 用索引创建预测结果的DataFrame，确保与原始数据对齐
    pred_df = pd.DataFrame({
        'original_index': indexes,  # 原始文件的行索引
        'PreDDG': predictions       # 对应索引的预测结果
    })
    # 5. 按原始索引排序（确保顺序与原始文件完全一致）
    pred_df = pred_df.sort_values(by='original_index').reset_index(drop=True)
    # 6. 按原始索引填充预测结果
    for idx, row in pred_df.iterrows():
        orig_idx = row['original_index']
        pred_val = row['PreDDG']
        if (0 <= orig_idx < original_len) and df.loc[orig_idx, 'index'] == orig_idx:  # 确保索引在有效范围内
            df.loc[orig_idx, 'PreDDG'] = pred_val
        else:
            warnings.warn(f"⚠️ 索引{orig_idx}在原始数据中不存在，已跳过")

    # 7. 统计填充情况并输出警告
    filled_count = df['PreDDG'].notna().sum()
    empty_count = df['PreDDG'].isna().sum()
    if empty_count > 0:
        warnings.warn(f"⚠️ 共{empty_count}行未匹配到预测结果，已填充空值；成功填充{filled_count}行")

    # # 删除ddg列
    # if 'ddg' in df.columns:
    #     df.drop(columns=['ddg'], inplace=True)

    # 原始路径示例：./data/dataset/xxx/mutations/xxx_processed.csv
    mutations_dir = os.path.dirname(original_file)  # → ./data/dataset/xxx/mutations
    test_root_dir = os.path.dirname(mutations_dir)  # → ./data/dataset/xxx（xxx目录，predictions将创建于此）
    original_filename = os.path.basename(original_file)  # → xxx_processed.csv

    # 1. 创建目标目录：./data/dataset/xxx/predictions
    target_dir = os.path.join(test_root_dir, "predictions")
    os.makedirs(target_dir, exist_ok=True)  # 不存在则创建，避免路径错误

    # 2. 构建目标文件名：xxx_processed_with_prediction.csv
    target_filename = original_filename.replace(".csv", "_with_prediction.csv")

    # 3. 最终输出路径：./data/dataset/xxx/predictions/xxx_processed_with_prediction.csv
    output_file = os.path.join(target_dir, target_filename)

    df.to_csv(output_file, index=False)
    print(f"✅ 已按索引匹配写入预测结果：{output_file}")


def move_batch_to_device(batch, device):
    """
    处理DataMain生成的batch结构：(wt_data, mut_data, masks, ddgs)
    - wt_data/mut_data：字典，含'tensor'和'graph'（pyg Batch对象）
    - masks/ddgs：普通张量
    作用：将所有张量和graph对象移至指定设备
    """
    wt_data, mut_data, masks, ddgs, index = batch  # 解包batch元组

    # 子函数：处理单个数据字典（wt_data 或 mut_data）
    def move_data_dict_to_device(data_dict, dev):
        moved_dict = {}
        for key, value in data_dict.items():
            if isinstance(value, torch.Tensor):
                # 普通张量：直接移至设备
                moved_dict[key] = value.to(dev)
            elif isinstance(value, Batch):
                # pyg Batch对象：用to()方法移至设备（pyg 2.0+支持）
                moved_dict[key] = value.to(dev)
            else:
                # 其他类型（如列表）：直接保留
                moved_dict[key] = value
        return moved_dict

    # 1. 处理wt_data和mut_data字典
    wt_data_moved = move_data_dict_to_device(wt_data, device)
    mut_data_moved = move_data_dict_to_device(mut_data, device)
    # 2. 处理masks和ddgs张量
    masks_moved = masks.to(device)
    ddgs_moved = ddgs.to(device)
    index_moved = index.to(device)

    # 返回移至设备后的新batch
    return wt_data_moved, mut_data_moved, masks_moved, ddgs_moved, index_moved


def main():
    # 确保args在全局范围内可用
    global args

    # 设置随机种子保证可复现性
    pl.seed_everything(args.seed, workers=True)

    # 初始化数据模块并获取测试数据加载器
    data_module = DataMain(**vars(args))
    data_module.setup('test')  # 确保测试数据正确设置
    test_dataloaders = data_module.test_dataloader()
    # 2. 获取原始文件路径（与测试集dataloader一一对应）
    original_file_paths = get_original_file_paths(args.test_name)
    # 加载模型
    model, _ = load_model(args)  # 假设load_model返回模型和可能的其他信息
    model.eval()  # 确保模型处于评估模式
    # 初始化设备
    device = torch.device(args.device)
    model = model.to(device)

    # 对每个测试数据加载器进行预测
    for dataloader_idx, test_loader in enumerate(test_dataloaders):
        batch_predictions = []
        indexes = []
        for batch in test_loader:
            batch_moved = move_batch_to_device(batch, device)
            wt_data_moved, mut_data_moved, _, _, index_moved = batch_moved  # 解包移至设备后的batch
            with torch.no_grad():
                outputs, _ = model(wt_data_moved, mut_data_moved)
                batch_predictions.extend(outputs.cpu().numpy())
                indexes.extend(index_moved.cpu().numpy())

        # 将当前测试集的预测结果写入对应原始文件
        current_original_file = original_file_paths[dataloader_idx]
        write_predictions_to_file(current_original_file, batch_predictions, indexes)



if __name__ == '__main__':
    parser = ArgumentParser()
    # Experiment information
    experiment_group = parser.add_argument_group("Experiment args")
    experiment_group.add_argument('--experiment_log_dir', default='./log/tensorboardX/u_net', type=str)
    experiment_group.add_argument('--experiment_name', default='predict', type=str)
    experiment_group.add_argument('--seed', default=1234, type=int)

    # Model parameters
    model_group = parser.add_argument_group("Model args")
    model_group.add_argument('--model_name', default='PreDDG', type=str)
    model_group.add_argument('--load_dir', default='./model/checkpoint/checkpoint.ckpt', type=str)
    model_group.add_argument('--device', default='cuda', type=str)
    model_group.add_argument('--load_best', default=False, type=bool)
    model_group.add_argument('--load_v_num', default=None, type=int)

    # Data parameters
    data_group = parser.add_argument_group("Data args")
    data_group.add_argument('--test_name', default='ptmul_nr_single', type=str)
    data_group.add_argument('--del_files', default=1, type=int,
                            help="1表示删除在后台保留的文件，0表示在后台保留本次处理的文件")

    args = parser.parse_args()

    main()

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
    # 2. 基础校验：预测结果数量与索引数量必须一致
    if len(predictions) != len(indexes):
        raise ValueError(f"预测结果数量({len(predictions)})与索引数量({len(indexes)})不匹配！")
    # 3. 关键：用索引创建预测结果的DataFrame，确保与原始数据对齐
    pred_df = pd.DataFrame({
        'original_index': indexes,  # 原始文件的行索引
        'preddg': predictions       # 对应索引的预测结果
    })
    # 4. 按原始索引排序（确保顺序与原始文件完全一致）
    pred_df = pred_df.sort_values(by='original_index').reset_index(drop=True)
    # 5. 校验：排序后的预测结果数量必须与原始数据行数一致
    if len(pred_df) != len(df):
        raise ValueError(
            f"排序后预测结果数量({len(pred_df)})与原始数据行数({len(df)})不匹配！"
            f"可能是测试集加载不完整或索引重复。"
        )
    # 6. 将预测结果写入原始数据的新列
    df['preddg'] = pred_df['preddg'].values  # 按排序后的结果赋值
    # 删除ddg列
    if 'ddg' in df.columns:
        df.drop(columns=['ddg'], inplace=True)
    # 7. 保存结果（不覆盖原文件）
    output_file = original_file.replace('.csv', '_with_prediction.csv')
    df.to_csv(output_file, index=False)
    print(f"✅ 已按索引匹配写入预测结果：{output_file}")


def move_batch_to_device(batch, device):
    """
    处理DataMain生成的batch结构：(wt_data, mut_data, masks, ddgs)
    - wt_data/mut_data：字典，含'tensor'和'graph'（pyg Batch对象）
    - masks/ddgs：普通张量
    作用：将所有张量和graph对象移至指定设备
    """
    wt_data, mut_data, masks, index = batch  # 解包batch元组

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
    index_moved = index.to(device)

    # 返回移至设备后的新batch
    return wt_data_moved, mut_data_moved, masks_moved, index_moved


def main():
    # 确保args在全局范围内可用
    global args

    # 设置随机种子保证可复现性
    pl.seed_everything(args.seed, workers=True)

    # 初始化数据模块并获取测试数据加载器
    data_module = DataMain(**vars(args))
    data_module.setup('test')  # 确保测试数据正确设置
    test_dataloaders = data_module.test_dataloader()
    # 2. 关键：获取原始文件路径（与测试集dataloader一一对应）
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
            wt_data_moved, mut_data_moved, _, index_moved = batch_moved  # 解包移至设备后的batch
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
    data_group.add_argument('--test_name', default='M28', type=str)

    args = parser.parse_args()

    main()

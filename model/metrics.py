#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project : PreDDG
@File    : metrics.py
@IDE     : PyCharm
@Author  : Henghui FAN
@Date    : 2025/3/6
"""
import torch
from torch import nn
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score, matthews_corrcoef


class GetMetrics(nn.Module):
    def __init__(self):
        super(GetMetrics, self).__init__()

    @staticmethod
    def compute_cls_metrics(pred, labels):
        """
        评估模型正确预测DDG符号的性能
        :param pred: 预测值
        :param labels: 标签值
        :return: 分类指标
        """
        pred = np.asarray(pred)
        labels = np.asarray(labels)

        # 明确符号定义
        stability_sign = (labels < 0)  # True=稳定突变
        pred_sign = (pred < 0)   # True=预测稳定

        # 计算AUROC
        auroc = roc_auc_score(stability_sign, -pred)  # 注意负号
        # 计算马修斯相关系数 the Matthews correlation coefficient (MCC)
        mcc = matthews_corrcoef(stability_sign, pred_sign)
        return {"auroc": auroc, "mcc": mcc}

    @staticmethod
    def compute_reg_metrics(pred, labels):
        """
        评估模型预测DDG值与实验DDG值之间的相关性
        :param pred: 预测值
        :param labels: 标签值
        :return: 回归指标
        """
        # 计算斯皮尔曼相关系数
        spear = stats.spearmanr(pred, labels)[0]

        return {"spear": spear}

    def forward(self,
                ddg_pred=None,
                ddg_labels=None):
        if ddg_pred is None or ddg_labels is None:
            return {}
        if isinstance(ddg_pred, torch.Tensor):
            ddg_pred = ddg_pred.detach().cpu().numpy()
        if isinstance(ddg_labels, torch.Tensor):
            ddg_labels = ddg_labels.cpu().numpy()

        metrics = {
            **self.compute_cls_metrics(ddg_pred, ddg_labels),
            **self.compute_reg_metrics(ddg_pred, ddg_labels)
        }
        
        return metrics

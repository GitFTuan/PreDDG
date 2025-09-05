#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project : PreDDG
@File    : model_main.py
@IDE     : PyCharm
@Author  : Henghui FAN
@Date    : 2025/3/6
"""
import importlib
import inspect
import traceback

import pytorch_lightning as pl
import torch
import torch.optim.lr_scheduler as lrs

from torch import nn
from .metrics import GetMetrics


class ModelMain(pl.LightningModule):
    def __init__(self, **args):
        super().__init__()
        self.loss_function = nn.HuberLoss()
        self.model = None

        self.save_hyperparameters()  # 保存__init__方法中传入的参数，并将其存储在self.hparams中
        self.load_model()
        self.get_metrics = GetMetrics()

        # 为验证和测试阶段分别初始化存储列表
        self.val_ddg_preds = []
        self.val_ddg_labels = []
        self.test_ddg_preds = []
        self.test_ddg_labels = []

    def forward(self, wild_data, mut_data, mask=None):
        return self.model(wild_data, mut_data, mask)

    def training_step(self, batch, prefix=None):
        return self.shared_step(batch, 'train')

    def validation_step(self, batch, dataloader_idx=0):
        return self.shared_step(batch, 'val')

    def on_validation_epoch_end(self):
        return self.shared_end_step(mode='val')

    def test_step(self, batch, dataloader_idx=0):
        return self.shared_step(batch, 'test')

    def on_test_epoch_end(self):
        return self.shared_end_step(mode='test')

    def shared_step(self, batch, prefix):
        # 模型预测
        wild_data, mut_data, mask, ddg_labels = batch
        ddg_pred, cls = self(wild_data, mut_data, mask)  # [B]
        ddg_pred = ddg_pred.squeeze(-1)  # [B]
        if prefix == 'train' or prefix == 'val':
            loss = self.loss_function(
                ddg_pred=ddg_pred,
                ddg_true=ddg_labels
            )  # type: ignore

            self.log(f'{prefix}_loss', loss,
                     on_step=True, on_epoch=True,
                     prog_bar=True, sync_dist=True,
                     batch_size=self.hparams.batch_size)
            if prefix == 'train':
                return loss
            else:
                self.val_ddg_preds.append(ddg_pred.detach().cpu())
                self.val_ddg_labels.append(ddg_labels.detach().cpu())
        else:
            self.test_ddg_preds.append(ddg_pred.detach().cpu())
            self.test_ddg_labels.append(ddg_labels.detach().cpu())

    def shared_end_step(self, mode='val'):
        # 根据模式选择正确的存储列表
        if mode == 'val':
            preds_list = self.val_ddg_preds
            labels_list = self.val_ddg_labels
        else:  # test
            preds_list = self.test_ddg_preds
            labels_list = self.test_ddg_labels

        # 检查是否有数据
        if not preds_list:
            self.print(f"警告: {mode}阶段没有收集到任何数据!")
            return
        # 聚合所有批次的预测值和真实标签
        all_ddg_preds = torch.cat(preds_list, dim=0)
        all_ddg_labels = torch.cat(labels_list, dim=0)
        # 计算最终的评价指标
        metrics = self.get_metrics(
            ddg_pred=all_ddg_preds,
            ddg_labels=all_ddg_labels)
        # 记录最终的评价指标，添加 sync_dist=True
        for name, metric in metrics.items():
            self.log(f'{mode}_final_{name}', metric,
                     on_step=False, on_epoch=True,
                     sync_dist=True, reduce_fx='mean')

        if mode == 'val':
            self.val_ddg_preds = []
            self.val_ddg_labels = []
        else:
            self.test_ddg_preds = []
            self.test_ddg_labels = []

    def configure_optimizers(self):
        weight_decay = getattr(self.hparams, 'weight_decay', 0)
        lr = getattr(self.hparams, 'lr', 1e-3)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        scheduler_type = getattr(self.hparams, 'lr_scheduler', None)
        if scheduler_type == 'step':
            scheduler = lrs.StepLR(optimizer,
                                   step_size=self.hparams.lr_decay_steps,
                                   gamma=self.hparams.lr_decay_rate)
        elif scheduler_type == 'cosine':
            scheduler = lrs.CosineAnnealingLR(optimizer,
                                              T_max=self.hparams.lr_decay_steps,
                                              eta_min=self.hparams.lr_decay_min_lr)
        else:
            raise ValueError(f'Invalid lr_scheduler type: {scheduler_type}!')
        return [optimizer], [scheduler]

    def load_model(self):
        name = self.hparams.model_name
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        try:
            model = getattr(importlib.import_module('.' + name, package=__package__), camel_name)
        except Exception:
            error_msg = traceback.format_exc()
            raise ValueError(f"Error loading model `{name}` -> `{camel_name}`:\n{error_msg}")
        self.model = self.model_instantiation(model)

    def model_instantiation(self, model, **other_args):
        """ 使用 self.hparams 字典中的参数来实例化一个 Model，并允许传入额外的参数来覆盖默认值.
        :param model: 模型类
        :param other_args: 其他参数
        """
        class_args = list(inspect.signature(model.__init__).parameters.keys())
        args = {arg: getattr(self.hparams, arg) for arg in class_args if arg in self.hparams}
        args.update(other_args)
        return model(**args)

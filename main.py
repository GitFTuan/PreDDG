#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project : PreDDG
@File    : main.py
@IDE     : PyCharm
@Author  : Henghui FAN
@Date    : 2025/3/6
"""
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import torch
from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import TensorBoardLogger
from model import load_model
from data import DataMain


torch.set_float32_matmul_precision('medium')


def load_callbacks(args):
    callbacks = []
    try:
        callbacks.append(plc.ModelSummary(max_depth=-1))
        callbacks.append(plc.ModelCheckpoint(
            monitor=None,
            save_top_k=1,
            mode='max',
            save_last=True,
            save_weights_only=True,
        ))
        if args.lr_scheduler:
            callbacks.append(plc.LearningRateMonitor(logging_interval='epoch'))

    except Exception as e:
        print(f"创建回调函数时出错: {e}")
    return callbacks


def get_best_checkpoint_path(trainer):
    """获取保存的最佳模型的路径"""
    if hasattr(trainer.checkpoint_callback, 'best_model_path'):
        best_ckpt_path = trainer.checkpoint_callback.best_model_path
        if best_ckpt_path and os.path.isfile(best_ckpt_path):
            print(f"发现最佳模型检查点: {best_ckpt_path}")
            return best_ckpt_path
    print("未发现最佳模型检查点，使用最后一个检查点。")
    return trainer.checkpoint_callback.last_model_path


def main():
    global args  # 使 args 在全局范围内可用
    pl.seed_everything(args.seed, workers=True)

    # 加载模型和数据
    model, ckpt_path = load_model(args)
    data_module = DataMain(**vars(args))

    # 创建日志记录器
    logger = TensorBoardLogger(
        save_dir=args.experiment_log_dir,
        name=args.experiment_name
    )

    # 创建回调函数
    callbacks = load_callbacks(args)

    # 创建训练器
    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.device_type,
        devices=args.devices,
        num_nodes=args.num_nodes,
        strategy=args.strategy,
        callbacks=callbacks,
        logger=logger,
        enable_progress_bar=True,
        precision='16-mixed',
        gradient_clip_val=5.,
        num_sanity_val_steps=0,
        enable_model_summary=False
    )

    if args.eval:  # 评估模式
        if ckpt_path is None:
            print("评估模式已启用，但未提供检查点路径！")
            return
        print(f"从检查点评估模型: {ckpt_path}")
        trainer.validate(model, data_module.val_dataloader())
        for test_loader in data_module.test_dataloader():
            trainer.test(model, test_loader)
    else:  # 训练模式
        if ckpt_path is None or not args.restart:
            trainer.fit(model, data_module)
        else:
            trainer.fit(model, data_module, ckpt_path=ckpt_path)

        target_ckpt_path = get_best_checkpoint_path(trainer)

        trainer.validate(model, data_module.val_dataloader(), ckpt_path=target_ckpt_path)
        for test_loader in data_module.test_dataloader():
            trainer.test(model, test_loader, ckpt_path=target_ckpt_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    # experiment information
    experiment_group = parser.add_argument_group("Experiment args")
    experiment_group.add_argument('--experiment_log_dir', default='./log/tensorboardX/u_net', type=str)
    experiment_group.add_argument('--experiment_name', default='DT', type=str)
    experiment_group.add_argument('--seed', default=1234, type=int)
    experiment_group.add_argument('--hp_op',
                                  action='store_true',
                                  # default=False,
                                  help='Enable hyperparameter optimization')
    experiment_group.add_argument('--n_trials', default=10, type=int,
                                  help='Number of trials for hyperparameter optimization')

    # Model Parameters
    model_group = parser.add_argument_group("Model args")
    model_group.add_argument('--model_name', default='PreDDG', type=str)

    # Basic Training Control
    trainer_group = parser.add_argument_group("Trainer args")
    trainer_group.add_argument('--batch_size', default=128, type=int)
    trainer_group.add_argument('--device_type', default='cuda', choices=["gpu", "cpu", "cuda"], type=str)
    trainer_group.add_argument('--devices', default=[0], type=list)
    trainer_group.add_argument('--num_nodes', default=1, type=int)
    trainer_group.add_argument('--strategy', default='auto', type=str)
    trainer_group.add_argument('--max_epochs', default=5, type=int)

    # LR Scheduler
    lr_group = parser.add_argument_group("LearningRateMonitor")
    lr_group.add_argument('--lr', default=1e-3, type=float)
    lr_group.add_argument('--lr_scheduler', default='cosine', choices=['step', 'cosine'], type=str)
    lr_group.add_argument('--lr_decay_steps', default=200, type=int)
    lr_group.add_argument('--lr_decay_rate', default=2, type=float)
    lr_group.add_argument('--lr_decay_min_lr', default=1e-5, type=float)


    # Optimizer
    optimizer_group = parser.add_argument_group("Optimizer args")
    optimizer_group.add_argument('--optimizer', default='adam', choices=['adam', 'sgd'], type=str)
    optimizer_group.add_argument('--weight_decay', default=1e-4, type=float)

    # Restart Control
    re_control = parser.add_argument_group("load models' args")
    re_control.add_argument('--restart', action='store_true')
    re_control.add_argument('--load_best', default=True, type=bool)
    re_control.add_argument('--load_dir', default=None, type=str)
    re_control.add_argument('--load_v_num', default=None, type=int)
    re_control.add_argument('--eval', default=False, help='Enable evaluation mode')

    args = parser.parse_args()

    main()

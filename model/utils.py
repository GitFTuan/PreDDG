#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project : PreDDG
@File    : utils.py
@IDE     : PyCharm
@Author  : Henghui FAN
@Date    : 2025/3/6
"""
import os
from pathlib import Path
from .model_main import ModelMain


def load_model(args):
    load_path = load_model_path(root=args.load_dir,
                                log_dir=args.experiment_log_dir,
                                experiment_name=args.experiment_name,
                                v_num=args.load_v_num,
                                best=args.load_best)
    if load_path is None:
        model = ModelMain(**vars(args))
        ckpt_path = None
    else:
        model = ModelMain.load_from_checkpoint(load_path, **vars(args), map_location='cpu', strict=False)
        ckpt_path = load_path
        print(f"Load model from {ckpt_path}")
    return model, ckpt_path


def load_model_path(root=None, log_dir=None, experiment_name=None, v_num=None, best=False):
    """

    """

    def sort_by_epoch(path):
        name = path.stem
        epoch = int(name.split('=')[2])
        return epoch

    def generate_root():
        if root is not None:
            return root
        elif experiment_name is not None and v_num is not None:
            return str(Path(log_dir, experiment_name, f'version_{v_num}', 'checkpoints'))

    if root is None and (experiment_name is None or v_num is None):
        return None

    root = generate_root()
    if Path(root).is_file():
        return root
    if best:
        files = [i for i in list(Path(root).iterdir()) if i.stem.startswith('best')]
        files.sort(key=sort_by_epoch, reverse=True)
        res = str(files[0])
    else:
        res = str(Path(root) / 'last.ckpt')
        if os.path.exists(res):
            return res
        else:
            files = [i for i in list(Path(root).iterdir()) if i.stem.startswith('epoch')]
            files.sort(key=sort_by_epoch, reverse=True)
            res = str(files[0])
    return res

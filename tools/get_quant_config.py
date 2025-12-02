import sys
import os.path as osp

# 1. 路径修复
project_root = osp.dirname(osp.dirname(osp.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import os
import sys
import torch
from mmcv import Config, DictAction
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mtl.data.build import build_datasets

# 1. 路径修复
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 显式导入模型
try:
    import models.multi.multitask_learner
    from quant.lsq_plus import LinearLSQ, Conv2dLSQ
except ImportError as e:
    pass

def parse_args():
    parser = argparse.ArgumentParser(description='Get Quantization Bit-width Configuration')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('checkpoint', help='FP32 pretrained checkpoint file path')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    # 修复了这里：添加了这行关键代码
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    
    print(f"Building model from {args.config}...")
    model = build_detector(cfg.model)
    
    print(f"Loading FP32 weights from {args.checkpoint}...")
    # 必须加载 FP32 权重，因为 TACQ 是根据 FP32 权重的数值分布来决定位宽的
    load_checkpoint(model, args.checkpoint, map_location='cpu', strict=False)
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    print("Re-running TACQ calibration to recover bit configuration...")
    
    # 1. 准备校准数据
    datasets = build_datasets(cfg.data)
    dataset_keys = list(datasets.keys())
    calib_dataset = datasets[dataset_keys[0]]
    
    # 自动推断任务类型
    target_task = 'cls'
    if 'dior' in dataset_keys[0].lower(): target_task = 'det'
    elif 'potsdam' in dataset_keys[0].lower(): target_task = 'seg'
    
    # 修复 GroupSampler
    if not hasattr(calib_dataset, 'flag'):
        import numpy as np
        calib_dataset.flag = np.zeros(len(calib_dataset), dtype=np.int64)
        
    from mmdet.datasets import build_dataloader
    calib_loader = build_dataloader(
        calib_dataset,
        samples_per_gpu=2,
        workers_per_gpu=0,
        dist=False,
        shuffle=True
    )
    
    # 2. 再次运行混合精度初始化 (不会训练，只为产生结构)
    if hasattr(model, 'apply_mixed_precision_quantization'):
        model.apply_mixed_precision_quantization(
            data_loader=calib_loader, 
            num_tasks=3, 
            ratio_8bit=0.5,  # 必须与你训练时设置的比例一致！
            default_task=target_task
        )
    elif hasattr(model, 'module'):
        model.module.apply_mixed_precision_quantization(
            data_loader=calib_loader, 
            num_tasks=3, 
            ratio_8bit=0.5,
            default_task=target_task
        )
    
    # 3. 打印结果
    print("\n" + "="*60)
    print("      Layer-wise Quantization Bit-width      ")
    print("="*60)
    print(f"{'Layer Name':<50} | {'Bits':<5}")
    print("-" * 60)
    
    count_8bit = 0
    count_4bit = 0
    
    # 遍历现在模型中的所有层
    for name, module in model.named_modules():
        if isinstance(module, (LinearLSQ, Conv2dLSQ)):
            # 读取层的位宽属性
            nbits = getattr(module, 'nbits_w', 4) 
            print(f"{name:<50} | {nbits:<5}")
            
            if nbits == 8: count_8bit += 1
            elif nbits == 4: count_4bit += 1

    print("-" * 60)
    print(f"Total Quantized Layers: {count_8bit + count_4bit}")
    print(f"8-bit Layers: {count_8bit}")
    print(f"4-bit Layers: {count_4bit}")
    print("="*60 + "\n")

if __name__ == '__main__':
    main()
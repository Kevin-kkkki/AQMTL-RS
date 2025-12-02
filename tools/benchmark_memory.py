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
import numpy as np
from mmcv import Config, DictAction
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mtl.data.build import build_datasets

# 1. 路径修复
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    import models.multi.multitask_learner
    from quant.lsq_plus import LinearLSQ, Conv2dLSQ
except ImportError as e:
    pass

def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark Model Memory Footprint (Theoretical)')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('checkpoint', help='FP32 pretrained checkpoint file path')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    args = parser.parse_args()
    return args

def get_model_size_mb(model):
    """
    计算模型的理论内存大小 (MB)
    逻辑：
    - 如果是量化层 (LinearLSQ/Conv2dLSQ) 且参数是 'weight' -> 使用 layer.nbits_w 计算
    - 其他所有参数 -> 使用 32-bit (FP32) 计算
    """
    total_bits = 0
    
    for name, param in model.named_parameters():
        # 1. 获取该参数所属的 Module
        # name 格式通常为: "backbone.layers.0.weight"
        # parent_name: "backbone.layers.0"
        # attr_name: "weight"
        if '.' in name:
            parent_name, attr_name = name.rsplit('.', 1)
            try:
                # 递归查找 module
                mod = model
                for part in parent_name.split('.'):
                    mod = getattr(mod, part)
            except AttributeError:
                mod = None
        else:
            mod = model
            attr_name = name

        # 2. 决定位宽
        bits = 32 # 默认为 FP32
        
        if mod is not None and isinstance(mod, (LinearLSQ, Conv2dLSQ)):
            # 只有量化层的 'weight' 享受低比特，bias 通常保持高精度
            if attr_name == 'weight':
                bits = getattr(mod, 'nbits_w', 4) # 获取该层的量化位宽
        
        # 3. 累加容量
        # param.numel(): 参数元素个数
        total_bits += param.numel() * bits

    # 转换为 MB (1 Byte = 8 bits, 1 MB = 1024*1024 Bytes)
    total_mb = total_bits / 8 / (1024 * 1024)
    return total_mb

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    
    print(f"Building model from {args.config}...")
    model = build_detector(cfg.model)
    
    # ---------------------------------------------------------
    # 1. 计算原始 FP32 大小
    # ---------------------------------------------------------
    fp32_size = get_model_size_mb(model)
    print(f"\n[Baseline] Original FP32 Model Size: {fp32_size:.2f} MB")

    print(f"Loading FP32 weights from {args.checkpoint} for calibration...")
    load_checkpoint(model, args.checkpoint, map_location='cpu', strict=False)
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    # ---------------------------------------------------------
    # 2. 运行 TACQ 混合精度分配
    # ---------------------------------------------------------
    print("\nRunning TACQ strategy to determine mixed-precision config...")
    
    datasets = build_datasets(cfg.data)
    calib_dataset = list(datasets.values())[0]
    
    target_task = 'cls'
    if 'dior' in str(type(calib_dataset)).lower(): target_task = 'det'
    elif 'potsdam' in str(type(calib_dataset)).lower(): target_task = 'seg'
    
    if not hasattr(calib_dataset, 'flag'):
        calib_dataset.flag = np.zeros(len(calib_dataset), dtype=np.int64)
        
    from mmdet.datasets import build_dataloader
    calib_loader = build_dataloader(
        calib_dataset, samples_per_gpu=2, workers_per_gpu=0, dist=False, shuffle=True
    )
    
    if hasattr(model, 'apply_mixed_precision_quantization'):
        # 确保这里的 ratio_8bit 和你实验时一致
        model.apply_mixed_precision_quantization(
            data_loader=calib_loader, num_tasks=3, ratio_8bit=0.5, default_task=target_task
        )
    
    # ---------------------------------------------------------
    # 3. 计算量化后大小
    # ---------------------------------------------------------
    quant_size = get_model_size_mb(model)
    
    # ---------------------------------------------------------
    # 4. 输出对比报告
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("      Model Memory Footprint Comparison      ")
    print("="*50)
    print(f"Original (FP32):      {fp32_size:.2f} MB")
    print(f"Quantized (Mixed):    {quant_size:.2f} MB")
    print("-" * 50)
    print(f"Reduction:            {fp32_size - quant_size:.2f} MB")
    print(f"Compression Ratio:    {fp32_size / quant_size:.2f}x")
    print(f"Space Savings:        {(1 - quant_size / fp32_size) * 100:.2f}%")
    print("="*50 + "\n")

if __name__ == '__main__':
    main()
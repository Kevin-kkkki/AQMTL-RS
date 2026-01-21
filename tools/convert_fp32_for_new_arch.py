import sys
import os.path as osp

# 1. 路径修复
project_root = osp.dirname(osp.dirname(osp.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import sys
import os

def convert_checkpoint(src_path, dst_path):
    print(f"Loading weights from {src_path}...")
    ckpt = torch.load(src_path, map_location='cpu')
    
    if 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt
        
    new_state_dict = state_dict.copy()
    
    # 查找 shared_encoder 的权重
    shared_keys = [k for k in state_dict.keys() if 'shared_encoder' in k]
    print(f"Found {len(shared_keys)} shared_encoder params.")
    
    converted_count = 0
    for key in shared_keys:
        # 构造新 key：将 shared_encoder 映射到 bbox_head.transformer.encoder
        # 原名: shared_encoder.layers.0...
        # 新名: bbox_head.transformer.encoder.layers.0...
        
        new_key = key.replace('shared_encoder', 'bbox_head.transformer.encoder')
        
        # 复制权重
        new_state_dict[new_key] = state_dict[key].clone()
        converted_count += 1
        
    # 保存结果
    if 'state_dict' in ckpt:
        ckpt['state_dict'] = new_state_dict
    else:
        ckpt = new_state_dict
        
    torch.save(ckpt, dst_path)
    print(f"Success! Converted {converted_count} params.")
    print(f"New checkpoint saved to: {dst_path}")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python tools/convert_fp32_for_new_arch.py <src_path> <dst_path>")
        exit(1)
        
    src = sys.argv[1]
    dst = sys.argv[2]
    convert_checkpoint(src, dst)
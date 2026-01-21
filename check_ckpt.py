import torch
import sys

# 用法: python check_ckpt.py work_dirs/stage2_qat_mixed/latest.pth
ckpt_path = sys.argv[1]
print(f"Checking {ckpt_path}...")
state_dict = torch.load(ckpt_path, map_location='cpu')

if 'state_dict' in state_dict:
    state_dict = state_dict['state_dict']

has_alpha_high = any('alpha_high' in k for k in state_dict.keys())
has_alpha = any('alpha' in k and 'high' not in k and 'low' not in k for k in state_dict.keys())

print(f"Contains 'alpha_high' (Mixed Precision)? : {has_alpha_high}")
print(f"Contains 'alpha' (Standard Quant)?      : {has_alpha}")

if has_alpha_high:
    print("✅ 这是 Phase 2 混合精度权重。")
elif has_alpha:
    print("⚠️ 这是普通量化权重 (Uniform)。")
else:
    print("❌ 这是 FP32 权重 (未量化)。加载到量化模型会导致崩塌！")
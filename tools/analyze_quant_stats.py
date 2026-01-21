import torch
import argparse
import os
import os.path as osp

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze Fine-Grained Mask Content Directly')
    parser.add_argument('mask_path', help='Path to the fine_grained_mask.pth file')
    return parser.parse_args()

def main():
    args = parse_args()
    
    if not osp.exists(args.mask_path):
        print(f"Error: Mask file not found at {args.mask_path}")
        return

    print(f"Loading mask from {args.mask_path} ...")
    try:
        mask_dict = torch.load(args.mask_path, map_location='cpu')
    except Exception as e:
        print(f"Error loading mask: {e}")
        return

    print(f"Analyzing mask content...")

    # 初始化统计变量
    total_params_in_mask = 0
    params_8bit = 0
    params_4bit = 0
    layer_count = 0

    # 遍历 Mask 字典
    # Mask 字典结构: { 'layer_name': tensor(0 or 1) }
    for layer_name, mask_tensor in mask_dict.items():
        layer_count += 1
        num_elements = mask_tensor.numel()
        
        # 统计 1 (8-bit) 和 0 (4-bit)
        # 注意：mask 是 float 还是 bool 取决于生成时的代码，这里通用处理
        n_8 = (mask_tensor == 1).sum().item()
        n_4 = num_elements - n_8
        
        total_params_in_mask += num_elements
        params_8bit += n_8
        params_4bit += n_4

    # 计算空间占用 (单位: MB)
    # 理论对比：这些参数如果是 FP32 (4 Bytes) vs 混合精度
    size_fp32 = total_params_in_mask * 4 / (1024 * 1024)
    size_quant = (params_8bit * 1 + params_4bit * 0.5) / (1024 * 1024)
    size_saved = size_fp32 - size_quant
    
    # 避免除零错误
    ratio_8bit = (params_8bit / total_params_in_mask * 100) if total_params_in_mask > 0 else 0
    ratio_4bit = (params_4bit / total_params_in_mask * 100) if total_params_in_mask > 0 else 0
    compression_ratio = size_fp32 / size_quant if size_quant > 0 else 0

    # 打印结果
    print("\n" + "="*60)
    print("           Fine-Grained Mask Content Analysis           ")
    print("="*60)
    print(f"Total Layers in Mask:   {layer_count}")
    print(f"Total Params in Mask:   {total_params_in_mask:,}")
    print("-" * 60)
    print(f"8-bit Params (Mask=1):  {int(params_8bit):<15,} ({ratio_8bit:.2f}%)")
    print(f"4-bit Params (Mask=0):  {int(params_4bit):<15,} ({ratio_4bit:.2f}%)")
    print("-" * 60)
    print(f"Size if FP32:           {size_fp32:.2f} MB")
    print(f"Size Mixed-Precision:   {size_quant:.2f} MB")
    print(f"Memory Saved:           {size_saved:.2f} MB")
    print(f"Compression Ratio:      {compression_ratio:.2f}x (on masked layers)")
    print("="*60 + "\n")
    
  

if __name__ == '__main__':
    main()
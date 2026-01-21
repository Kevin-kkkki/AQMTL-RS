import torch
import os
import argparse
import sys

def inspect_mask(mask_path):
    if not os.path.exists(mask_path):
        print(f"错误: 找不到文件 {mask_path}")
        return

    print(f"正在加载 Mask 文件: {mask_path} ...")
    try:
        mask_dict = torch.load(mask_path, map_location='cpu')
    except Exception as e:
        print(f"加载失败: {e}")
        return

    print(f"成功加载，共包含 {len(mask_dict)} 个层的 Mask 信息。\n")

    total_params = 0
    total_8bit = 0
    total_4bit = 0

    print(f"{'Layer Name':<60} | {'Shape':<20} | {'8-bit %':<10} | {'4-bit %':<10}")
    print("-" * 110)

    # 按照 Mask 字典中的顺序遍历（或者你可以按名字排序）
    for name in sorted(mask_dict.keys()):
        mask = mask_dict[name]
        
        # 统计数量
        n_params = mask.numel()
        n_8bit = mask.sum().item()
        n_4bit = n_params - n_8bit
        
        # 累加全局统计
        total_params += n_params
        total_8bit += n_8bit
        total_4bit += n_4bit
        
        # 计算比例
        ratio_8bit = n_8bit / n_params * 100
        ratio_4bit = n_4bit / n_params * 100
        
        shape_str = str(list(mask.shape))
        
        # 打印单层信息
        # 只有当该层包含 8-bit 权重时高亮显示，或者打印所有层
        print(f"{name:<60} | {shape_str:<20} | {ratio_8bit:>6.2f}%    | {ratio_4bit:>6.2f}%")

    print("-" * 110)
    print("\n【全局统计 Summary】")
    print(f"总参数量 (Total Params): {total_params}")
    print(f"8-bit 参数 (High Precision): {int(total_8bit)} ({total_8bit/total_params*100:.2f}%)")
    print(f"4-bit 参数 (Low Precision):  {int(total_4bit)} ({total_4bit/total_params*100:.2f}%)")
    print("-" * 30)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inspect Fine-Grained Quantization Mask')
    # 默认路径设为你之前设定的保存路径
    parser.add_argument('--path', type=str, default='work_dirs/stage2_qat_mixed/fine_grained_mask.pth', 
                        help='Path to the fine_grained_mask.pth file')
    args = parser.parse_args()
    
    inspect_mask(args.path)
    
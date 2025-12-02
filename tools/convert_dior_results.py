import sys
import os.path as osp

# 1. 路径修复
project_root = osp.dirname(osp.dirname(osp.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import mmcv
import os
from mmcv import Config
from mmdet.datasets import build_dataset

def convert_results():
    # 1. 配置文件路径
    config_file = 'configs/multi/MTL_qat_slvlcls_swin-t-p4-w7_1x1_resisc&dior&potsdam.py'
    # 2. 推理结果路径
    pkl_file = 'work_dirs/quant_results_dior.pkl'
    # 3. 输出 JSON 路径前缀
    out_json_prefix = 'work_dirs/quant_results_dior' 

    print(f"Loading config from {config_file}...")
    cfg = Config.fromfile(config_file)
    
    # --- [核心修复] 正确加载 DIOR 子配置 ---
    if 'dior' in cfg.data:
        # 1. 获取子配置文件的路径 ('configs/_base_/det/dior.py')
        dior_config_path = cfg.data.dior.config
        print(f"Loading sub-config for DIOR from: {dior_config_path}")
        
        # 2. 加载真正的 DIOR 配置文件
        dior_cfg = Config.fromfile(dior_config_path)
        
        # 3. 提取 test 数据集配置
        dataset_cfg = dior_cfg.data.test
    else:
        # 如果是单任务配置 (fallback)
        dataset_cfg = cfg.data.test
    # -----------------------------------
        
    print("Building DIOR dataset...")
    dataset = build_dataset(dataset_cfg)
    
    print(f"Loading results from {pkl_file}...")
    results = mmcv.load(pkl_file)
    
    print(f"Converting to COCO JSON format...")
    # 这会生成 work_dirs/quant_results_dior.bbox.json
    dataset.format_results(results, jsonfile_prefix=out_json_prefix)
    
    # 重命名生成的文件
    expected_file = out_json_prefix + '.bbox.json'
    target_file = out_json_prefix + '.json'
    
    if os.path.exists(expected_file):
        os.rename(expected_file, target_file)
        print(f"Success! Result saved to: {target_file}")
    else:
        print(f"Warning: Expected output file {expected_file} not found. Check if format_results skipped generation.")

if __name__ == '__main__':
    convert_results()
import sys
import os.path as osp

# 1. 路径修复
project_root = osp.dirname(osp.dirname(osp.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# tools/analyze_dior.py
from tidecv import TIDE, datasets
import os
import json

def clean_gt_json(gt_path, clean_path):
    """
    读取 GT JSON，为每个标注生成伪造的 segmentation 数据，
    以防止 pycocotools 和 tidecv 报错。
    """
    print(f"Cleaning GT file: {gt_path} ...")
    with open(gt_path, 'r') as f:
        gt_data = json.load(f)
    
    fixed_count = 0
    for ann in gt_data['annotations']:
        # 强制将 iscrowd 设为 0，确保按多边形(Polygon)格式处理
        ann['iscrowd'] = 0
        
        # 获取 bbox: [x, y, w, h]
        bbox = ann.get('bbox', [0, 0, 10, 10])
        x, y, w, h = bbox
        
        # 生成一个矩形多边形: [x,y, x+w,y, x+w,y+h, x,y+h]
        # 这符合 COCO segmentation 的 [[x1,y1, x2,y2...]] 格式
        poly = [
            x, y,
            x + w, y,
            x + w, y + h,
            x, y + h
        ]
        
        # 覆盖原有的 segmentation
        ann['segmentation'] = [poly]
        fixed_count += 1
            
    print(f"Fixed segmentation for {fixed_count} annotations.")
    
    with open(clean_path, 'w') as f:
        json.dump(gt_data, f)
    print(f"Cleaned GT saved to: {clean_path}")
    return clean_path

def run_tide_analysis():
    # 1. 原始 Ground Truth 路径
    gt_path = 'data/DIOR/coco_ann/DIOR_test_coco.json'
    # 定义一个清洗后的临时文件路径
    clean_gt_path = 'work_dirs/DIOR_test_coco_clean.json'
    
    # 2. 预测结果路径
    result_path = 'work_dirs/quant_results_dior.json'
    
    if not os.path.exists(gt_path):
        print(f"Error: GT file not found at {gt_path}")
        return
    if not os.path.exists(result_path):
        print(f"Error: Result file not found at {result_path}")
        return

    # --- [核心修改] 清洗 GT 数据 ---
    # 如果清洗后的文件不存在，或者你想要重新生成，执行清洗
    if not os.path.exists(clean_gt_path):
        clean_gt_json(gt_path, clean_gt_path)
    else:
        print(f"Using existing cleaned GT: {clean_gt_path}")
    # -----------------------------

    print(f"Loading Results from: {result_path}")
    print("Initializing TIDE evaluation...")
    
    tide = TIDE()
    
    # 注意：这里使用 clean_gt_path
    tide.evaluate(datasets.COCO(clean_gt_path), datasets.COCOResult(result_path), mode=TIDE.BOX)
    
    print("\n" + "="*60)
    print("                TIDE Error Analysis Summary (DIOR)                ")
    print("="*60)
    
    tide.summarize()
    
    out_dir = 'tide_vis_dior'
    os.makedirs(out_dir, exist_ok=True)
    tide.plot(out_dir=out_dir)
    print("\n" + "="*60)
    print(f"可视化图表已保存至文件夹: {out_dir}/")
    print("="*60)

if __name__ == '__main__':
    run_tide_analysis()
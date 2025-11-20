import argparse
import os
import os.path as osp
import sys
from functools import partial 
import warnings
import torch
from mmcv import Config, DictAction, ConfigDict
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet.apis import single_gpu_test # 需要它来模拟测试循环
from mmdet.datasets import build_dataloader
from mmdet.models import build_detector
from typing import List

# --- 1. 路径与注册修复 (HARD FIX) ---
# 确保项目根目录 (RSCoTr-master) 被加入到 Python 搜索路径中
project_root = osp.dirname(osp.dirname(osp.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ------------------------------------

# --- 2. 导入项目内部模块（必须在路径修复后）---
# 导入数据构建器
try:
    from mtl.data.build import build_datasets 
except ModuleNotFoundError:
    # 如果环境设置正确，这个异常不会被触发
    raise ModuleNotFoundError("CRITICAL ERROR: Failed to import mtl.data.build. Check your project structure or Python path setup.")
    
# 导入模型注册模块 (触发模型注册)
try:
    import models.multi.multitask_learner
    import models.multi.cls_head.slvl_cls_head
    import models.multi.bbox_head.dino_head
    import models.multi.seg_head.mask2former_head
except ImportError as e:
    print(f"Warning: Explicit model import failed: {e}")
# ---------------------------------------------

# --- 2. 【核心修复】 自定义测试函数来注入 task 参数 ---
def custom_test_runner(model, data_loader, task_name, show=False, out_dir=None):
    """
    功能：模拟 single_gpu_test 流程，但手动注入 task 参数给 model.forward
    """
    # 修复 eval 属性缺失
    model_to_eval = model.module if hasattr(model, 'module') else model
    model_to_eval.eval() 
    
    results = []
    from mmcv.utils import ProgressBar
    prog_bar = ProgressBar(len(data_loader.dataset))
    
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            # 注入 task_name 到 data 字典
            data['task'] = task_name 
            
            # MMDataParallel 会将 data dict 解包并传给 self.module.forward(**data)
            # outputs = model(return_loss=False, rescale=True, **data)
            
            # [修正] 确保 model(task, img, img_metas, ...) 调用的参数顺序正确
            # DataParallel 接收 *inputs, **kwargs，但 model.forward 是 (task, img, img_metas, ...)
            # 经过 MMDataParallel 封装，我们只需确保 model(**data) 包含所有键
            outputs = model(return_loss=False, rescale=True, **data)

        # 结果处理
        results.extend(outputs)
        prog_bar.update(len(data['img_metas']))

    return results
# ----------------------------------------------------

# --- 3. 评估指标筛选器 ---
def get_task_metrics(task_type: str, eval_args: List[str]):
    """根据任务类型，从 eval_args 中筛选出正确的指标。"""
    lower_args = [m.lower() for m in eval_args]
    if task_type == 'cls':
        metrics = [m for m in lower_args if 'accuracy' in m or 'top' in m]
        return metrics if metrics else ['accuracy']
    elif task_type == 'det':
        metrics = [m for m in lower_args if 'bbox' in m]
        return metrics if metrics else ['bbox']
    elif task_type == 'seg':
        metrics = [m for m in lower_args if 'miou' in m or 'fscore' in m]
        return metrics if metrics else ['mIoU']
    return args.eval 


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--show-dir', help='directory where painted images will be saved')
    parser.add_argument('--show-score-thr', type=float, default=0.3, help='score threshold (default: 0.3)')
    parser.add_argument('--eval', type=str, nargs='+', help='evaluation metrics')
    parser.add_argument('--options', nargs='+', action=DictAction)
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
        
    cfg.model.pretrained = None

    # 1. 构建模型
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
    
    # 2. 应用量化结构
    if cfg.get('quantize', False):
        print("\n[Test] Applying TSQ-MTC Quantization Structure...")
        if hasattr(model, 'apply_quantization'):
            model.apply_quantization(num_tasks=3)
        elif hasattr(model, 'module') and hasattr(model.module, 'apply_quantization'):
            model.module.apply_quantization(num_tasks=3)
        print("[Test] Quantization layers applied. Ready to load QAT weights.\n")
    
    # 3. 加载 QAT 权重
    print(f"[Test] Loading checkpoint from {args.checkpoint}...")
    load_checkpoint(model, args.checkpoint, map_location='cpu')

    # 4. 构建数据集和 DataParallel
    model = MMDataParallel(model, device_ids=[0])
    datasets_dict = build_datasets(cfg.data, split='test')
    
    # 5. 循环测试每个任务
    for dataset_name, dataset in datasets_dict.items():
        print(f"\n\n{'='*20} Testing Task: {dataset_name} {'='*20}")
        
        # 修复：安全地提取 workers_per_gpu 的值
        task_config_dict = cfg.data.get(dataset_name)
        workers_per_gpu_val = getattr(task_config_dict.get('config'), 'workers_per_gpu', 2)

        data_loader = build_dataloader(
            dataset,
            samples_per_gpu=1, 
            workers_per_gpu=int(workers_per_gpu_val), 
            dist=False,
            shuffle=False)
        
        # --- 运行测试：使用自定义 runner ---
        outputs = custom_test_runner(
            model, 
            data_loader, 
            task_name=dataset.task, # 确保传入正确的 task 属性
            show=args.show, 
            out_dir=args.show_dir
        )
        # ----------------------------------------
        
        # 评估结果
        if args.eval:
            routing_task_type = getattr(dataset, 'task', dataset_name)
            print(f"Evaluating {dataset_name} (Type: {routing_task_type})...")
            
            current_metrics = get_task_metrics(routing_task_type, args.eval)
            task_eval_kwargs = cfg.get('evaluation', {}).get(dataset_name, {})
            
            # 调用评估
            eval_res = dataset.evaluate(outputs, metric=current_metrics, **task_eval_kwargs)
            
            # --- 打印最终结果 ---
            if eval_res:
                 print("="*50)
                 for name, val in eval_res.items():
                     print(f"[Eval Result] {name}: {val}")
                 print("="*50)
            else:
                 print("[Eval Result] Evaluation function returned empty results.")
                 
if __name__ == '__main__':
    main()

# import argparse
# import os
# import os.path as osp
# import sys
# from functools import partial # 引入 partial

# # --- 1. 路径与注册修复 (与 train.py 保持一致) ---
# project_root = osp.dirname(osp.dirname(osp.abspath(__file__)))
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)

# try:
#     import models.multi.multitask_learner
#     import models.multi.cls_head.slvl_cls_head
#     import models.multi.bbox_head.dino_head
#     import models.multi.seg_head.mask2former_head
# except ImportError as e:
#     print(f"Warning: Explicit import failed: {e}")
# # ---------------------------------------------

# import torch
# from mmcv import Config, DictAction
# from mmcv.parallel import MMDataParallel
# from mmcv.runner import load_checkpoint
# from mmdet.apis import single_gpu_test
# from mmdet.datasets import build_dataloader
# from mmdet.models import build_detector
# from mtl.data.build import build_datasets 

# # --- 【核心修复】 自定义测试函数来注入 task 参数 ---
# def custom_test_runner(model, data_loader, task_name, show=False, out_dir=None):
#     """
#     功能：模拟 single_gpu_test 流程，但手动注入 task 参数给 model.forward
#     """
#     # 修复 AttributeError: partial object has no attribute 'eval'
#     model_to_eval = model.module if hasattr(model, 'module') else model
#     model_to_eval.eval() 
    
#     results = []
#     from mmcv.utils import ProgressBar
#     prog_bar = ProgressBar(len(data_loader.dataset))
    
#     for i, data in enumerate(data_loader):
#         with torch.no_grad():
#             # 注入 task_name 到 data 字典
#             data['task'] = task_name 
            
#             # MMDataParallel 会解包，所以 model(**data) 最终就是 MTL.forward(task=task_name, ...)
#             outputs = model(return_loss=False, rescale=True, **data)

#         # 结果处理
#         results.extend(outputs)
#         prog_bar.update(len(data['img_metas']))

#     return results
# # ----------------------------------------------------


# def parse_args():
#     parser = argparse.ArgumentParser(description='MMDet test (and eval) a model')
#     parser.add_argument('config', help='test config file path')
#     parser.add_argument('checkpoint', help='checkpoint file')
#     parser.add_argument('--out', help='output result file in pickle format')
#     parser.add_argument('--show', action='store_true', help='show results')
#     parser.add_argument('--show-dir', help='directory where painted images will be saved')
#     parser.add_argument('--show-score-thr', type=float, default=0.3, help='score threshold (default: 0.3)')
#     parser.add_argument('--eval', type=str, nargs='+', help='evaluation metrics')
#     parser.add_argument('--options', nargs='+', action=DictAction)
#     parser.add_argument('--cfg-options', nargs='+', action=DictAction)
#     parser.add_argument('--local_rank', type=int, default=0)
#     args = parser.parse_args()
#     if 'LOCAL_RANK' not in os.environ:
#         os.environ['LOCAL_RANK'] = str(args.local_rank)
#     return args

# def main():
#     args = parse_args()
#     cfg = Config.fromfile(args.config)
#     if args.cfg_options is not None:
#         cfg.merge_from_dict(args.cfg_options)
    
#     if cfg.get('custom_imports', None):
#         from mmcv.utils import import_modules_from_strings
#         import_modules_from_strings(**cfg['custom_imports'])
        
#     cfg.model.pretrained = None

#     # 1. 构建模型
#     model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
    
#     # 2. 应用量化结构
#     if cfg.get('quantize', False):
#         print("\n[Test] Applying TSQ-MTC Quantization Structure...")
#         if hasattr(model, 'apply_quantization'):
#             model.apply_quantization(num_tasks=3)
#         elif hasattr(model, 'module') and hasattr(model.module, 'apply_quantization'):
#             model.module.apply_quantization(num_tasks=3)
#         print("[Test] Quantization layers applied. Ready to load QAT weights.\n")
    
#     # 3. 加载 QAT 权重
#     print(f"[Test] Loading checkpoint from {args.checkpoint}...")
#     load_checkpoint(model, args.checkpoint, map_location='cpu')

#     # 4. 构建数据集和 DataParallel
#     model = MMDataParallel(model, device_ids=[0])
#     datasets_dict = build_datasets(cfg.data, split='test')
    
#     # 5. 循环测试每个任务
#     for dataset_name, dataset in datasets_dict.items(): # 使用 dataset_name 来避免混淆
#         print(f"\n\n{'='*20} Testing Task: {dataset_name} {'='*20}")
        
#         # --- 【核心修复】 注入正确的路由类型 ---
#         # 我们知道 resisc 是分类，dior 是检测，potsdam 是分割
#         if 'resisc' in dataset_name:
#             routing_task_type = 'cls'
#         elif 'dior' in dataset_name:
#             routing_task_type = 'det'
#         elif 'potsdam' in dataset_name:
#             routing_task_type = 'seg'
#         else:
#             # 兜底，防止意外
#             routing_task_type = 'cls' 
#         # ----------------------------------------

#         # 修复：安全地提取 workers_per_gpu 的值
#         task_config_dict = cfg.data.get(dataset_name)
#         workers_per_gpu_val = getattr(task_config_dict.get('config'), 'workers_per_gpu', 2)

#         data_loader = build_dataloader(
#             dataset,
#             samples_per_gpu=1, 
#             workers_per_gpu=int(workers_per_gpu_val), 
#             dist=False,
#             shuffle=False)
        
#         # --- 运行测试：使用自定义 runner ---
#         outputs = custom_test_runner(
#             model, 
#             data_loader, 
#             task_name=routing_task_type, 
#             show=args.show, 
#             out_dir=args.show_dir
#         )
#         # ----------------------------------------
        
#         # 评估结果
#         if args.eval:
#             print(f"Evaluating {dataset_name} (Type: {routing_task_type})...")
            
#             # 修正：根据任务类型筛选正确的 metric
#             current_metrics = get_task_metrics(routing_task_type, args.eval)

#             # 这里使用 dataset_name 作为 evaluation config 的 key
#             task_eval_kwargs = cfg.get('evaluation', {}).get(dataset_name, {}) 
            
#             # 调用评估
#             eval_res = dataset.evaluate(outputs, metric=current_metrics, **task_eval_kwargs)
            
#             # --- 核心：打印评估结果 ---
#             if eval_res:
#                 print("="*50)
#                 for name, val in eval_res.items():
#                     print(f"[Eval Result] {name}: {val}")
#                 print("="*50)
#             else:
#                 print("[Eval Result] Evaluation function returned empty results.")

# if __name__ == '__main__':
#     main()


# import argparse
# import os
# import os.path as osp
# import time
# import warnings
# import sys
# from functools import partial # <--- 引入 partial

# # --- 1. 路径与注册修复 (与 train.py 保持一致) ---
# project_root = osp.dirname(osp.dirname(osp.abspath(__file__)))
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)

# try:
#     import models.multi.multitask_learner
#     import models.multi.cls_head.slvl_cls_head
#     import models.multi.bbox_head.dino_head
#     import models.multi.seg_head.mask2former_head
# except ImportError as e:
#     print(f"Warning: Explicit import failed: {e}")
# # ---------------------------------------------

# import mmcv
# import torch
# from mmcv import Config, DictAction, ConfigDict
# from mmcv.parallel import MMDataParallel
# from mmcv.runner import load_checkpoint
# from mmdet.apis import single_gpu_test
# from mmdet.datasets import build_dataloader
# from mmdet.models import build_detector
# from mtl.data.build import build_datasets 

# def parse_args():
#     parser = argparse.ArgumentParser(description='MMDet test (and eval) a model')
#     parser.add_argument('config', help='test config file path')
#     parser.add_argument('checkpoint', help='checkpoint file')
#     parser.add_argument('--out', help='output result file in pickle format')
#     parser.add_argument('--show', action='store_true', help='show results')
#     parser.add_argument('--show-dir', help='directory where painted images will be saved')
#     parser.add_argument('--show-score-thr', type=float, default=0.3, help='score threshold (default: 0.3)')
#     parser.add_argument('--eval', type=str, nargs='+', help='evaluation metrics')
#     parser.add_argument('--options', nargs='+', action=DictAction)
#     parser.add_argument('--cfg-options', nargs='+', action=DictAction)
#     parser.add_argument('--local_rank', type=int, default=0)
#     args = parser.parse_args()
#     if 'LOCAL_RANK' not in os.environ:
#         os.environ['LOCAL_RANK'] = str(args.local_rank)
#     return args

# def main():
#     args = parse_args()
#     cfg = Config.fromfile(args.config)
#     if args.cfg_options is not None:
#         cfg.merge_from_dict(args.cfg_options)
    
#     if cfg.get('custom_imports', None):
#         from mmcv.utils import import_modules_from_strings
#         import_modules_from_strings(**cfg['custom_imports'])
        
#     cfg.model.pretrained = None

#     # 1. 构建模型
#     model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
    
#     # 2. 应用量化结构
#     if cfg.get('quantize', False):
#         print("\n[Test] Applying TSQ-MTC Quantization Structure...")
#         if hasattr(model, 'apply_quantization'):
#             model.apply_quantization(num_tasks=3)
#         elif hasattr(model, 'module') and hasattr(model.module, 'apply_quantization'):
#             model.module.apply_quantization(num_tasks=3)
#         print("[Test] Quantization layers applied. Ready to load QAT weights.\n")
    
#     # 3. 加载 QAT 权重
#     print(f"[Test] Loading checkpoint from {args.checkpoint}...")
#     load_checkpoint(model, args.checkpoint, map_location='cpu')

#     # 4. 构建数据集和 DataParallel
#     model = MMDataParallel(model, device_ids=[0])
#     datasets_dict = build_datasets(cfg.data, split='test')
    
#     # 5. 循环测试每个任务
#     for task_name, dataset in datasets_dict.items():
#         print(f"\n\n{'='*20} Testing Task: {task_name} {'='*20}")
        
#         # 修复：安全地提取 workers_per_gpu 的值
#         task_config_dict = cfg.data.get(task_name)
#         workers_per_gpu_val = getattr(task_config_dict.get('config'), 'workers_per_gpu', 2)
#         workers_per_gpu_val = int(workers_per_gpu_val) 

#         data_loader = build_dataloader(
#             dataset,
#             samples_per_gpu=1, 
#             workers_per_gpu=workers_per_gpu_val, 
#             dist=False,
#             shuffle=False)
        
#         # --- 【核心修复】 使用 partial 预先绑定 task 参数 ---
#         # 这样 single_gpu_test 调用 model 时，task 参数就自动注入了
#         # model.forward 的签名是 model(task, img, img_metas, ...)
        
#         # 创建一个包装函数，将 task_name 作为第一个参数传入 model.forward
#         model_wrapped = partial(model.forward, task=task_name)
        
#         # 运行测试：这里调用的是 model_wrapped，它会把 task_name 传给 model.forward
#         outputs = single_gpu_test(
#             model_wrapped, # 使用包装后的函数
#             data_loader, 
#             show=args.show, 
#             out_dir=args.show_dir,
#         )
        
#         # 评估结果
#         if args.eval:
#             print(f"Evaluating {task_name}...")
#             task_eval_kwargs = cfg.get('evaluation', {}).get(task_name, {})
            
#             # 由于 evaluation API 期望 model 对象，这里我们直接传入 dataset.evaluate
#             eval_res = dataset.evaluate(outputs, metric=args.eval, **task_eval_kwargs)
            
#             for name, val in eval_res.items():
#                 print(f"[Eval] {name}: {val}")

# if __name__ == '__main__':
#     main()
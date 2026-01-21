import sys
import os.path as osp

# 1. 路径修复
project_root = osp.dirname(osp.dirname(osp.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 2. 显式导入所有模型文件以触发注册
try:
    import models.multi.multitask_learner
    import models.multi.cls_head.slvl_cls_head
    import models.multi.bbox_head.dino_head
    import models.multi.seg_head.mask2former_head
except ImportError as e:
    print(f"Warning: Failed to import models explicitly: {e}")

import argparse
import copy
import os
import time
import warnings
import mmcv
import torch
import torch.distributed as dist
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmdet.utils import collect_env, get_root_logger
from mmdet.apis import set_random_seed
from mtl.apis.train import train_model
from mmdet.models import build_detector as build_model
from mtl.data.build import build_datasets

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--resume-from', help='the checkpoint file to resume from')
    parser.add_argument('--auto-resume', action='store_true')
    parser.add_argument('--no-validate', action='store_true')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument('--gpus', type=int)
    group_gpus.add_argument('--gpu-ids', type=int, nargs='+')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--diff-seed', action='store_true')
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--options', nargs='+', action=DictAction)
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--load_task_pretrain', action='store_true')
    
    # [新增] Phase 1 开关
    parser.add_argument('--gen-mask', action='store_true', help='Phase 1: Generate fine-grained mask and exit')
    
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    if args.options and args.cfg_options:
        raise ValueError('--options and --cfg-options cannot be both specified')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options
    return args

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpus is not None:
        cfg.gpu_ids = range(1)
    else:
        cfg.gpu_ids = [0]
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)
    
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)
    
    meta = dict()
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    logger.info('Environment info:\n' + '-' * 60 + '\n' + env_info + '\n' + '-' * 60)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text
    meta['seed'] = args.seed
    seed = args.seed
    if seed is not None:
        logger.info(f'Set random seed to {seed}, deterministic: {args.deterministic}')
        set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed

    # Step 1: 构建模型
    model = build_model(cfg.model)
    model.init_weights()

    # Step 2: 加载 FP32 权重
    fp32_ckpt = cfg.get('load_from_fp32', None) or cfg.get('load_from', None)
    if fp32_ckpt:
        logger.info(f"Loading FP32 weights from {fp32_ckpt} for initialization/calibration ...")
        load_checkpoint(model, fp32_ckpt, map_location='cpu', strict=False, logger=logger)
    elif args.load_task_pretrain:
        if hasattr(model, 'load_task_pretrain'):
            logger.info("Loading task pretrained weights via model.load_task_pretrain() ...")
            model.load_task_pretrain()
    else:
        logger.warning("Warning: No pretrained weights loaded! Calibration will be random if quantize=True.")

    # Step 3: 构建数据集
    logger.info("Building datasets...")
    datasets = build_datasets(cfg.data)
    CLASSES = {name: dataset.CLASSES for name, dataset in datasets.items()}
    model.CLASSES = CLASSES

    # =============================================================================
    # Step 4: 应用量化策略 (TACQ + Multi-Task Support)
    # =============================================================================
    if cfg.get('quantize', False):
        mask_file_path = osp.join(cfg.work_dir, 'fine_grained_mask.pth')
        
        # [核心类] 多任务校准加载器
        class MultiTaskCalibLoader:
            def __init__(self, datasets_dict, samples_per_task=5):
                self.loaders = []
                self.samples_per_task = samples_per_task
                from mmdet.datasets import build_dataloader
                import numpy as np
                
                logger.info(f"Building Multi-Task Calibration Loader (samples_per_task={samples_per_task})...")
                for name, ds in datasets_dict.items():
                    # 自动推断任务类型
                    task_type = 'cls'
                    if 'dior' in name.lower(): task_type = 'det'
                    elif 'potsdam' in name.lower(): task_type = 'seg'
                    elif 'resisc' in name.lower(): task_type = 'cls'
                    
                    if not hasattr(ds, 'flag'):
                        ds.flag = np.zeros(len(ds), dtype=np.int64)
                        
                    loader = build_dataloader(
                        ds, samples_per_gpu=2, workers_per_gpu=0, dist=False, shuffle=True
                    )
                    self.loaders.append((loader, task_type))
                    logger.info(f" - Added calibration task: {name} -> Task: {task_type}")

            def __iter__(self):
                # 轮流遍历每个任务的数据
                for loader, task_type in self.loaders:
                    for i, data in enumerate(loader):
                        if i >= self.samples_per_task: break
                        
                        # [关键] 注入任务标识，让模型知道这是哪个任务的数据
                        if isinstance(data, dict):
                            data['task'] = task_type
                        elif hasattr(data, 'data') and isinstance(data.data[0], dict):
                             data.data[0]['task'] = task_type
                        
                        yield data

        def build_calib_loader():
            # 使用所有数据集，每个任务跑 5 个 batch
            return MultiTaskCalibLoader(datasets, samples_per_task=5)

        # ---------------------------------------------------------------------
        # [Phase 1] 生成 Mask
        # ---------------------------------------------------------------------
        if args.gen_mask:
            logger.info(f"\n{'='*20} [Phase 1] Generating Fine-Grained Mask (TACQ + Multi-Task) {'='*20}")
            calib_loader = build_calib_loader()
            
            if hasattr(model, 'generate_fine_grained_mask'):
                model.generate_fine_grained_mask(
                    calib_loader, 
                    save_path=mask_file_path, 
                    ratio_high=0.05, 
                    default_task='cls' 
                )
            elif hasattr(model, 'module') and hasattr(model.module, 'generate_fine_grained_mask'):
                model.module.generate_fine_grained_mask(
                    calib_loader, 
                    save_path=mask_file_path, 
                    ratio_high=0.05, 
                    default_task='cls'
                )
            else:
                raise RuntimeError("Model does not implement 'generate_fine_grained_mask'.")
                
            logger.info("Mask generation finished. Exiting program.")
            return

        # ---------------------------------------------------------------------
        # [Phase 2] 应用 Mask
        # ---------------------------------------------------------------------
        elif osp.exists(mask_file_path):
            logger.info(f"\n{'='*20} [Phase 2] Applying Fine-Grained Mask for QAT {'='*20}")
            logger.info(f"Loading mask from: {mask_file_path}")
            
            if hasattr(model, 'apply_fine_grained_quantization'):
                model.apply_fine_grained_quantization(mask_file_path, num_tasks=3)
            elif hasattr(model, 'module') and hasattr(model.module, 'apply_fine_grained_quantization'):
                model.module.apply_fine_grained_quantization(mask_file_path, num_tasks=3)
            else:
                logger.warning("Model missing 'apply_fine_grained_quantization' method!")
        else:
            raise FileNotFoundError(f"【严重错误】在 {mask_file_path} 未找到 Mask 文件！请先运行 --gen-mask。")

        if cfg.get('resume_from'):
             logger.info(f"Resuming QAT training from {cfg.resume_from} ...")

    # Step 5: 开始训练
    if not hasattr(cfg, 'device'):
        cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    train_model(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)

if __name__ == '__main__':
    main()





# import sys
# import os.path as osp

# # 1. 路径修复
# project_root = osp.dirname(osp.dirname(osp.abspath(__file__)))
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)

# # 2. 显式导入所有模型文件以触发注册 (HEADS & DETECTORS)
# try:
#     import models.multi.multitask_learner
#     import models.multi.cls_head.slvl_cls_head
#     import models.multi.bbox_head.dino_head
#     import models.multi.seg_head.mask2former_head
# except ImportError as e:
#     print(f"Warning: Failed to import models explicitly: {e}")

# import argparse
# import copy
# import os
# import time
# import warnings
# import mmcv
# import torch
# import torch.distributed as dist
# from mmcv import Config, DictAction
# from mmcv.runner import get_dist_info, init_dist, load_checkpoint
# from mmdet.utils import collect_env, get_root_logger
# from mmdet.apis import set_random_seed
# from mtl.apis.train import train_model
# from mmdet.models import build_detector as build_model
# from mtl.data.build import build_datasets

# def parse_args():
#     parser = argparse.ArgumentParser(description='Train a detector')
#     parser.add_argument('config', help='train config file path')
#     parser.add_argument('--work-dir', help='the dir to save logs and models')
#     parser.add_argument('--resume-from', help='the checkpoint file to resume from')
#     parser.add_argument('--auto-resume', action='store_true')
#     parser.add_argument('--no-validate', action='store_true')
#     group_gpus = parser.add_mutually_exclusive_group()
#     group_gpus.add_argument('--gpus', type=int)
#     group_gpus.add_argument('--gpu-ids', type=int, nargs='+')
#     parser.add_argument('--seed', type=int, default=None)
#     parser.add_argument('--diff-seed', action='store_true')
#     parser.add_argument('--deterministic', action='store_true')
#     parser.add_argument('--options', nargs='+', action=DictAction)
#     parser.add_argument('--cfg-options', nargs='+', action=DictAction)
#     parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none')
#     parser.add_argument('--local_rank', type=int, default=0)
#     parser.add_argument('--load_task_pretrain', action='store_true')
    
#     # [新增] Phase 1 开关：仅生成 Mask 并退出
#     parser.add_argument('--gen-mask', action='store_true', help='Phase 1: Generate fine-grained mask and exit')
    
#     args = parser.parse_args()
#     if 'LOCAL_RANK' not in os.environ:
#         os.environ['LOCAL_RANK'] = str(args.local_rank)
#     if args.options and args.cfg_options:
#         raise ValueError('--options and --cfg-options cannot be both specified')
#     if args.options:
#         warnings.warn('--options is deprecated in favor of --cfg-options')
#         args.cfg_options = args.options
#     return args

# def main():
#     args = parse_args()
#     cfg = Config.fromfile(args.config)
#     if args.cfg_options is not None:
#         cfg.merge_from_dict(args.cfg_options)
#     if cfg.get('cudnn_benchmark', False):
#         torch.backends.cudnn.benchmark = True
#     if args.work_dir is not None:
#         cfg.work_dir = args.work_dir
#     elif cfg.get('work_dir', None) is None:
#         cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])
#     if args.resume_from is not None:
#         cfg.resume_from = args.resume_from
#     if args.gpus is not None:
#         cfg.gpu_ids = range(1)
#     else:
#         cfg.gpu_ids = [0]
#     if args.launcher == 'none':
#         distributed = False
#     else:
#         distributed = True
#         init_dist(args.launcher, **cfg.dist_params)
#         _, world_size = get_dist_info()
#         cfg.gpu_ids = range(world_size)
    
#     mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
#     timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
#     log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
#     logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)
    
#     meta = dict()
#     env_info_dict = collect_env()
#     env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
#     logger.info('Environment info:\n' + '-' * 60 + '\n' + env_info + '\n' + '-' * 60)
#     meta['env_info'] = env_info
#     meta['config'] = cfg.pretty_text
#     meta['seed'] = args.seed
#     seed = args.seed
#     if seed is not None:
#         logger.info(f'Set random seed to {seed}, deterministic: {args.deterministic}')
#         set_random_seed(seed, deterministic=args.deterministic)
#     cfg.seed = seed

#     # =============================================================================
#     # Step 1: 构建 FP32 模型
#     # =============================================================================
#     model = build_model(cfg.model)
#     model.init_weights()

#     # =============================================================================
#     # Step 2: 加载 FP32 权重 (必须在量化前完成)
#     # =============================================================================
#     fp32_ckpt = cfg.get('load_from_fp32', None) or cfg.get('load_from', None)
    
#     if fp32_ckpt:
#         logger.info(f"Loading FP32 weights from {fp32_ckpt} for initialization/calibration ...")
#         load_checkpoint(model, fp32_ckpt, map_location='cpu', strict=False, logger=logger)
#     elif args.load_task_pretrain:
#         if hasattr(model, 'load_task_pretrain'):
#             logger.info("Loading task pretrained weights via model.load_task_pretrain() ...")
#             model.load_task_pretrain()
#     else:
#         logger.warning("Warning: No pretrained weights loaded! Calibration will be random if quantize=True.")

#     # =============================================================================
#     # Step 3: 构建数据集 (提前构建，因量化校准需要数据)
#     # =============================================================================
#     logger.info("Building datasets...")
#     datasets = build_datasets(cfg.data)
    
#     CLASSES = {name: dataset.CLASSES for name, dataset in datasets.items()}
#     model.CLASSES = CLASSES

#     # =============================================================================
#     # Step 4: 应用量化策略 (Two-Stage Fine-Grained QAT)
#     # =============================================================================
#     if cfg.get('quantize', False):
        
#         # 定义 Mask 文件路径
#         mask_file_path = osp.join(cfg.work_dir, 'fine_grained_mask.pth')
        
#         # 定义辅助函数：构建校准 DataLoader
#         def build_calib_loader():
#             logger.info("Preparing calibration dataloader...")
#             dataset_keys = list(datasets.keys())
#             calib_dataset_name = dataset_keys[0]
#             calib_dataset = datasets[calib_dataset_name]
            
#             # 自动推断任务
#             target_task = 'cls'
#             if 'resisc' in calib_dataset_name.lower(): target_task = 'cls'
#             elif 'dior' in calib_dataset_name.lower(): target_task = 'det'
#             elif 'potsdam' in calib_dataset_name.lower(): target_task = 'seg'
            
#             logger.info(f"Calibration dataset: {calib_dataset_name}, Task: {target_task}")

#             if not hasattr(calib_dataset, 'flag'):
#                 import numpy as np
#                 calib_dataset.flag = np.zeros(len(calib_dataset), dtype=np.int64)

#             from mmdet.datasets import build_dataloader
#             loader = build_dataloader(
#                 calib_dataset, samples_per_gpu=2, workers_per_gpu=0, dist=False, shuffle=True
#             )
#             return loader, target_task

#         # ---------------------------------------------------------------------
#         # [Phase 1] 生成 Mask 阶段 (使用 --gen-mask 参数触发)
#         # ---------------------------------------------------------------------
#         if args.gen_mask:
#             logger.info(f"\n{'='*20} [Phase 1] Generating Fine-Grained Mask {'='*20}")
#             calib_loader, default_task = build_calib_loader()
            
#             if hasattr(model, 'generate_fine_grained_mask'):
#                 # ratio_high=0.05 即 Top 5% 权重为 8-bit
#                 model.generate_fine_grained_mask(
#                     calib_loader, 
#                     save_path=mask_file_path, 
#                     ratio_high=0.05, 
#                     default_task=default_task
#                 )
#             elif hasattr(model, 'module') and hasattr(model.module, 'generate_fine_grained_mask'):
#                 model.module.generate_fine_grained_mask(
#                     calib_loader, 
#                     save_path=mask_file_path, 
#                     ratio_high=0.05, 
#                     default_task=default_task
#                 )
#             else:
#                 raise RuntimeError("Model does not implement 'generate_fine_grained_mask'. Check multitask_learner.py.")
                
#             logger.info("Mask generation finished. Exiting program.")
#             return  # 阶段一完成后直接退出

#         # ---------------------------------------------------------------------
#         # [Phase 2] 应用 Mask 进行训练阶段
#         # ---------------------------------------------------------------------
#         # 这里增加了强制检查，如果没开生成模式且 mask 不存在，直接报错，防止回退到普通量化
#         elif osp.exists(mask_file_path):
#             logger.info(f"\n{'='*20} [Phase 2] Applying Fine-Grained Mask for QAT {'='*20}")
#             logger.info(f"Loading mask from: {mask_file_path}")
            
#             if hasattr(model, 'apply_fine_grained_quantization'):
#                 model.apply_fine_grained_quantization(mask_file_path, num_tasks=3)
#             elif hasattr(model, 'module') and hasattr(model.module, 'apply_fine_grained_quantization'):
#                 model.module.apply_fine_grained_quantization(mask_file_path, num_tasks=3)
#             else:
#                 logger.warning("Model missing 'apply_fine_grained_quantization' method!")

#         # ---------------------------------------------------------------------
#         # [Error/Fallback] 如果没有 Mask 也没指定生成，报错或回退
#         # ---------------------------------------------------------------------
#         else:
#             # 强烈建议这里直接报错，防止你再次遇到精度崩塌的问题
#             # 如果你想回退，可以把 raise 换成 logger.warning 并取消下面的注释
#             raise FileNotFoundError(f"【严重错误】在 {mask_file_path} 未找到 Mask 文件！\n"
#                                     "请先运行 Phase 1 (--gen-mask) 生成 Mask，或检查路径。")
            
#             # logger.info("\n[Warning] No mask found and --gen-mask not set. Falling back to standard TACQ/Uniform QAT...")
#             # calib_loader, default_task = build_calib_loader()
#             # if hasattr(model, 'apply_mixed_precision_quantization'):
#             #     model.apply_mixed_precision_quantization(
#             #         data_loader=calib_loader, num_tasks=3, ratio_8bit=0.5, default_task=default_task)
#             # ...

#         # 加载 QAT Checkpoint (Resume 逻辑)
#         if cfg.get('resume_from'):
#              logger.info(f"Resuming QAT training from {cfg.resume_from} ...")
#              # checkpoint 由 train_model 内部的 runner 处理，这里仅打印日志

#     # =============================================================================
#     # Step 5: 开始训练
#     # =============================================================================
#     if not hasattr(cfg, 'device'):
#         cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
#     train_model(
#         model,
#         datasets,
#         cfg,
#         distributed=distributed,
#         validate=(not args.no_validate),
#         timestamp=timestamp,
#         meta=meta)

# if __name__ == '__main__':
#     main()


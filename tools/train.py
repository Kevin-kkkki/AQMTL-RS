import sys
import os.path as osp

# 1. 路径修复
project_root = osp.dirname(osp.dirname(osp.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 2. 显式导入所有模型文件以触发注册 (HEADS & DETECTORS)
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
    
    # [新增] Phase 1 开关：仅生成 Mask 并退出
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

    # =============================================================================
    # Step 1: 构建 FP32 模型
    # =============================================================================
    model = build_model(cfg.model)
    model.init_weights()

    # =============================================================================
    # Step 2: 加载 FP32 权重 (必须在量化前完成)
    # =============================================================================
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

    # =============================================================================
    # Step 3: 构建数据集 (提前构建，因量化校准需要数据)
    # =============================================================================
    logger.info("Building datasets...")
    datasets = build_datasets(cfg.data)
    
    CLASSES = {name: dataset.CLASSES for name, dataset in datasets.items()}
    model.CLASSES = CLASSES

    # =============================================================================
    # Step 4: 应用量化策略 (Two-Stage Fine-Grained QAT)
    # =============================================================================
    if cfg.get('quantize', False):
        
        # 定义 Mask 文件路径
        mask_file_path = osp.join(cfg.work_dir, 'fine_grained_mask.pth')
        
        # 定义辅助函数：构建校准 DataLoader
        def build_calib_loader():
            logger.info("Preparing calibration dataloader...")
            dataset_keys = list(datasets.keys())
            calib_dataset_name = dataset_keys[0]
            calib_dataset = datasets[calib_dataset_name]
            
            # 自动推断任务
            target_task = 'cls'
            if 'resisc' in calib_dataset_name.lower(): target_task = 'cls'
            elif 'dior' in calib_dataset_name.lower(): target_task = 'det'
            elif 'potsdam' in calib_dataset_name.lower(): target_task = 'seg'
            
            logger.info(f"Calibration dataset: {calib_dataset_name}, Task: {target_task}")

            if not hasattr(calib_dataset, 'flag'):
                import numpy as np
                calib_dataset.flag = np.zeros(len(calib_dataset), dtype=np.int64)

            from mmdet.datasets import build_dataloader
            loader = build_dataloader(
                calib_dataset, samples_per_gpu=2, workers_per_gpu=0, dist=False, shuffle=True
            )
            return loader, target_task

        # ---------------------------------------------------------------------
        # [Phase 1] 生成 Mask 阶段 (使用 --gen-mask 参数触发)
        # ---------------------------------------------------------------------
        if args.gen_mask:
            logger.info(f"\n{'='*20} [Phase 1] Generating Fine-Grained Mask {'='*20}")
            calib_loader, default_task = build_calib_loader()
            
            if hasattr(model, 'generate_fine_grained_mask'):
                # ratio_high=0.05 即 Top 5% 权重为 8-bit
                model.generate_fine_grained_mask(
                    calib_loader, 
                    save_path=mask_file_path, 
                    ratio_high=0.05, 
                    default_task=default_task
                )
            elif hasattr(model, 'module') and hasattr(model.module, 'generate_fine_grained_mask'):
                model.module.generate_fine_grained_mask(
                    calib_loader, 
                    save_path=mask_file_path, 
                    ratio_high=0.05, 
                    default_task=default_task
                )
            else:
                raise RuntimeError("Model does not implement 'generate_fine_grained_mask'. Check multitask_learner.py.")
                
            logger.info("Mask generation finished. Exiting program.")
            return  # 阶段一完成后直接退出

        # ---------------------------------------------------------------------
        # [Phase 2] 应用 Mask 进行训练阶段
        # ---------------------------------------------------------------------
        # 这里增加了强制检查，如果没开生成模式且 mask 不存在，直接报错，防止回退到普通量化
        elif osp.exists(mask_file_path):
            logger.info(f"\n{'='*20} [Phase 2] Applying Fine-Grained Mask for QAT {'='*20}")
            logger.info(f"Loading mask from: {mask_file_path}")
            
            if hasattr(model, 'apply_fine_grained_quantization'):
                model.apply_fine_grained_quantization(mask_file_path, num_tasks=3)
            elif hasattr(model, 'module') and hasattr(model.module, 'apply_fine_grained_quantization'):
                model.module.apply_fine_grained_quantization(mask_file_path, num_tasks=3)
            else:
                logger.warning("Model missing 'apply_fine_grained_quantization' method!")

        # ---------------------------------------------------------------------
        # [Error/Fallback] 如果没有 Mask 也没指定生成，报错或回退
        # ---------------------------------------------------------------------
        else:
            # 强烈建议这里直接报错，防止你再次遇到精度崩塌的问题
            # 如果你想回退，可以把 raise 换成 logger.warning 并取消下面的注释
            raise FileNotFoundError(f"【严重错误】在 {mask_file_path} 未找到 Mask 文件！\n"
                                    "请先运行 Phase 1 (--gen-mask) 生成 Mask，或检查路径。")
            
            # logger.info("\n[Warning] No mask found and --gen-mask not set. Falling back to standard TACQ/Uniform QAT...")
            # calib_loader, default_task = build_calib_loader()
            # if hasattr(model, 'apply_mixed_precision_quantization'):
            #     model.apply_mixed_precision_quantization(
            #         data_loader=calib_loader, num_tasks=3, ratio_8bit=0.5, default_task=default_task)
            # ...

        # 加载 QAT Checkpoint (Resume 逻辑)
        if cfg.get('resume_from'):
             logger.info(f"Resuming QAT training from {cfg.resume_from} ...")
             # checkpoint 由 train_model 内部的 runner 处理，这里仅打印日志

    # =============================================================================
    # Step 5: 开始训练
    # =============================================================================
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
# sys.path.append(r'/media/at03/4tb/jiangrongkai_workspace/RSCoTr-master')
# import argparse
# import copy
# import os
# import os.path as osp
# import time
# import warnings

# import mmcv
# import torch
# import torch.distributed as dist
# from mmcv import Config, DictAction
# from mmcv.runner import get_dist_info, init_dist
# from mmcv.runner import load_checkpoint
# from mmcv.utils import get_git_hash
# from mmcv.cnn import MODELS

# from mmcls import __version__ as mmcls_version
# from mmdet import __version__ as mmdet_version
# from mmseg import __version__ as mmseg_version

# from mmdet.apis import init_random_seed, set_random_seed
# from mmdet.utils import (collect_env, get_device, get_root_logger,
#                          replace_cfg_vals, setup_multi_processes)
# import sys
# sys.path.append(r'F:\RSCoTr')
# from mtl.apis import train_model
# from mtl.data.build import build_datasets, load_data_cfg


# def parse_args():
#     parser = argparse.ArgumentParser(description='Train a detector')
#     parser.add_argument('config', help='train config file path')
#     parser.add_argument(
#         '--load-task-pretrain',
#         action='store_true',
#         help='whether to load pretrained weight of deformable detr')
#     parser.add_argument('--work-dir', help='the dir to save logs and models')
#     parser.add_argument(
#         '--resume-from', help='the checkpoint file to resume from')
#     parser.add_argument(
#         '--auto-resume',
#         action='store_true',
#         help='resume from the latest checkpoint automatically')
#     parser.add_argument(
#         '--no-validate',
#         action='store_true',
#         help='whether not to evaluate the checkpoint during training')
#     group_gpus = parser.add_mutually_exclusive_group()
#     group_gpus.add_argument(
#         '--gpus',
#         type=int,
#         help='(Deprecated, please use --gpu-id) number of gpus to use '
#         '(only applicable to non-distributed training)')
#     group_gpus.add_argument(
#         '--gpu-ids',
#         type=int,
#         nargs='+',
#         help='(Deprecated, please use --gpu-id) ids of gpus to use '
#         '(only applicable to non-distributed training)')
#     group_gpus.add_argument(
#         '--gpu-id',
#         type=int,
#         default=0,
#         help='id of gpu to use '
#         '(only applicable to non-distributed training)')
#     parser.add_argument('--seed', type=int, default=None, help='random seed')
#     parser.add_argument(
#         '--diff-seed',
#         action='store_true',
#         help='Whether or not set different seeds for different ranks')
#     parser.add_argument(
#         '--deterministic',
#         action='store_true',
#         help='whether to set deterministic options for CUDNN backend.')
#     parser.add_argument(
#         '--options',
#         nargs='+',
#         action=DictAction,
#         help='override some settings in the used config, the key-value pair '
#         'in xxx=yyy format will be merged into config file (deprecate), '
#         'change to --cfg-options instead.')
#     parser.add_argument(
#         '--cfg-options',
#         nargs='+',
#         action=DictAction,
#         help='override some settings in the used config, the key-value pair '
#         'in xxx=yyy format will be merged into config file. If the value to '
#         'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
#         'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
#         'Note that the quotation marks are necessary and that no white space '
#         'is allowed.')
#     parser.add_argument(
#         '--launcher',
#         choices=['none', 'pytorch', 'slurm', 'mpi'],
#         default='none',
#         help='job launcher')
#     parser.add_argument('--local_rank', type=int, default=0)
#     parser.add_argument(
#         '--auto-scale-lr',
#         action='store_true',
#         help='enable automatically scaling LR.')
#     args = parser.parse_args()
#     if 'LOCAL_RANK' not in os.environ:
#         os.environ['LOCAL_RANK'] = str(args.local_rank)

#     if args.options and args.cfg_options:
#         raise ValueError(
#             '--options and --cfg-options cannot be both '
#             'specified, --options is deprecated in favor of --cfg-options')
#     if args.options:
#         warnings.warn('--options is deprecated in favor of --cfg-options')
#         args.cfg_options = args.options

#     return args


# def main():
#     args = parse_args()

#     cfg = Config.fromfile(args.config)

#     # replace the ${key} with the value of cfg.key
#     cfg = replace_cfg_vals(cfg)

#     # load configs from dataset config files
#     load_data_cfg(cfg)

#     if args.cfg_options is not None:
#         cfg.merge_from_dict(args.cfg_options)

#     if args.auto_scale_lr:
#         if 'auto_scale_lr' in cfg and \
#                 'enable' in cfg.auto_scale_lr and \
#                 'base_batch_size' in cfg.auto_scale_lr:
#             cfg.auto_scale_lr.enable = True
#         else:
#             warnings.warn('Can not find "auto_scale_lr" or '
#                           '"auto_scale_lr.enable" or '
#                           '"auto_scale_lr.base_batch_size" in your'
#                           ' configuration file.')

#     # set multi-process settings
#     setup_multi_processes(cfg)

#     # set cudnn_benchmark
#     if cfg.get('cudnn_benchmark', False):
#         torch.backends.cudnn.benchmark = True

#     # work_dir is determined in this priority: CLI > segment in file > filename
#     if args.work_dir is not None:
#         # update configs according to CLI args if args.work_dir is not None
#         cfg.work_dir = args.work_dir
#     elif cfg.get('work_dir', None) is None:
#         # use config filename as default work_dir if cfg.work_dir is None
#         cfg.work_dir = osp.join('./work_dirs',
#                                 osp.splitext(osp.basename(args.config))[0])

#     if args.resume_from is not None:
#         cfg.resume_from = args.resume_from
#     cfg.auto_resume = args.auto_resume
#     if args.gpus is not None:
#         cfg.gpu_ids = range(1)
#         warnings.warn('`--gpus` is deprecated because we only support '
#                       'single GPU mode in non-distributed training. '
#                       'Use `gpus=1` now.')
#     if args.gpu_ids is not None:
#         cfg.gpu_ids = args.gpu_ids[0:1]
#         warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
#                       'Because we only support single GPU mode in '
#                       'non-distributed training. Use the first GPU '
#                       'in `gpu_ids` now.')
#     if args.gpus is None and args.gpu_ids is None:
#         cfg.gpu_ids = [args.gpu_id]

#     # init distributed env first, since logger depends on the dist info.
#     if args.launcher == 'none':
#         distributed = False
#     else:
#         distributed = True
#         init_dist(args.launcher, **cfg.dist_params)
#         # re-set gpu_ids with distributed training mode
#         _, world_size = get_dist_info()
#         cfg.gpu_ids = range(world_size)

#     # create work_dir
#     mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
#     # dump config
#     cfg.load_task_pretrain = args.load_task_pretrain
#     cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
#     # init the logger before other steps
#     timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
#     log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
#     logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

#     # init the meta dict to record some important information such as
#     # environment info and seed, which will be logged
#     meta = dict()
#     # log env info
#     env_info_dict = collect_env()
#     env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
#     dash_line = '-' * 60 + '\n'
#     logger.info('Environment info:\n' + dash_line + env_info + '\n' +
#                 dash_line)
#     meta['env_info'] = env_info
#     meta['config'] = cfg.pretty_text
#     # log some basic info
#     logger.info(f'Distributed training: {distributed}')
#     logger.info(f'Config:\n{cfg.pretty_text}')

#     cfg.device = get_device()
#     # set random seeds
#     seed = init_random_seed(args.seed, device=cfg.device)
#     seed = seed + dist.get_rank() if args.diff_seed else seed
#     logger.info(f'Set random seed to {seed}, '
#                 f'deterministic: {args.deterministic}')
#     set_random_seed(seed, deterministic=args.deterministic)
#     cfg.seed = seed
#     meta['seed'] = seed
#     meta['exp_name'] = osp.basename(args.config)

#     ###################### start exp. ###################
#     model = MODELS.build(cfg.model)
#     model.init_weights()
#     fp32_ckpt = cfg.get('load_from_fp32', None)
#     if fp32_ckpt:
#         logger.info(f"Loading FP32 weights from {fp32_ckpt}...")
#         load_checkpoint(model, fp32_ckpt, map_location='cpu', strict=False, logger=logger)

#     if args.load_task_pretrain and not fp32_ckpt:
#         model.load_task_pretrain()
#     # if args.load_task_pretrain:
#     #     model.load_task_pretrain()

#     # 应用量化替换
#     if cfg.get('quantize', False):
#         logger.info("Applying TSQ-MTC Quantization (replacing layers)...")
#         # 调用我们在 MTL 中写好的 apply_quantization
#         model.apply_quantization(num_tasks=3) 
#         logger.info("Quantization layers applied.")
        
#         # 如果是恢复 QAT 训练 (resume/load_from)，在这之后加载 QAT 权重
#         if cfg.get('load_from'):
#             logger.info(f"Loading QAT weights from {cfg.load_from}...")
#             load_checkpoint(model, cfg.load_from, map_location='cpu', logger=logger)
            
#     else:
#         # 非量化模式下的常规加载
#         if cfg.get('load_from'):
#             load_checkpoint(model, cfg.load_from, map_location='cpu', logger=logger)

#     datasets = build_datasets(cfg.data)
#     CLASSES = {name: dataset.CLASSES for name, dataset in datasets.items()}
#     if cfg.checkpoint_config is not None:
#         # save mmdet version, config file content and class names in
#         # checkpoints as meta data
#         cfg.checkpoint_config.meta = dict(
#             mmcls_version=mmcls_version + get_git_hash()[:7],
#             mmdet_version=mmdet_version + get_git_hash()[:7],
#             mmseg_version=mmseg_version + get_git_hash()[:7],
#             CLASSES=CLASSES)
#     # add an attribute for visualization convenience
#     model.CLASSES = CLASSES
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

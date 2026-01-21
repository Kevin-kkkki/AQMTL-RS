import sys
import os.path as osp
import warnings
from collections import OrderedDict
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.distributed as dist
from mmcv.runner import BaseModule, auto_fp16, load_checkpoint
from mmcv.utils import print_log
from mmdet.core import bbox2result
from mmseg.ops import resize
from mtl.model.build import build_backbone, build_neck, build_head
from mmdet.models.builder import MODELS 
from quant.lsq_plus import Conv2dLSQ, LinearLSQ, ActLSQ, QuantActLSQ # 导入 QuantActLSQ
from mmcv.parallel import DataContainer
import numpy as np
import mmcv
from mmdet.core.visualization import imshow_det_bboxes


try:
    from models.multi.cls_head.slvl_cls_head import SlvlClsHead
    from models.multi.bbox_head.dino_head import DINOHead
    from models.multi.seg_head.mask2former_head import Mask2FormerHead
except ImportError:
    pass

try:
    from mmdet.models.utils.transformer import Linear as MMDETLinear
except ImportError:
    MMDETLinear = None


# --- [新增] 辅助函数：计算单层的 TACQ 重要性分数 ---
def compute_tacq_layer_score(module, nbits_probe=4):
    """
    计算层的 TACQ 重要性分数。
    TACQ Score = Mean( |W| * |Grad| * |Q(W) - W| )
    """
    # 1. 基础检查
    if not hasattr(module, 'weight') or module.weight is None:
        return 0.0
    if module.weight.grad is None:
        # 如果没有梯度，说明该层在反向传播中未被激活或被冻结，重要性视为0
        return 0.0

    w = module.weight.data
    grad = module.weight.grad.data
    
    # 2. 模拟量化误差 (Estimation of Quantization Error)
    # 既然我们在做 4-bit vs 8-bit 的决策，我们用 4-bit 的扰动来探测敏感度
    # 使用简单的 MinMax 对称量化来估算
    Qn = -2 ** (nbits_probe - 1)
    Qp = 2 ** (nbits_probe - 1) - 1
    
    # 计算缩放因子 (Per-tensor 或 Per-channel 均可，这里用 Per-tensor 简化估计)
    scale = w.abs().max() / Qp
    # 避免除零
    scale = torch.max(scale, torch.tensor(1e-8, device=w.device))
    
    w_quant = (w / scale).round().clamp(Qn, Qp) * scale
    quant_err = (w_quant - w).abs()

    # 3. 计算 TACQ Metric: |W| * |Grad| * |Quant_Err|
    saliency_map = w.abs() * grad.abs() * quant_err
    
    # 4. 聚合为层级分数 (使用平均值)
    layer_score = saliency_map.mean().item()
    
    return layer_score

def add_prefix(log_dict, prefix):
    return {f'{prefix}.{k}': v for k, v in log_dict.items()}

supported_tasks = ['cls', 'det', 'seg']

@MODELS.register_module()
class MTL(BaseModule):
    PALETTE = None
    def __init__(self, backbone, neck, shared_encoder, cls_head=None, bbox_head=None, seg_head=None,
                 task_weight=None, train_cfg=None, test_cfg=None, init_cfg=None, quantize=False, **kwargs): 
        # 可选：打印警告以查看哪些参数被忽略了
        if kwargs:
            print_log(f"Warning: MTL.__init__ ignoring unexpected args: {kwargs}", logger='root')
        super(MTL, self).__init__(init_cfg)
        self.quantize = quantize
        self.task_map = {'cls': 0, 'det': 1, 'seg': 2}
        self.backbone = build_backbone(backbone)
        self.neck = build_neck(neck)
        from mmdet.models.utils.transformer import build_transformer_layer_sequence
        self.shared_encoder = build_transformer_layer_sequence(shared_encoder)
        self.task_weight = dict(cls=1, det=1, seg=1)
        if task_weight is not None: self.task_weight.update(task_weight)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        cls_augments_cfg = train_cfg['cls'].get('augments', None)
        if cls_augments_cfg is not None:
            from mmcls.models.utils import Augments
            self.cls_augments = Augments(cls_augments_cfg)
        if bbox_head is not None:
            bbox_head.update(train_cfg=train_cfg['det'])
            bbox_head.update(test_cfg=test_cfg['det'])
        self.task_pretrain = self.train_cfg.get('task_pretrain', None)
        
        # 强力修复通道不匹配
        if cls_head is not None: cls_head['in_channels'] = 256

        self.cls_head = build_head(cls_head, 'mmdet')
        self.bbox_head = build_head(bbox_head, 'mmdet')
        self.seg_head = build_head(seg_head, 'mmseg')

    def generate_fine_grained_mask(self, data_loader, save_path='fine_grained_mask.pth', ratio_high=0.05, default_task='cls'):
        """
        [Phase 1] TACQ 风格的敏感度分析与 Mask 生成
        修改: 支持不同任务头使用不同的 8-bit 保留比例。
        配置: Shared=10%, CLS=5%, DET=20%, SEG=5%
        """
        from mmcv.utils import print_log
        import math
        
        # ================= [修改点 1] 定义各组的量化比例配置 =================
        group_ratios = {
            'shared': 0.40,    # 共享编码器: 40%
            'cls_head': 0.05,  # 分类任务: 5%
            'det_head': 0.15,  # 目标检测: 15%
            'seg_head': 0.05   # 分割任务: 5%
        }
        # =================================================================
        
        print_log(f"\n{'='*20} [Phase 1] Starting TACQ Sensitivity Analysis {'='*20}", logger='root')
        print_log(f"混合精度策略配置 (Target 8-bit Ratios): {group_ratios}", logger='root')

        # 1. 初始化梯度累加器
        self.eval()
        self.zero_grad()
        device = next(self.parameters()).device
        
        accumulated_grads = {}
        target_modules = (nn.Conv2d, nn.Linear)
        
        for name, module in self.named_modules():
            if isinstance(module, target_modules) and module.weight.requires_grad:
                accumulated_grads[name] = torch.zeros_like(module.weight.data)

        # 2. 梯度累积循环
        print_log("正在进行多任务梯度累积 (Gradient Accumulation)...", logger='root')
        
        max_steps = 15 
        
        for step, data in enumerate(data_loader):
            if step >= max_steps: break
            
            # 解包数据
            unpacked_data = {}
            for k, v in data.items():
                if 'DataContainer' in str(type(v)): obj = v.data[0]
                else: obj = v
                if isinstance(obj, torch.Tensor): obj = obj.to(device)
                elif isinstance(obj, list): obj = [x.to(device) if isinstance(x, torch.Tensor) else x for x in obj]
                unpacked_data[k] = obj
            
            if 'task' not in unpacked_data:
                unpacked_data['task'] = default_task
            
            with torch.enable_grad():
                losses = self(**unpacked_data)
                loss, _ = self._parse_losses(losses)
                loss.backward()
            
            for name, module in self.named_modules():
                if name in accumulated_grads and module.weight.grad is not None:
                    accumulated_grads[name] += module.weight.grad.data.abs()
            
            self.zero_grad()

        print_log("梯度累积完成。开始计算 TACQ 得分...", logger='root')

        # 3. 计算 TACQ 得分并分组
        scores_dict = {
            'shared': [], 'cls_head': [], 'det_head': [], 'seg_head': []
        }
        layer_map = {} 
        layer_scores = {}
        
        skip_keywords = ['fc_reg', 'fc_cls', 'bbox_pred', 'mask_embed', 'norm', 'bias']

        for name, module in self.named_modules():
            if any(k in name for k in skip_keywords): continue
            if name not in accumulated_grads: continue
            
            sum_grad = accumulated_grads[name]
            
            if sum_grad.max() <= 1e-9:
                score = torch.zeros_like(module.weight.data).view(-1)
            else:
                w = module.weight.data
                w_abs = w.abs()
                Qp = 2**(4-1) - 1
                scale = w_abs.max() / Qp
                scale = torch.max(scale, torch.tensor(1e-5, device=device))
                w_quant = (w / scale).round().clamp(-Qp, Qp) * scale
                term_error = (w - w_quant).abs()
                score = w_abs * sum_grad * term_error
                score = score.view(-1)

            layer_scores[name] = score

            group_name = 'shared'
            if name.startswith('cls_head'): group_name = 'cls_head'
            elif name.startswith('bbox_head'): group_name = 'det_head'
            elif name.startswith('seg_head'): group_name = 'seg_head'
            
            scores_dict[group_name].append(score)
            layer_map[name] = group_name

        # 4. 计算各组阈值
        def get_threshold(score_list, ratio):
            if not score_list: return float('inf')
            all_s = torch.cat(score_list)
            if all_s.max() <= 1e-9: return float('inf')
            
            k = int(all_s.numel() * ratio)
            if k < 1: k = 1
            return torch.topk(all_s, k)[0][-1]

        thresholds = {}
        for group, scores in scores_dict.items():
            # ================= [修改点 2] 动态读取不同组的比例 =================
            current_ratio = group_ratios.get(group, ratio_high)
            # ===============================================================
            
            thresholds[group] = get_threshold(scores, current_ratio)
            print_log(f"Group [{group}] Ratio: {current_ratio*100}% | Threshold: {thresholds[group]:.4e}", logger='root')

        # 5. 保存 Mask
        mask_dict = {}
        total_params = 0
        total_8bit = 0
        
        for name, score in layer_scores.items():
            if name not in layer_map: continue
            thresh = thresholds[layer_map[name]]
            mask = (score >= thresh).float()
            
            mask_dict[name] = mask.cpu()
            total_params += mask.numel()
            total_8bit += mask.sum().item()

        print_log(f"Mask 生成完毕。全局 8-bit 比例: {total_8bit/total_params*100:.2f}%", logger='root')
        torch.save(mask_dict, save_path)
    # def generate_fine_grained_mask(self, data_loader, save_path='fine_grained_mask.pth', ratio_high=0.05, default_task='cls'):
    #     """
    #     Phase 1: 计算重要性得分。
    #     【策略升级】: 实施 4 组独立排序 (Shared, CLS, DET, SEG)。
    #     这能确保检测任务的新独立 Encoder 获得足够的高精度权重，同时不影响其他任务。
    #     """
    #     from mmcv.utils import print_log
    #     print_log(f"\n[Phase 1] 正在生成细粒度混合精度 Mask...", logger='root')
    #     print_log(f"策略: 4组独立排序 (Shared/CLS/DET/SEG)，Top {ratio_high*100}% 保留 8-bit。", logger='root')
        
    #     # 1. 计算梯度 (Calibration)
    #     self.eval()
    #     self.zero_grad()
    #     device = next(self.parameters()).device
        
    #     # 运行少量 Batch
    #     for i, data in enumerate(data_loader):
    #         if i >= 2: break 
            
    #         unpacked_data = {}
    #         for k, v in data.items():
    #             if 'DataContainer' in str(type(v)): obj = v.data[0]
    #             else: obj = v
    #             if isinstance(obj, torch.Tensor): obj = obj.to(device)
    #             elif isinstance(obj, list): obj = [x.to(device) if isinstance(x, torch.Tensor) else x for x in obj]
    #             unpacked_data[k] = obj
            
    #         if 'task' not in unpacked_data: unpacked_data['task'] = default_task
            
    #         with torch.enable_grad():
    #             losses = self(**unpacked_data)
    #             loss, _ = self._parse_losses(losses)
    #             loss.backward()
        
    #     print_log("梯度计算完成，开始分组计算得分...", logger='root')

    #     # 2. 定义分组容器
    #     scores_dict = {
    #         'shared': [],
    #         'cls_head': [],
    #         'det_head': [],
    #         'seg_head': []
    #     }
        
    #     layer_map = {} 
    #     layer_scores = {}

    #     skip_keywords = ['fc_reg', 'fc_cls', 'bbox_pred', 'mask_embed'] 

    #     for name, module in self.named_modules():
    #         if any(k in name for k in skip_keywords): continue
            
    #         if isinstance(module, (nn.Conv2d, nn.Linear)):
    #             # [修改] 强制处理无梯度层
    #             if module.weight.grad is None:
    #                 # 如果层是可训练的 (requires_grad=True) 但没有梯度 (grad is None)，
    #                 # 说明它是其他任务的层（例如检测头/分割头）。
    #                 # 我们将其重要性设为 0，从而强制它被分配为 4-bit (低精度)。
    #                 if module.weight.requires_grad:
    #                     score = torch.zeros_like(module.weight.data).view(-1)
    #                 else:
    #                     # 如果是真的被冻结的层 (Backbone Frozen Stages)，则跳过
    #                     continue
    #             else:
    #                 # 正常情况：有梯度，计算重要性 |W| * |Grad|
    #                 score = module.weight.data.abs() * module.weight.grad.data.abs()
                
    #             # 记录分数
    #             layer_scores[name] = score
    #     # for name, module in self.named_modules():
    #     #     if any(k in name for k in skip_keywords): continue
            
    #     #     if isinstance(module, (nn.Conv2d, nn.Linear)):
    #     #         if module.weight.grad is None: continue
                
    #     #         # Metric: |W| * |Grad|
    #     #         score = module.weight.data.abs() * module.weight.grad.data.abs()
    #     #         layer_scores[name] = score
                
    #             # ========================================================
    #             # 【关键逻辑】 智能分组 (4组)
    #             # ========================================================
    #             group_name = 'shared' # 默认
                
    #             if name.startswith('cls_head'):
    #                 group_name = 'cls_head'
                
    #             elif name.startswith('bbox_head'):
    #                 # 包含: 检测头本身 + 新增的独立 Encoder (现在都在 bbox_head 下)
    #                 group_name = 'det_head'
                
    #             elif name.startswith('seg_head'):
    #                 group_name = 'seg_head'
                
    #             elif any(name.startswith(p) for p in ['backbone', 'neck', 'shared_encoder']):
    #                 # Shared Encoder 现在只服务 CLS/SEG，归类为共享组
    #                 group_name = 'shared'
                
    #             scores_dict[group_name].append(score.view(-1))
    #             layer_map[name] = group_name

    #     # 3. 分别计算各组阈值
    #     def get_threshold(score_list, ratio):
    #         if not score_list: return float('inf')
    #         all_s = torch.cat(score_list) 
    #         k = int(all_s.numel() * ratio)
    #         if k < 1: k = 1
    #         return torch.topk(all_s, k)[0][-1]

    #     thresholds = {}
    #     print_log(f"{'-'*40}", logger='root')
    #     for group, scores in scores_dict.items():
    #         thresholds[group] = get_threshold(scores, ratio_high)
    #         print_log(f"Threshold [{group:<10}]: {thresholds[group]:.4e} (Params: {sum(s.numel() for s in scores)})", logger='root')
    #     print_log(f"{'-'*40}", logger='root')

    #     # 4. 生成并保存 Mask
    #     mask_dict = {}
    #     total_params = 0
    #     total_8bit = 0
        
    #     for name, score in layer_scores.items():
    #         if name not in layer_map: continue
            
    #         group = layer_map[name]
    #         thresh = thresholds[group]
            
    #         # 生成 0/1 Mask
    #         mask = (score >= thresh).float()
            
    #         mask_dict[name] = mask.cpu()
    #         total_params += mask.numel()
    #         total_8bit += mask.sum().item()

    #     print_log(f"Mask 生成完毕。全局 8-bit 比例: {total_8bit/total_params*100:.2f}%", logger='root')
    #     torch.save(mask_dict, save_path)
    #     print_log(f"Mask 已保存至: {save_path}", logger='root')

    def apply_fine_grained_quantization(self, mask_path, num_tasks=3):
        """
        Phase 2: 加载 Mask，替换层为 MixedLSQ，准备训练。
        """
        from mmcv.utils import print_log
        from quant.mixed_lsq import MixedConv2dLSQ, MixedLinearLSQ # 确保导入
        
        print_log(f"\n[Phase 2] 正在应用细粒度混合精度 (加载 Mask: {mask_path})...", logger='root')
        
        if not osp.exists(mask_path):
            raise FileNotFoundError(f"Mask file not found: {mask_path}")
            
        mask_dict = torch.load(mask_path)
        device = next(self.parameters()).device
        replaced_count = 0
        
        # 递归替换函数
        def _replace_recursive(module, parent_name=''):
            nonlocal replaced_count
            for name, child in list(module.named_children()):
                full_name = f"{parent_name}.{name}" if parent_name else name
                
                # 如果该层在 mask 字典中，则进行替换
                if full_name in mask_dict:
                    mask = mask_dict[full_name].to(device)
                    
                    new_layer = None
                    if isinstance(child, nn.Conv2d):
                        new_layer = MixedConv2dLSQ(
                            child.in_channels, child.out_channels, child.kernel_size,
                            child.stride, child.padding, child.dilation, child.groups, 
                            child.bias is not None, 
                            num_tasks=num_tasks, nbits_high=8, nbits_low=4
                        )
                    elif isinstance(child, nn.Linear):
                        new_layer = MixedLinearLSQ(
                            child.in_features, child.out_features, child.bias is not None,
                            num_tasks=num_tasks, nbits_high=8, nbits_low=4
                        )
                    
                    if new_layer:
                        # 1. 复制权重
                        new_layer.load_state_dict(child.state_dict(), strict=False)
                        # 2. 注入 Mask
                        new_layer.weight_mask = mask
                        # 3. 部署到设备
                        new_layer = new_layer.to(device)
                        
                        setattr(module, name, new_layer)
                        replaced_count += 1
                
                else:
                    _replace_recursive(child, full_name)

        _replace_recursive(self)
        print_log(f"成功替换 {replaced_count} 层为混合精度层。", logger='root')

    def set_task(self, task_id):
        for module in self.modules():
            if hasattr(module, 'change_buffer'):
                module.change_buffer(task_id)

    def apply_mixed_precision_quantization(self, data_loader, num_tasks=3, ratio_8bit=0.5, default_task='cls'):
        """
        基于 TACQ 的混合精度量化策略 (Layer-wise Mixed Precision)。
        """
        print_log(f"正在启动 TACQ 混合精度策略 (Top {ratio_8bit*100}% Layers -> 8-bit)...", logger='root')
        
        # 1. 定义黑名单
        skip_keywords = [
            'fc_reg', 'fc_cls', 'bbox_pred', 'cls_score', 'mask_embed', 'cls_head.fc',
            'patch_embed', 'absolute_pos_embed', 'pos_embed', 'backbone.stem'
        ]

        # 2. 收集梯度
        print_log("Step 1/3: 利用校准数据计算梯度...", logger='root')
        self.eval()
        self.zero_grad()
        
        # 获取模型所在的设备 (CPU or CUDA)
        device = next(self.parameters()).device
        
        calibration_batches = 2
        
        try:
            for i, data in enumerate(data_loader):
                if i >= calibration_batches: break
                
                # 解包 DataContainer 并移动到 GPU
                unpacked_data = {}
                for k, v in data.items():
                    if isinstance(v, DataContainer):
                        obj = v.data[0]
                    else:
                        obj = v
                    
                    if isinstance(obj, torch.Tensor):
                        obj = obj.to(device)
                    elif isinstance(obj, list):
                        obj = [x.to(device) if isinstance(x, torch.Tensor) else x for x in obj]
                    
                    unpacked_data[k] = obj
                
                data = unpacked_data

                if 'task' not in data:
                    data['task'] = default_task

                with torch.enable_grad():
                    losses = self(**data)
                    loss, _ = self._parse_losses(losses)
                    loss.backward()
            
            print_log("梯度计算完成。", logger='root')
            
        except Exception as e:
            print_log(f"Error during gradient calibration: {e}", logger='root')
            import traceback
            traceback.print_exc()
            print_log("CRITICAL: Calibration failed. Please check data loader.", logger='root')
            return 

        # 3. 计算分数
        print_log("Step 2/3: 计算层重要性并分配位宽...", logger='root')
        layer_scores = {}
        quantizable_layers = []

        for name, module in self.named_modules():
            if any(k in name for k in skip_keywords): continue
            if isinstance(module, (nn.Conv2d, nn.Linear, MMDETLinear if MMDETLinear else nn.Linear)):
                score = compute_tacq_layer_score(module, nbits_probe=4)
                layer_scores[name] = score
                quantizable_layers.append(name)
        
        if not layer_scores:
             print_log("Warning: No layers found with valid gradients. Skipping quantization setup.", logger='root')
             return

        sorted_layers = sorted(layer_scores.items(), key=lambda x: x[1], reverse=True)
        num_8bit = int(len(quantizable_layers) * ratio_8bit)
        layers_8bit_set = set([x[0] for x in sorted_layers[:num_8bit]])
        
        if sorted_layers:
            print_log(f"Top-1 Sensitive Layer: {sorted_layers[0][0]} (Score: {sorted_layers[0][1]:.2e})", logger='root')

        self.zero_grad()

        # 4. 执行替换并打印分配结果
        print_log("Step 3/3: 替换网络层并记录位宽...", logger='root')
        
        self.layer_bit_configs = {} 
        
        def _replace_layers_mixed(module, num_tasks, parent_name=''):
            for name, child in list(module.named_children()):
                full_name = f"{parent_name}.{name}" if parent_name else name
                
                if any(k in name for k in skip_keywords): continue

                current_w_bits = 8 if full_name in layers_8bit_set else 4
                current_a_bits = 4 

                # 替换 Linear
                if (isinstance(child, nn.Linear) or (MMDETLinear and isinstance(child, MMDETLinear))) \
                   and not isinstance(child, LinearLSQ):
                    new_linear = LinearLSQ(child.in_features, child.out_features, child.bias is not None, 
                        num_tasks=num_tasks, nbits_w=current_w_bits, nbits_a=current_a_bits)
                    new_linear.load_state_dict(child.state_dict(), strict=False)
                    
                    # [Fix] 关键修复：将新层移动到正确的设备 (GPU)
                    new_linear = new_linear.to(device) 
                    
                    setattr(module, name, new_linear)
                    self.layer_bit_configs[full_name] = current_w_bits
                
                # 替换 Conv2d
                elif isinstance(child, nn.Conv2d) and not isinstance(child, Conv2dLSQ):
                    new_conv = Conv2dLSQ(child.in_channels, child.out_channels, child.kernel_size,
                        child.stride, child.padding, child.dilation, child.groups, child.bias is not None, 
                        num_tasks=num_tasks, nbits_w=current_w_bits, nbits_a=current_a_bits)
                    new_conv.load_state_dict(child.state_dict(), strict=False)
                    
                    # [Fix] 关键修复：将新层移动到正确的设备 (GPU)
                    new_conv = new_conv.to(device)
                    
                    setattr(module, name, new_conv)
                    self.layer_bit_configs[full_name] = current_w_bits
                
                # 替换激活函数
                elif isinstance(child, (nn.ReLU, nn.GELU)) and not isinstance(child, ActLSQ):
                    new_act = QuantActLSQ(activation_cls=type(child), in_features=1, num_tasks=num_tasks, nbits_a=current_a_bits)
                    
                    # [Fix] 关键修复：将新层移动到正确的设备 (GPU)
                    new_act = new_act.to(device)
                    
                    setattr(module, name, new_act)
                
                else:
                    _replace_layers_mixed(child, num_tasks, full_name)

        _replace_layers_mixed(self, num_tasks)
        
        # 打印统计
        print_log("\n" + "="*80, logger='root')
        print_log(f"TACQ Mixed Precision Allocation Summary (Top {ratio_8bit*100:.0f}% -> 8-bit)", logger='root')
        print_log("-" * 80, logger='root')
        
        sorted_configs = sorted(self.layer_bit_configs.items())
        for lname, lbits in sorted_configs:
            print_log(f"{lname:<60} | {lbits:<10}", logger='root')
            
        print_log("-" * 80, logger='root')
        count_8bit = sum(1 for v in self.layer_bit_configs.values() if v == 8)
        count_4bit = sum(1 for v in self.layer_bit_configs.values() if v == 4)
        print_log(f"Total Quantized Layers: {len(self.layer_bit_configs)}", logger='root')
        print_log(f"8-bit Layers: {count_8bit} (High Sensitivity)", logger='root')
        print_log(f"4-bit Layers: {count_4bit} (Low Sensitivity)", logger='root')
        print_log("="*80 + "\n", logger='root')
        
        print_log("混合精度量化初始化完成。", logger='root')
    # def apply_mixed_precision_quantization(self, data_loader, num_tasks=3, ratio_8bit=0.5, default_task='cls'):
    #     """
    #     基于 TACQ 的混合精度量化策略 (Layer-wise Mixed Precision)。
    #     """
    #     print_log(f"正在启动 TACQ 混合精度策略 (Top {ratio_8bit*100}% Layers -> 8-bit)...", logger='root')
        
    #     # 1. 定义黑名单
    #     skip_keywords = [
    #         'fc_reg', 'fc_cls', 'bbox_pred', 'cls_score', 'mask_embed', 'cls_head.fc',
    #         'patch_embed', 'absolute_pos_embed', 'pos_embed', 'backbone.stem'
    #     ]

    #     # 2. 收集梯度
    #     print_log("Step 1/3: 利用校准数据计算梯度...", logger='root')
    #     self.eval()
    #     self.zero_grad()
        
    #     # 获取模型所在的设备 (CPU or CUDA)
    #     device = next(self.parameters()).device
        
    #     calibration_batches = 2
        
    #     try:
    #         for i, data in enumerate(data_loader):
    #             if i >= calibration_batches: break
                
    #             # ================= Fix Start: 解包 DataContainer 并移动到 GPU =================
    #             unpacked_data = {}
    #             for k, v in data.items():
    #                 # 1. 解包 DataContainer
    #                 if isinstance(v, DataContainer):
    #                     obj = v.data[0]
    #                 else:
    #                     obj = v
                    
    #                 # 2. 移动到设备 (GPU)
    #                 if isinstance(obj, torch.Tensor):
    #                     obj = obj.to(device)
    #                 elif isinstance(obj, list):
    #                     # 处理列表中的 Tensor (例如 img 可能是 [Tensor])
    #                     obj = [x.to(device) if isinstance(x, torch.Tensor) else x for x in obj]
                    
    #                 unpacked_data[k] = obj
                
    #             data = unpacked_data
    #             # ================= Fix End ==========================================

    #             # 注入 task 参数
    #             if 'task' not in data:
    #                 data['task'] = default_task

    #             with torch.enable_grad():
    #                 losses = self(**data)
    #                 loss, _ = self._parse_losses(losses)
    #                 loss.backward()
            
    #         print_log("梯度计算完成。", logger='root')
            
    #     except Exception as e:
    #         print_log(f"Error during gradient calibration: {e}", logger='root')
    #         import traceback
    #         traceback.print_exc() # 打印详细报错堆栈，方便排查
    #         print_log("CRITICAL: Calibration failed. Please check data loader.", logger='root')
    #         return 

    #     # 3. 计算分数
    #     print_log("Step 2/3: 计算层重要性并分配位宽...", logger='root')
    #     layer_scores = {}
    #     quantizable_layers = []

    #     for name, module in self.named_modules():
    #         if any(k in name for k in skip_keywords): continue
    #         if isinstance(module, (nn.Conv2d, nn.Linear, MMDETLinear if MMDETLinear else nn.Linear)):
    #             score = compute_tacq_layer_score(module, nbits_probe=4)
    #             layer_scores[name] = score
    #             quantizable_layers.append(name)
        
    #     # 排序与分配
    #     if not layer_scores:
    #          print_log("Warning: No layers found with valid gradients. Skipping quantization setup.", logger='root')
    #          return

    #     sorted_layers = sorted(layer_scores.items(), key=lambda x: x[1], reverse=True)
    #     num_8bit = int(len(quantizable_layers) * ratio_8bit)
    #     layers_8bit_set = set([x[0] for x in sorted_layers[:num_8bit]])
        
    #     if sorted_layers:
    #         print_log(f"Top-1 Sensitive Layer: {sorted_layers[0][0]} (Score: {sorted_layers[0][1]:.2e})", logger='root')

    #     self.zero_grad()

    #     # 4. 执行替换并打印分配结果
    #     print_log("Step 3/3: 替换网络层并记录位宽...", logger='root')
        
    #     self.layer_bit_configs = {} 
        
    #     def _replace_layers_mixed(module, num_tasks, parent_name=''):
    #         # 使用 list(module.named_children()) 避免迭代时修改字典报错
    #         for name, child in list(module.named_children()):
    #             full_name = f"{parent_name}.{name}" if parent_name else name
                
    #             if any(k in name for k in skip_keywords): continue

    #             current_w_bits = 8 if full_name in layers_8bit_set else 4
    #             current_a_bits = 4 

    #             # 替换 Linear
    #             if (isinstance(child, nn.Linear) or (MMDETLinear and isinstance(child, MMDETLinear))) \
    #                and not isinstance(child, LinearLSQ):
    #                 new_linear = LinearLSQ(child.in_features, child.out_features, child.bias is not None, 
    #                     num_tasks=num_tasks, nbits_w=current_w_bits, nbits_a=current_a_bits)
    #                 new_linear.load_state_dict(child.state_dict(), strict=False)
    #                 setattr(module, name, new_linear)
    #                 self.layer_bit_configs[full_name] = current_w_bits
                
    #             # 替换 Conv2d
    #             elif isinstance(child, nn.Conv2d) and not isinstance(child, Conv2dLSQ):
    #                 new_conv = Conv2dLSQ(child.in_channels, child.out_channels, child.kernel_size,
    #                     child.stride, child.padding, child.dilation, child.groups, child.bias is not None, 
    #                     num_tasks=num_tasks, nbits_w=current_w_bits, nbits_a=current_a_bits)
    #                 new_conv.load_state_dict(child.state_dict(), strict=False)
    #                 setattr(module, name, new_conv)
    #                 self.layer_bit_configs[full_name] = current_w_bits
                
    #             # 替换激活函数
    #             elif isinstance(child, (nn.ReLU, nn.GELU)) and not isinstance(child, ActLSQ):
    #                 new_act = QuantActLSQ(activation_cls=type(child), in_features=1, num_tasks=num_tasks, nbits_a=current_a_bits)
    #                 setattr(module, name, new_act)
                
    #             else:
    #                 _replace_layers_mixed(child, num_tasks, full_name)

    #     _replace_layers_mixed(self, num_tasks)
        
    #     # 打印最终统计报表
    #     print_log("\n" + "="*80, logger='root')
    #     print_log(f"TACQ Mixed Precision Allocation Summary (Top {ratio_8bit*100:.0f}% -> 8-bit)", logger='root')
    #     print_log("-" * 80, logger='root')
    #     print_log(f"{'Layer Name':<60} | {'Bits':<10}", logger='root')
    #     print_log("-" * 80, logger='root')
        
    #     sorted_configs = sorted(self.layer_bit_configs.items())
    #     for lname, lbits in sorted_configs:
    #         print_log(f"{lname:<60} | {lbits:<10}", logger='root')
            
    #     print_log("-" * 80, logger='root')
    #     count_8bit = sum(1 for v in self.layer_bit_configs.values() if v == 8)
    #     count_4bit = sum(1 for v in self.layer_bit_configs.values() if v == 4)
    #     print_log(f"Total Quantized Layers: {len(self.layer_bit_configs)}", logger='root')
    #     print_log(f"8-bit Layers: {count_8bit} (High Sensitivity)", logger='root')
    #     print_log(f"4-bit Layers: {count_4bit} (Low Sensitivity)", logger='root')
    #     print_log("="*80 + "\n", logger='root')
        
    #     print_log("混合精度量化初始化完成。", logger='root')
    
    
    def init_weights(self) -> None:
        super(MTL, self).init_weights()
        if hasattr(self.shared_encoder, 'layers'):
            for layer in self.shared_encoder.layers:
                for attn in layer.attentions:
                    if hasattr(attn, 'init_weights'):
                        attn.init_weights()

    def extract_feat(self, img, task=0):
        backbone_feature = self.backbone(img)
        if isinstance(backbone_feature, (tuple, list)):
            neck_feature = self.neck(backbone_feature[-3:])
        else:
            neck_feature = self.neck(backbone_feature)
        return neck_feature, backbone_feature
    
    def show_result(self, img, result, task_name=None, dataset_name=None, **kwargs):
        """
        统一的可视化入口，根据 task_name 分发给具体的画图函数
        """
        # 1. 尝试根据 task_name 分发
        if task_name == 'cls':
            return self.show_cls_result(img, result, dataset_name=dataset_name, **kwargs)
        elif task_name == 'det':
            return self.show_det_result(img, result, dataset_name=dataset_name, **kwargs)
        elif task_name == 'seg':
            return self.show_seg_result(img, result, dataset_name=dataset_name, **kwargs)
        
        # 2. 兜底逻辑 (修复 'list' has no attribute 'ndim' 的问题)
        # 检测结果通常是 list of numpy arrays
        if isinstance(result, list):
            # 如果是分割结果 (mask通常是单个大矩阵或list[tensor])，但这里主要区分检测
            # 检测结果 result[0] 是 bbox 数组 (N, 5)
            # 只要检查里面元素是不是 array 或者空 array
            if len(result) > 0 and (isinstance(result[0], np.ndarray) or len(result[0]) == 0):
                 # 这是一个不太严谨的判断，但在你的多任务结构下，如果没有 task_name，
                 # 最好还是依赖上面明确传入的 task_name。
                 # 如果非要兜底，假设 list 就是检测结果
                 return self.show_det_result(img, result, dataset_name=dataset_name, **kwargs)

        # 默认回退到分类
        return self.show_cls_result(img, result, dataset_name=dataset_name, **kwargs)

    def show_cls_result(self, img, result, dataset_name=None, out_file=None, **kwargs):
        # 1. 解析结果
        if isinstance(result, list): result = result[0]
        if isinstance(result, dict): result = result['pred_class']
        if isinstance(result, torch.Tensor): result = result.cpu().numpy()
        
        # 2. 获取类别
        class_id = np.argmax(result) if result.size > 1 else int(result)
        class_name = str(class_id)
        
        if hasattr(self, 'CLASSES') and self.CLASSES:
            labels = None
            if dataset_name and dataset_name in self.CLASSES:
                labels = self.CLASSES[dataset_name]
            elif 'resisc' in self.CLASSES:
                labels = self.CLASSES['resisc']
            
            if labels and class_id < len(labels):
                class_name = labels[class_id]

        # 3. 绘制并保存
        if out_file:
            import mmcv
            import cv2
            
            img_vis = mmcv.imread(img).copy() # 读取并复制，防止修改原图
            
            # 使用 OpenCV 绘制文字 (位置: (10, 30), 字体: 0, 大小: 1, 颜色: 红色BGR, 线宽: 2)
            # 确保 img_vis 是 contiguous array
            img_vis = np.ascontiguousarray(img_vis)
            
            text = f'Pred: {class_name}'
            cv2.putText(img_vis, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            mmcv.imwrite(img_vis, out_file)
            
        return img

    def show_det_result(self, img, result, dataset_name=None, score_thr=0.3, out_file=None, **kwargs):
        # 1. 准备类别名称
        class_names = None
        if hasattr(self, 'CLASSES') and self.CLASSES:
            if dataset_name and dataset_name in self.CLASSES:
                class_names = self.CLASSES[dataset_name]
            elif 'dior' in self.CLASSES:
                class_names = self.CLASSES['dior']
        
        # 2. 【核心修复】解包检测结果 (List -> bboxes, labels)
        if isinstance(result, tuple):
            bbox_result, segm_result = result
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]
        else:
            bbox_result, segm_result = result, None
            
        # 如果没有检测到任何框，直接保存原图并返回
        if len(bbox_result) == 0:
            if out_file:
                import mmcv
                mmcv.imwrite(img, out_file)
            return img

        # 将每个类别的 list 堆叠成一个大的 numpy 数组
        bboxes = np.vstack(bbox_result)
        
        # 生成对应的标签数组
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        
        # 3. 调用画图函数
        # 注意：这里传入的是 bboxes 和 labels，而不是 result
        img = imshow_det_bboxes(
            img,
            bboxes, 
            labels,
            class_names=class_names,
            score_thr=score_thr,
            bbox_color='green',
            text_color='green',
            thickness=2,
            font_size=10,
            win_name='',
            show=False,
            out_file=out_file
        )
        return img

    def show_seg_result(self, img, result, dataset_name=None, palette=None, opacity=0.5, out_file=None, **kwargs):
        if isinstance(result, list): result = result[0]
        seg = result
        
        if palette is None:
            if hasattr(self, 'PALETTE') and self.PALETTE is not None:
                palette = self.PALETTE
            else:
                palette = [
                    [255, 255, 255], [0, 0, 255], [0, 255, 255], 
                    [0, 255, 0], [255, 255, 0], [255, 0, 0]
                ]
        palette = np.array(palette)

        img = mmcv.imread(img)
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        
        max_label = len(palette) - 1
        seg[seg > max_label] = max_label 
        
        for label, color in enumerate(palette):
            color_seg[seg == label, :] = color
        
        color_seg = color_seg[..., ::-1]
        img = img * (1 - opacity) + color_seg * opacity
        img = img.astype(np.uint8)

        if out_file:
            mmcv.imwrite(img, out_file)
        return img

    def forward_train(self, task: str, *args, **kwargs):
        assert task in supported_tasks
        return getattr(self, f'forward_train_{task}')(*args, **kwargs)

    def forward_test(self, task, img, img_metas, *args, **kwargs):
        if isinstance(task, list):
            task = list(set(task))
            if len(task) == 1: task = task[0]
            else: raise NotImplementedError
        if isinstance(img, list): img = img[0]
        if isinstance(img_metas[0], list): img_metas = img_metas[0]
        return self.simple_test(task, img, img_metas, *args, **kwargs)

    def simple_test(self, task: 'str', *args, **kwargs):
        assert task in supported_tasks
        return getattr(self, f'simple_test_{task}')(*args, **kwargs)

    def forward_train_cls(self, img, gt_label, img_metas=None, **kwargs):
        if img_metas is None: img_metas = kwargs.get('img_metas')
        if img_metas is not None:
            batch_input_shape = tuple(img[0].size()[-2:])
            for img_meta in img_metas: img_meta['batch_input_shape'] = batch_input_shape
        if hasattr(self, 'cls_augments') and self.cls_augments is not None:
            img, gt_label = self.cls_augments(img, gt_label)
        neck_feature, backbone_feature = self.extract_feat(img)
        losses = dict()
        loss = self.cls_head.forward_train(neck_feature, backbone_feature, gt_label, self.shared_encoder, img_metas=img_metas)
        losses.update(loss)
        return losses

    # def forward_train_det(self, img, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore=None):
    #     if img_metas and 'batch_input_shape' not in img_metas[0]:
    #          for img_meta in img_metas: img_meta['batch_input_shape'] = tuple(img.size()[-2:])
    #     x = self.extract_feat(img)[0]
    #     losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore, self.shared_encoder)
    #     return losses
    # [修改] 解耦检测任务调用
    def forward_train_det(self, img, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore=None):
        if img_metas and 'batch_input_shape' not in img_metas[0]:
             for img_meta in img_metas: img_meta['batch_input_shape'] = tuple(img.size()[-2:])
        x = self.extract_feat(img)[0]
        # [修改] 不再传入 self.shared_encoder
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore)
        return losses

    def forward_train_seg(self, img, img_metas, gt_semantic_seg):
        neck_feature, backbone_feature = self.extract_feat(img)
        losses = dict()
        loss_decode = self.seg_head.forward_train(neck_feature, backbone_feature, img_metas, gt_semantic_seg, self.shared_encoder)
        losses.update(add_prefix(loss_decode, 'seg'))
        return losses

    def simple_test_cls(self, img, img_metas=None, **kwargs):
        if img_metas is not None:
            if isinstance(img, list): batch_input_shape = tuple(img[0].size()[-2:])
            else: batch_input_shape = tuple(img.size()[-2:])
            for img_meta in img_metas: img_meta['batch_input_shape'] = batch_input_shape
        neck_feature, backbone_feature = self.extract_feat(img)
        res = self.cls_head.simple_test(neck_feature, backbone_feature, self.shared_encoder, img_metas=img_metas, **kwargs)
        return res

    # def simple_test_det(self, img, img_metas, rescale=False):
    #     if img.dim() == 3: img = img.unsqueeze(0)
    #     if img_metas and 'batch_input_shape' not in img_metas[0]:
    #          for img_meta in img_metas: img_meta['batch_input_shape'] = tuple(img.size()[-2:])
    #     feat = self.extract_feat(img)[0]
    #     results_list = self.bbox_head.simple_test(feat, img_metas, rescale=rescale, shared_encoder=self.shared_encoder)
    #     bbox_results = [bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes) for det_bboxes, det_labels in results_list]
    #     return bbox_results
    # [修改] 解耦检测任务调用
    def simple_test_det(self, img, img_metas, rescale=False):
        if img.dim() == 3: img = img.unsqueeze(0)
        if img_metas and 'batch_input_shape' not in img_metas[0]:
             for img_meta in img_metas: img_meta['batch_input_shape'] = tuple(img.size()[-2:])
        feat = self.extract_feat(img)[0]
        # [修改] 不再传入 self.shared_encoder
        results_list = self.bbox_head.simple_test(feat, img_metas, rescale=rescale)
        bbox_results = [bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes) for det_bboxes, det_labels in results_list]
        return bbox_results

    def whole_inference_seg(self, img, img_meta, rescale):
        neck_feature, backbone_feature = self.extract_feat(img)
        seg_logit = self.seg_head.forward_test(neck_feature, backbone_feature, img_meta, self.shared_encoder)
        seg_logit = resize(input=seg_logit, size=img.shape[2:], mode='bilinear', align_corners=self.seg_head.align_corners)
        if rescale:
            if torch.onnx.is_in_onnx_export(): size = img.shape[2:]
            else: size = img_meta[0]['ori_shape'][:2]
            seg_logit = resize(seg_logit, size=size, mode='bilinear', align_corners=self.seg_head.align_corners, warning=False)
        return seg_logit

    def inference_seg(self, img, img_meta, rescale):
        assert self.test_cfg['seg'].mode in ['whole']
        if self.test_cfg['seg'].mode == 'whole': seg_logit = self.whole_inference_seg(img, img_meta, rescale)
        else: raise NotImplementedError
        output = F.softmax(seg_logit, dim=1)
        flip = img_meta[0]['flip']
        if flip:
            flip_direction = img_meta[0]['flip_direction']
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal': output = output.flip(dims=(3, ))
            elif flip_direction == 'vertical': output = output.flip(dims=(2, ))
        return output

    def simple_test_seg(self, img, img_meta, rescale=True):
        seg_logit = self.inference_seg(img, img_meta, rescale)
        seg_pred = seg_logit.argmax(dim=1)
        if torch.onnx.is_in_onnx_export():
            seg_pred = seg_pred.unsqueeze(0)
            return seg_pred
        seg_pred = seg_pred.cpu().numpy()
        seg_pred = list(seg_pred)
        return seg_pred

    def train_step(self, data, optimizer):
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)
        task = data.get('task', None)
        dataset_name = data.get('dataset_name', None)
        log_vars = add_prefix(log_vars, f'{task}.{dataset_name}')
        if hasattr(self, 'task_weight'):
            weight = self.task_weight[task]
            loss *= weight
            log_vars = {k: v * weight for k, v in log_vars.items()}
        outputs = dict(loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))
        return outputs

    def val_step(self, data, optimizer=None):
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)
        task = data.get('task', None)
        dataset_name = data.get('dataset_name', None)
        log_vars = add_prefix(log_vars, f'{task}.{dataset_name}')
        outputs = dict(loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))
        return outputs

    def _parse_losses(self, losses):
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else: raise TypeError(f'{loss_name} is not a tensor or list of tensors')
        loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)
        if dist.is_available() and dist.is_initialized():
            log_var_length = torch.tensor(len(log_vars), device=loss.device)
            dist.all_reduce(log_var_length)
            message = (f'rank {dist.get_rank()} len(log_vars): {len(log_vars)} keys: ' + ','.join(log_vars.keys()))
            assert log_var_length == len(log_vars) * dist.get_world_size(), 'loss log variables are different across GPUs!\n' + message
        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()
        return loss, log_vars

    def load_task_pretrain(self): pass
    
    @auto_fp16(apply_to=('img',))
    def forward(self, task, img, img_metas, return_loss=True,
                dataset_name=None, **kwargs):
        if torch.onnx.is_in_onnx_export():
            assert len(img_metas) == 1
            return self.onnx_export(img[0], img_metas[0])

        # ================== 【新增/修改的代码】 ==================
        # 1. 优先处理 task 是 list 的情况 (解包)
        # 测试时 DataLoader 会把 string 包装成 list，例如 ['cls']
        if isinstance(task, list):
            # 简单取第一个元素，因为这里假设 batch 内任务一致
            task = task[0] 
        
        # 2. 再执行量化任务切换逻辑
        if getattr(self, 'quantize', False):
            # 现在的 task 已经是 string 了，可以用作 key
            task_id = self.task_map[task]
            self.set_task(task_id)
        # ========================================================

        if return_loss:
            return self.forward_train(
                task=task, img=img, img_metas=img_metas, **kwargs)
        else:
            return self.forward_test(
                task=task, img=img, img_metas=img_metas, **kwargs)
    # @auto_fp16(apply_to=('img',))
    # def forward(self, task, img, img_metas, return_loss=True, dataset_name=None, **kwargs):
    #     if torch.onnx.is_in_onnx_export():
    #         assert len(img_metas) == 1
    #         return self.onnx_export(img[0], img_metas[0])
    #     if self.quantize:
    #         task_id = self.task_map[task]
    #         self.set_task(task_id)
    #     if return_loss:
    #         return self.forward_train(task=task, img=img, img_metas=img_metas, **kwargs)
    #     else:
    #         return self.forward_test(task=task, img=img, img_metas=img_metas, **kwargs)


# import warnings
# from collections import OrderedDict

# import numpy as np
# import matplotlib.font_manager as fm
# import matplotlib.pyplot as plt
# from matplotlib.collections import PatchCollection
# from matplotlib.patches import Polygon

# import torch
# from torch import Tensor
# import torch.nn.functional as F
# import torch.distributed as dist

# import mmcv
# from mmcv.runner import BaseModule, auto_fp16
# from mmcv.cnn.bricks.transformer import (build_transformer_layer_sequence,
#                                          MultiScaleDeformableAttention)
# from mmcv.cnn import MODELS

# from mmcls.models.utils.augment import Augments
# from mmdet.core import bbox2result
# from mmseg.core import add_prefix
# from mmseg.ops import resize

# from mmdet.core.visualization import color_val_matplotlib

# from mtl.model.build import build_backbone, build_neck, build_head


# supported_tasks = ('cls', 'det', 'seg')


# @MODELS.register_module()
# class MTL(BaseModule):
#     PALETTE = None
#     def __init__(self,
#                  backbone,
#                  neck,
#                  shared_encoder,
#                  cls_head=None,
#                  bbox_head=None,
#                  seg_head=None,
#                  task_weight=None,
#                  train_cfg=None,
#                  test_cfg=None,
#                  init_cfg=None):
#         super(MTL, self).__init__(init_cfg)
#         self.backbone = build_backbone(backbone)
#         self.neck = build_neck(neck)
#         self.shared_encoder = build_transformer_layer_sequence(shared_encoder)

#         self.task_weight = dict(cls=1, det=1, seg=1)
#         if task_weight is not None:
#             assert isinstance(task_weight, dict)
#             self.task_weight.update(task_weight)

#         self.train_cfg = train_cfg
#         self.test_cfg = test_cfg
#         cls_augments_cfg = train_cfg['cls'].get('augments', None)
#         if cls_augments_cfg is not None:
#             self.cls_augments = Augments(cls_augments_cfg)
#         bbox_head.update(train_cfg=train_cfg['det'])
#         bbox_head.update(test_cfg=test_cfg['det'])
#         # seg_head.update(train_cfg=train_cfg['seg'])
#         # seg_head.update(test_cfg=test_cfg['seg'])
#         self.task_pretrain = self.train_cfg.get('task_pretrain', None)

#         self.cls_head = build_head(cls_head, 'mmcls')
#         self.bbox_head = build_head(bbox_head, 'mmdet')
#         self.seg_head = build_head(seg_head, 'mmseg')

#     def init_weights(self) -> None:
#         super(MTL, self).init_weights()
#         # init_weights defined in MultiScaleDeformableAttention
#         for layer in self.shared_encoder.layers:
#             for attn in layer.attentions:
#                 if isinstance(attn, MultiScaleDeformableAttention):
#                     attn.init_weights()

#     def extract_feat(self, img):
#         """Directly extract features from the backbone+neck."""
#         backbone_feature = self.backbone(img)
#         neck_feature = self.neck(backbone_feature[-3:])
#         return neck_feature, backbone_feature

#     def forward_train(self, task: str, *args, **kwargs):
#         assert task in supported_tasks
#         return getattr(self, f'forward_train_{task}')(*args, **kwargs)

#     def forward_test(self,
#                      task: 'str, List[str]',
#                      img: 'Tensor, List[Tensor]',
#                      img_metas: list,
#                      *args,
#                      **kwargs):
#         if isinstance(task, list):
#             task = list(set(task))
#             if len(task) == 1:
#                 task = task[0]
#             else:
#                 raise NotImplementedError(
#                     'The current implementation only '
#                     'support same task in a batch')
#         if isinstance(img, list):
#             num_augs = len(img)
#             if num_augs != 1:
#                 raise NotImplementedError(
#                     'The current implementation does not support TTA ')
#             img = img[0]
#         if isinstance(img_metas[0], list):
#             img_metas = img_metas[0]
#         return self.simple_test(task, img, img_metas, *args, **kwargs)

#     def simple_test(self, task: 'str', *args, **kwargs):
#         assert task in supported_tasks
#         return getattr(self, f'simple_test_{task}')(*args, **kwargs)

#     def forward_train_cls(self, img, gt_label, **kwargs):
#         if self.cls_augments is not None:
#             img, gt_label = self.cls_augments(img, gt_label)
#         neck_feature, backbone_feature = self.extract_feat(img)
#         losses = dict()
#         loss = self.cls_head.forward_train(
#             neck_feature, backbone_feature, gt_label, self.shared_encoder)
#         losses.update(loss)
#         return losses

#     def forward_train_det(self, img, img_metas, gt_bboxes, gt_labels,
#                           gt_bboxes_ignore=None):
#         batch_input_shape = tuple(img[0].size()[-2:])
#         for img_meta in img_metas:
#             img_meta['batch_input_shape'] = batch_input_shape
#         x = self.extract_feat(img)[0]
#         losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
#                                               gt_labels, gt_bboxes_ignore,
#                                               self.shared_encoder)
#         return losses

#     def forward_train_seg(self, img, img_metas, gt_semantic_seg):
#         neck_feature, backbone_feature = self.extract_feat(img)
#         losses = dict()
#         loss_decode = self.seg_head.forward_train(
#             neck_feature, backbone_feature, img_metas,
#             gt_semantic_seg, self.shared_encoder)
#         losses.update(add_prefix(loss_decode, 'seg'))
#         return losses

#     def simple_test_cls(self, img, img_metas=None, **kwargs):
#         """Test without augmentation."""
#         neck_feature, backbone_feature = self.extract_feat(img)
#         res = self.cls_head.simple_test(
#             neck_feature, backbone_feature,
#             shared_encoder=self.shared_encoder, **kwargs)
#         return res

#     def simple_test_det(self, img, img_metas, rescale=False):
#         batch_size = len(img_metas)
#         for img_id in range(batch_size):
#             img_metas[img_id]['batch_input_shape'] = tuple(img.size()[-2:])
#         feat = self.extract_feat(img)[0]
#         results_list = self.bbox_head.simple_test(
#             feat, img_metas, rescale=rescale,
#             shared_encoder=self.shared_encoder)
#         bbox_results = [
#             bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
#             for det_bboxes, det_labels in results_list
#         ]
#         return bbox_results

#     def whole_inference_seg(self, img, img_meta, rescale):
#         neck_feature, backbone_feature = self.extract_feat(img)
#         seg_logit = self.seg_head.forward_test(
#             neck_feature, backbone_feature, img_meta, self.shared_encoder)
#         seg_logit = resize(
#             input=seg_logit,
#             size=img.shape[2:],
#             mode='bilinear',
#             align_corners=self.seg_head.align_corners)
#         if rescale:
#             # support dynamic shape for onnx
#             if torch.onnx.is_in_onnx_export():
#                 size = img.shape[2:]
#             else:
#                 # remove padding area
#                 resize_shape = img_meta[0]['img_shape'][:2]
#                 seg_logit = seg_logit[:, :, :resize_shape[0], :resize_shape[1]]
#                 size = img_meta[0]['ori_shape'][:2]
#             seg_logit = resize(
#                 seg_logit,
#                 size=size,
#                 mode='bilinear',
#                 align_corners=self.seg_head.align_corners,
#                 warning=False)
#         return seg_logit

#     def inference_seg(self, img, img_meta, rescale):
#         assert self.test_cfg['seg'].mode in ['whole']
#         ori_shape = img_meta[0]['ori_shape']
#         assert all(_['ori_shape'] == ori_shape for _ in img_meta)
#         if self.test_cfg['seg'].mode == 'whole':
#             seg_logit = self.whole_inference_seg(img, img_meta, rescale)
#         else:
#             raise NotImplementedError
#         output = F.softmax(seg_logit, dim=1)
#         flip = img_meta[0]['flip']
#         if flip:
#             flip_direction = img_meta[0]['flip_direction']
#             assert flip_direction in ['horizontal', 'vertical']
#             if flip_direction == 'horizontal':
#                 output = output.flip(dims=(3, ))
#             elif flip_direction == 'vertical':
#                 output = output.flip(dims=(2, ))

#         return output

#     def simple_test_seg(self, img, img_meta, rescale=True):
#         seg_logit = self.inference_seg(img, img_meta, rescale)
#         seg_pred = seg_logit.argmax(dim=1)
#         if torch.onnx.is_in_onnx_export():
#             # our inference backend only support 4D output
#             seg_pred = seg_pred.unsqueeze(0)
#             return seg_pred
#         seg_pred = seg_pred.cpu().numpy()
#         # unravel batch dim
#         seg_pred = list(seg_pred)
#         return seg_pred

#     def train_step(self, data, optimizer):
#         losses = self(**data)
#         loss, log_vars = self._parse_losses(losses)

#         task = data.get('task', None)
#         dataset_name = data.get('dataset_name', None)
#         log_vars = add_prefix(log_vars, f'{task}.{dataset_name}')

#         if hasattr(self, 'task_weight'):
#             weight = self.task_weight[task]
#             loss *= weight
#             log_vars = {k: v * weight for k, v in log_vars.items()}

#         outputs = dict(
#             loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

#         return outputs

#     def val_step(self, data, optimizer=None):
#         losses = self(**data)
#         loss, log_vars = self._parse_losses(losses)

#         task = data.get('task', None)
#         dataset_name = data.get('dataset_name', None)
#         log_vars = add_prefix(log_vars, f'{task}.{dataset_name}')

#         outputs = dict(
#             loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

#         return outputs

#     @auto_fp16(apply_to=('img',))
#     def forward(self, task, img, img_metas, return_loss=True,
#                 dataset_name=None, **kwargs):
#         if torch.onnx.is_in_onnx_export():
#             assert len(img_metas) == 1
#             return self.onnx_export(img[0], img_metas[0])

#         if return_loss:
#             return self.forward_train(
#                 task=task, img=img, img_metas=img_metas, **kwargs)
#         else:
#             return self.forward_test(
#                 task=task, img=img, img_metas=img_metas, **kwargs)

#     def _parse_losses(self, losses):
#         log_vars = OrderedDict()
#         for loss_name, loss_value in losses.items():
#             if isinstance(loss_value, torch.Tensor):
#                 log_vars[loss_name] = loss_value.mean()
#             elif isinstance(loss_value, list):
#                 log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
#             else:
#                 raise TypeError(
#                     f'{loss_name} is not a tensor or list of tensors')

#         loss = sum(_value for _key, _value in log_vars.items()
#                    if 'loss' in _key)

#         # If the loss_vars has different length, GPUs will wait infinitely
#         if dist.is_available() and dist.is_initialized():
#             log_var_length = torch.tensor(len(log_vars), device=loss.device)
#             dist.all_reduce(log_var_length)
#             message = (f'rank {dist.get_rank()}' +
#                        f' len(log_vars): {len(log_vars)}' + ' keys: ' +
#                        ','.join(log_vars.keys()))
#             assert log_var_length == len(log_vars) * dist.get_world_size(), \
#                 'loss log variables are different across GPUs!\n' + message

#         log_vars['loss'] = loss
#         for loss_name, loss_value in log_vars.items():
#             # reduce loss when distributed training
#             if dist.is_available() and dist.is_initialized():
#                 loss_value = loss_value.data.clone()
#                 dist.all_reduce(loss_value.div_(dist.get_world_size()))
#             log_vars[loss_name] = loss_value.item()

#         return loss, log_vars

#     def load_task_pretrain(self):

#         def get_mapped_name(name: str) -> str:
#             new_name = name
#             if new_name.startswith('bbox_head.transformer.encoder'):
#                 new_name = new_name.replace('bbox_head.transformer.encoder',
#                                             'shared_encoder')
#             return new_name

#         def mapping_state_dict(state_dict: OrderedDict) -> OrderedDict:
#             out = OrderedDict()
#             for name, param in state_dict.items():
#                 new_name = get_mapped_name(name)
#                 assert new_name not in out, f'{name}-->{new_name}'
#                 out[new_name] = param
#             return out

#         def delete_neck_convs_bias(state_dict: OrderedDict) -> OrderedDict:
#             out = OrderedDict()
#             for name, param in state_dict.items():
#                 if name.startswith('neck') and name.endswith('conv.bias'):
#                     continue
#                 out[name] = param
#             return out

#         if self.task_pretrain is None:
#             print('#######################################\n'
#                   'You did not set task_pretrain, hence it is skipped.'
#                   '#######################################\n')
#             return

#         rule = self.task_pretrain.get('rule', None)
#         pretrainded = self.task_pretrain['pretrained']
#         # dm: -> dino_mmdet
#         sd = torch.load(pretrainded)
#         if 'state_dict' in sd:
#             sd = sd['state_dict']
#         if rule == 'dino_mmdet':
#             sd = delete_neck_convs_bias(sd)
#             sd = mapping_state_dict(sd)
#         incompatiblekeys = self.load_state_dict(sd, strict=False)
#         print('#######################################\n'
#               f'load task pretrain of rule:{rule}\n'
#               f'ckpt path: {pretrainded}\n'
#               f'incompatiblekeys: {incompatiblekeys}\n'
#               '#######################################\n')

#     def show_result(self, img, pred, *args, **kwargs):
#         if 'pred_class' in pred:  # cls
#             return self.show_cls_result(img, pred, *args, **kwargs)
#         elif isinstance(pred, list):
#             if len(pred) > 1 or isinstance(pred[0], list):
#                 pred = pred[0] if isinstance(pred[0], list) else pred
#                 kwargs = {k: v for k, v in kwargs.items() if v is not None}
#                 kwargs['show_with_gt'] = 'annotation' in kwargs
#                 return self.show_det_result(img, pred, *args, **kwargs)
#             elif pred[0].ndim > 1:
#                 return self.show_seg_result(img, pred, *args, **kwargs)
#             else:
#                 return self.show_cls_result(img, pred, *args, **kwargs)
#         else:
#             raise ValueError()

#     def show_cls_result(self,
#                         img,
#                         result,
#                         **kwargs):
#         if isinstance(result, list):
#             result = result[0]
#         if isinstance(result, dict):
#             assert 'pred_class' in result
#             result = result['pred_class']
#         # print(np.argmax(result))

#     def show_det_result(self,
#                         img,
#                         result,
#                         score_thr=0.3,
#                         bbox_color=(255, 110, 110),
#                         text_color='black',
#                         # text_color=(72, 101, 241),
#                         mask_color=None,
#                         thickness=2,
#                         font_size=15,
#                         win_name='',
#                         show=False,
#                         wait_time=0,
#                         out_file=None,
#                         show_with_gt=True,
#                         annotation=None):
#         """Draw `result` over `img`.

#         Args:
#             img (str or Tensor): The image to be displayed.
#             result (Tensor or tuple): The results to draw over `img`
#                 bbox_result or (bbox_result, segm_result).
#             score_thr (float, optional): Minimum score of bboxes to be shown.
#                 Default: 0.3.
#             bbox_color (str or tuple(int) or :obj:`Color`):Color of bbox lines.
#                The tuple of color should be in BGR order. Default: 'green'
#             text_color (str or tuple(int) or :obj:`Color`):Color of texts.
#                The tuple of color should be in BGR order. Default: 'green'
#             mask_color (None or str or tuple(int) or :obj:`Color`):
#                Color of masks. The tuple of color should be in BGR order.
#                Default: None
#             thickness (int): Thickness of lines. Default: 2
#             font_size (int): Font size of texts. Default: 13
#             win_name (str): The window name. Default: ''
#             wait_time (float): Value of waitKey param.
#                 Default: 0.
#             show (bool): Whether to show the image.
#                 Default: False.
#             out_file (str or None): The filename to write the image.
#                 Default: None.

#         Returns:
#             img (Tensor): Only if not `show` or `out_file`
#         """
#         img = mmcv.imread(img)
#         img = img.copy()
#         if isinstance(result, tuple):
#             bbox_result, segm_result = result
#             if isinstance(segm_result, tuple):
#                 segm_result = segm_result[0]  # ms rcnn
#         else:
#             bbox_result, segm_result = result, None
#         bboxes = np.vstack(bbox_result)
#         labels = [
#             np.full(bbox.shape[0], i, dtype=np.int32)
#             for i, bbox in enumerate(bbox_result)
#         ]
#         labels = np.concatenate(labels)
#         # draw segmentation masks
#         segms = None
#         if segm_result is not None and len(labels) > 0:  # non empty
#             segms = mmcv.concat_list(segm_result)
#             if isinstance(segms[0], torch.Tensor):
#                 segms = torch.stack(segms, dim=0).detach().cpu().numpy()
#             else:
#                 segms = np.stack(segms, axis=0)
#         # if out_file specified, do not show image in window
#         if out_file is not None:
#             show = False
#         # draw bounding boxes  # Modified by LQY
#         kwargs = dict(class_names=self.CLASSES['dior'],
#                       score_thr=score_thr,
#                       thickness=thickness,
#                       font_size=font_size,
#                       win_name=win_name,
#                       show=show,
#                       wait_time=wait_time,
#                       out_file=out_file)
#         if not show_with_gt:
#             args = [img, bboxes, labels, segms]
#             kwargs["bbox_color"] = bbox_color
#             kwargs["text_color"] = text_color
#             kwargs["mask_color"] = mask_color
#             imshow_results = imshow_det_bboxes
#         else:
#             args = [img, annotation, result]
#             det_color = tuple([v for v in bbox_color[::-1]])
#             kwargs["det_bbox_color"] = det_color
#             kwargs["det_text_color"] = 'black'
#             kwargs["det_mask_color"] = det_color
#             gt_color = tuple([v * 255 for v in (0.09,0.78,1)[::-1]])
#             kwargs["gt_bbox_color"] = gt_color
#             kwargs["gt_text_color"] = gt_color
#             kwargs["gt_mask_color"] = gt_color
#             kwargs["face_alpha"] = 0.9
#             imshow_results = imshow_gt_det_bboxes
#         img = imshow_results(*args, **kwargs)

#         if not (show or out_file):
#             return img

#     def show_seg_result(self,
#                         img,
#                         result,
#                         palette=None,
#                         win_name='',
#                         show=False,
#                         wait_time=0,
#                         out_file=None,
#                         **kwargs):
#         """Draw `result` over `img`.

#         Args:
#             img (str or Tensor): The image to be displayed.
#             result (Tensor): The semantic segmentation results to draw over
#                 `img`.
#             palette (list[list[int]]] | np.ndarray | None): The palette of
#                 segmentation map. If None is given, random palette will be
#                 generated. Default: None
#             win_name (str): The window name.
#             wait_time (int): Value of waitKey param.
#                 Default: 0.
#             show (bool): Whether to show the image.
#                 Default: False.
#             out_file (str or None): The filename to write the image.
#                 Default: None.

#         Returns:
#             img (Tensor): Only if not `show` or `out_file`
#         """
#         opacity = 1
#         """
#         opacity(float): Opacity of painted segmentation map.
#                 Default 0.5.
#                 Must be in (0, 1] range.
#         """

#         CLASSES = self.CLASSES['potsdam']

#         img = mmcv.imread(img)
#         img = img.copy()
#         seg = result[0]
#         if 5 in seg:
#             seg += (seg == 5).astype(seg.dtype) * -5
#         if palette is None:
#             if self.PALETTE is None:
#                 # Get random state before set seed,
#                 # and restore random state later.
#                 # It will prevent loss of randomness, as the palette
#                 # may be different in each iteration if not specified.
#                 # See: https://github.com/open-mmlab/mmdetection/issues/5844
#                 state = np.random.get_state()
#                 np.random.seed(42)
#                 # random palette
#                 palette = np.random.randint(
#                     0, 255, size=(len(CLASSES), 3))
#                 np.random.set_state(state)
#             else:
#                 palette = self.PALETTE
#         palette = np.array(palette)
#         assert palette.shape[0] == len(CLASSES)
#         assert palette.shape[1] == 3
#         assert len(palette.shape) == 2
#         assert 0 < opacity <= 1.0
#         color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
#         for label, color in enumerate(palette):
#             color_seg[seg == label, :] = color
#         # convert to BGR
#         color_seg = color_seg[..., ::-1]

#         img = img * (1 - opacity) + color_seg * opacity
#         img = img.astype(np.uint8)
#         # if out_file specified, do not show image in window
#         if out_file is not None:
#             show = False

#         if show:
#             mmcv.imshow(img, win_name, wait_time)
#         if out_file is not None:
#             mmcv.imwrite(img, out_file)

#         if not (show or out_file):
#             warnings.warn('show==False and out_file is not specified, only '
#                           'result image will be returned')
#             return img


# def imshow_det_bboxes(img,
#                       bboxes,
#                       labels,
#                       segms=None,
#                       class_names=None,
#                       score_thr=0,
#                       bbox_color='green',
#                       text_color='green',
#                       mask_color=None,
#                       thickness=2,
#                       font_size=13,
#                       win_name='',
#                       show=True,
#                       wait_time=0,
#                       out_file=None,
#                       with_text=True,
#                       face_color='black',
#                       face_alpha=0.4):  # Modified by LQY
#     """Draw bboxes and class labels (with scores) on an image.

#     Args:
#         img (str or ndarray): The image to be displayed.
#         bboxes (ndarray): Bounding boxes (with scores), shaped (n, 4) or
#             (n, 5).
#         labels (ndarray): Labels of bboxes.
#         segms (ndarray or None): Masks, shaped (n,h,w) or None
#         class_names (list[str]): Names of each classes.
#         score_thr (float): Minimum score of bboxes to be shown.  Default: 0
#         bbox_color (str or tuple(int) or :obj:`Color`):Color of bbox lines.
#            The tuple of color should be in BGR order. Default: 'green'
#         text_color (str or tuple(int) or :obj:`Color`):Color of texts.
#            The tuple of color should be in BGR order. Default: 'green'
#         mask_color (str or tuple(int) or :obj:`Color`, optional):
#            Color of masks. The tuple of color should be in BGR order.
#            Default: None
#         thickness (int): Thickness of lines. Default: 2
#         font_size (int): Font size of texts. Default: 13
#         show (bool): Whether to show the image. Default: True
#         win_name (str): The window name. Default: ''
#         wait_time (float): Value of waitKey param. Default: 0.
#         out_file (str, optional): The filename to write the image.
#             Default: None

#     Returns:
#         ndarray: The image with bboxes drawn on it.
#     """

#     myfont = fm.FontProperties(fname='./times.ttf')

#     assert bboxes.ndim == 2, \
#         f' bboxes ndim should be 2, but its ndim is {bboxes.ndim}.'
#     assert labels.ndim == 1, \
#         f' labels ndim should be 1, but its ndim is {labels.ndim}.'
#     assert bboxes.shape[0] == labels.shape[0], \
#         'bboxes.shape[0] and labels.shape[0] should have the same length.'
#     assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5, \
#         f' bboxes.shape[1] should be 4 or 5, but its {bboxes.shape[1]}.'
#     img = mmcv.imread(img).astype(np.uint8)

#     if score_thr > 0:
#         assert bboxes.shape[1] == 5
#         scores = bboxes[:, -1]
#         inds = scores > score_thr
#         bboxes = bboxes[inds, :]
#         labels = labels[inds]
#         if segms is not None:
#             segms = segms[inds, ...]

#     mask_colors = []
#     if labels.shape[0] > 0:
#         if mask_color is None:
#             # Get random state before set seed, and restore random state later.
#             # Prevent loss of randomness.
#             # See: https://github.com/open-mmlab/mmdetection/issues/5844
#             state = np.random.get_state()
#             # random color
#             np.random.seed(42)
#             mask_colors = [
#                 np.random.randint(0, 256, (1, 3), dtype=np.uint8)
#                 for _ in range(max(labels) + 1)
#             ]
#             np.random.set_state(state)
#         else:
#             # specify  color
#             mask_colors = [
#                 np.array(mmcv.color_val(mask_color)[::-1], dtype=np.uint8)
#             ] * (
#                 max(labels) + 1)

#     bbox_color = color_val_matplotlib(bbox_color)
#     text_color = color_val_matplotlib(text_color)
#     # text_color = (0,0,0)

#     img = mmcv.bgr2rgb(img)
#     width, height = img.shape[1], img.shape[0]
#     img = np.ascontiguousarray(img)

#     fig = plt.figure(win_name, frameon=False)
#     plt.title(win_name)
#     canvas = fig.canvas
#     dpi = fig.get_dpi()
#     # add a small EPS to avoid precision lost due to matplotlib's truncation
#     # (https://github.com/matplotlib/matplotlib/issues/15363)
#     EPS = 1e-2
#     fig.set_size_inches((width + EPS) / dpi, (height + EPS) / dpi)

#     # remove white edges by set subplot margin
#     plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
#     ax = plt.gca()
#     ax.axis('off')

#     polygons = []
#     color = []
#     for i, (bbox, label) in enumerate(zip(bboxes, labels)):
#         bbox_int = bbox.astype(np.int32)
#         poly = [[bbox_int[0], bbox_int[1]], [bbox_int[0], bbox_int[3]],
#                 [bbox_int[2], bbox_int[3]], [bbox_int[2], bbox_int[1]]]
#         np_poly = np.array(poly).reshape((4, 2))
#         polygons.append(Polygon(np_poly))
#         color.append(bbox_color)
#         label_text = class_names[
#             label] if class_names is not None else f'class {label}'
#         if len(bbox) > 4:
#             label_text += f'|{bbox[-1]:.02f}'
#         if with_text:
#             ax.text(
#                 bbox_int[0],
#                 # bbox_int[1],
#                 bbox_int[1],
#                 f'{label_text}',
#                 bbox={
#                     'facecolor': face_color,
#                     'alpha': face_alpha,
#                     'pad': 0.7,
#                     'edgecolor': 'none'
#                 },
#                 color=text_color,
#                 fontsize=font_size,
#                 verticalalignment='bottom',
#                 horizontalalignment='left',
#                 fontproperties = myfont)
#         if segms is not None:
#             color_mask = mask_colors[labels[i]]
#             mask = segms[i].astype(bool)
#             img[mask] = img[mask] * 0.5 + color_mask * 0.5

#     plt.imshow(img)

#     p = PatchCollection(
#         polygons, facecolor='none', edgecolors=color, linewidths=thickness)
#     ax.add_collection(p)

#     stream, _ = canvas.print_to_buffer()
#     buffer = np.frombuffer(stream, dtype='uint8')
#     img_rgba = buffer.reshape(height, width, 4)
#     rgb, alpha = np.split(img_rgba, [3], axis=2)
#     img = rgb.astype('uint8')
#     img = mmcv.rgb2bgr(img)

#     if show:
#         # We do not use cv2 for display because in some cases, opencv will
#         # conflict with Qt, it will output a warning: Current thread
#         # is not the object's thread. You can refer to
#         # https://github.com/opencv/opencv-python/issues/46 for details
#         if wait_time == 0:
#             plt.show()
#         else:
#             plt.show(block=False)
#             plt.pause(wait_time)
#     if out_file is not None:
#         mmcv.imwrite(img, out_file)

#     plt.close()

#     return img


# def imshow_gt_det_bboxes(img,
#                          annotation,
#                          result,
#                          class_names=None,
#                          score_thr=0,
#                          gt_bbox_color=(255, 102, 61),
#                          gt_text_color=(255, 102, 61),
#                          gt_mask_color=(255, 102, 61),
#                          det_bbox_color=(72, 101, 241),
#                          det_text_color=(72, 101, 241),
#                          det_mask_color=(72, 101, 241),
#                          thickness=2,
#                          font_size=13,
#                          win_name='',
#                          show=True,
#                          wait_time=0,
#                          out_file=None,
#                          face_color='black',
#                          face_alpha=0.4):
#     """General visualization GT and result function.

#     Args:
#       img (str or ndarray): The image to be displayed.)
#       annotation (dict): Ground truth annotations where contain keys of
#           'gt_bboxes' and 'gt_labels' or 'gt_masks'
#       result (tuple[list] or list): The detection result, can be either
#           (bbox, segm) or just bbox.
#       class_names (list[str]): Names of each classes.
#       score_thr (float): Minimum score of bboxes to be shown.  Default: 0
#       gt_bbox_color (str or tuple(int) or :obj:`Color`):Color of bbox lines.
#            The tuple of color should be in BGR order. Default: (255, 102, 61)
#       gt_text_color (str or tuple(int) or :obj:`Color`):Color of texts.
#            The tuple of color should be in BGR order. Default: (255, 102, 61)
#       gt_mask_color (str or tuple(int) or :obj:`Color`, optional):
#            Color of masks. The tuple of color should be in BGR order.
#            Default: (255, 102, 61)
#       det_bbox_color (str or tuple(int) or :obj:`Color`):Color of bbox lines.
#            The tuple of color should be in BGR order. Default: (72, 101, 241)
#       det_text_color (str or tuple(int) or :obj:`Color`):Color of texts.
#            The tuple of color should be in BGR order. Default: (72, 101, 241)
#       det_mask_color (str or tuple(int) or :obj:`Color`, optional):
#            Color of masks. The tuple of color should be in BGR order.
#            Default: (72, 101, 241)
#       thickness (int): Thickness of lines. Default: 2
#       font_size (int): Font size of texts. Default: 13
#       win_name (str): The window name. Default: ''
#       show (bool): Whether to show the image. Default: True
#       wait_time (float): Value of waitKey param. Default: 0.
#       out_file (str, optional): The filename to write the image.
#          Default: None

#     Returns:
#         ndarray: The image with bboxes or masks drawn on it.
#     """
#     assert 'gt_bboxes' in annotation
#     assert 'gt_labels' in annotation
#     assert isinstance(
#         result,
#         (tuple, list)), f'Expected tuple or list, but get {type(result)}'

#     gt_masks = annotation.get('gt_masks', None)
#     if gt_masks is not None:
#         gt_masks = mask2ndarray(gt_masks)

#     img = mmcv.imread(img)

#     img = imshow_det_bboxes(
#         img,
#         annotation['gt_bboxes'],
#         annotation['gt_labels'],
#         gt_masks,
#         class_names=class_names,
#         bbox_color=gt_bbox_color,
#         text_color=gt_text_color,
#         mask_color=gt_mask_color,
#         thickness=thickness,
#         font_size=font_size,
#         win_name=win_name,
#         show=False,
#         with_text=False)

#     if isinstance(result, tuple):
#         bbox_result, segm_result = result
#         if isinstance(segm_result, tuple):
#             segm_result = segm_result[0]  # ms rcnn
#     else:
#         bbox_result, segm_result = result, None

#     bboxes = np.vstack(bbox_result)
#     labels = [
#         np.full(bbox.shape[0], i, dtype=np.int32)
#         for i, bbox in enumerate(bbox_result)
#     ]
#     labels = np.concatenate(labels)

#     segms = None
#     if segm_result is not None and len(labels) > 0:  # non empty
#         segms = mmcv.concat_list(segm_result)
#         segms = mask_util.decode(segms)
#         segms = segms.transpose(2, 0, 1)

#     # img = imshow_gt_det_bboxes(
#     img = imshow_det_bboxes(
#         img,
#         bboxes,
#         labels,
#         segms=segms,
#         class_names=class_names,
#         score_thr=score_thr,
#         bbox_color=det_bbox_color,
#         text_color=det_text_color,
#         mask_color=det_mask_color,
#         thickness=thickness,
#         font_size=font_size,
#         win_name=win_name,
#         show=show,
#         wait_time=wait_time,
#         out_file=out_file,
#         face_color=tuple([v / 255.0 for v in det_bbox_color[::-1]] + [1.]),
#         face_alpha=face_alpha)
#     return img


# if __name__ == '__main__':
#     from mmcv import Config
#     config = Config.fromfile('configs/multi/DINO-MTL_swin-t-p4-w7_1x1.py')
#     model = MTL(**config.model)
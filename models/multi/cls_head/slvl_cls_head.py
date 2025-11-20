import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule
from mmcv.utils import print_log
from mmdet.models.builder import HEADS
from mmdet.models.utils import SinePositionalEncoding

def get_reference_points(spatial_shapes, valid_ratios, device):
    """
    手动生成 Reference Points (B, Total_L, n_levels, 2)
    """
    reference_points_list = []
    for lvl, (H, W) in enumerate(spatial_shapes):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H - 0.5, H, dtype=torch.float32, device=device),
            torch.linspace(0.5, W - 0.5, W, dtype=torch.float32, device=device)
        )
        ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H)
        ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W)
        ref = torch.stack((ref_x, ref_y), -1)
        reference_points_list.append(ref)
    
    reference_points = torch.cat(reference_points_list, 1)
    reference_points = reference_points[:, :, None] * valid_ratios[:, None]
    return reference_points

@HEADS.register_module()
class SlvlClsHead(BaseModule):
    def __init__(self,
                 in_channels,
                 num_classes,
                 encoder_in_channels=256, 
                 use_shared=False,
                 loss=None,
                 init_cfg=dict(type='Normal', layer='Linear', std=0.01),
                 **kwargs):
        super(SlvlClsHead, self).__init__(init_cfg)
        
        # 1. 强制维度统一 (解决 256 vs 768 报错)
        self.target_dim = 256 
        self.encoder_in_channels = 256
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.use_shared = use_shared
        self.criterion = nn.CrossEntropyLoss()

        if self.use_shared:
            self.fc = nn.Linear(self.target_dim, num_classes)
        else:
            self.fc = nn.Linear(self.target_dim, num_classes)
            
        # 2. 构建投影层
        self.input_proj = nn.ModuleList()
        channels_list = in_channels if isinstance(in_channels, (list, tuple)) else [in_channels]
        
        for ch in channels_list:
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv2d(ch, self.target_dim, kernel_size=1),
                    nn.GroupNorm(32, self.target_dim),
                )
            )

        self.positional_encoding = SinePositionalEncoding(
            num_feats=self.target_dim // 2, normalize=True)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def _prepare_encoder_inputs(self, neck_feature, img_metas):
        projected_feats = []
        for i, feat in enumerate(neck_feature):
            if i < len(self.input_proj):
                layer = self.input_proj[i]
            else:
                layer = self.input_proj[-1]
            projected_feats.append(layer(feat))

        if 'batch_input_shape' not in img_metas[0]:
             max_h = max([m['img_shape'][0] for m in img_metas])
             max_w = max([m['img_shape'][1] for m in img_metas])
             pad_h = int((max_h + 31) / 32) * 32
             pad_w = int((max_w + 31) / 32) * 32
             input_img_h, input_img_w = pad_h, pad_w
        else:
             input_img_h, input_img_w = img_metas[0]['batch_input_shape']

        batch_size = projected_feats[0].size(0)
        img_masks = projected_feats[0].new_ones((batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            img_h, img_w, _ = img_metas[img_id]['img_shape']
            img_masks[img_id, :img_h, :img_w] = 0
            
        mlvl_masks = []
        mlvl_pos_embeds = []
        for feat in projected_feats:
            mlvl_masks.append(
                F.interpolate(img_masks[None], size=feat.shape[-2:]).to(torch.bool).squeeze(0))
            mlvl_pos_embeds.append(self.positional_encoding(mlvl_masks[-1]))

        feat_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        
        for lvl, (feat, mask, pos_embed) in enumerate(zip(projected_feats, mlvl_masks, mlvl_pos_embeds)):
            bs, c, h, w = feat.shape
            spatial_shapes.append((h, w))
            
            # Flatten: (B, C, H, W) -> (B, L, C)
            feat = feat.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            
            lvl_pos_embed_flatten.append(pos_embed)
            feat_flatten.append(feat)
            mask_flatten.append(mask)
            
        feat_flatten = torch.cat(feat_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat_flatten.device)
        
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in mlvl_masks], 1)
        
        return feat_flatten, mask_flatten, lvl_pos_embed_flatten, spatial_shapes, level_start_index, valid_ratios

    def forward_train(self,
                      neck_feature,
                      backbone_feature,
                      gt_label,
                      shared_encoder,
                      img_metas, 
                      task=None,
                      **kwargs):
        
        query, key_padding_mask, query_pos, spatial_shapes, level_start_index, valid_ratios = \
            self._prepare_encoder_inputs(neck_feature, img_metas)

        reference_points = get_reference_points(spatial_shapes, valid_ratios, device=query.device)

        # --- [核心修复] 主动转置以适配 batch_first=False ---
        # query, query_pos: (B, L, C) -> (L, B, C)
        query = query.transpose(0, 1)
        query_pos = query_pos.transpose(0, 1)
        
        # 注意：key_padding_mask 和 reference_points 不需要转置！
        # MMCV 内部对它们的维度定义是固定的：mask 是 (B, L)，ref_points 是 (B, L, ...)
        # -----------------------------------------------

        memory = shared_encoder(
            query=query,
            key=None,
            value=None,
            query_pos=query_pos,
            query_key_padding_mask=key_padding_mask,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reference_points=reference_points,
            task=task 
        )
        
        # Encoder 输出也是 (L, B, C)，转回 (B, L, C)
        memory = memory.transpose(0, 1)
        
        # 后处理
        start_idx = level_start_index[-1]
        h, w = spatial_shapes[-1]
        start = start_idx.item()
        end = start + h * w
        
        last_lvl_feat = memory[:, start : end, :] 
        
        x = last_lvl_feat.mean(dim=1)
        cls_score = self.fc(x) 

        losses = dict()
        loss = self.criterion(cls_score, gt_label)
        losses['loss_cls'] = loss
        return losses

    def simple_test(self,
                    neck_feature,
                    backbone_feature,
                    shared_encoder,
                    img_metas=None,
                    task=None,
                    **kwargs):
        
        query, key_padding_mask, query_pos, spatial_shapes, level_start_index, valid_ratios = \
            self._prepare_encoder_inputs(neck_feature, img_metas)
            
        reference_points = get_reference_points(spatial_shapes, valid_ratios, device=query.device)

        # 转置
        query = query.transpose(0, 1)
        query_pos = query_pos.transpose(0, 1)

        memory = shared_encoder(
            query=query,
            key=None,
            value=None,
            query_pos=query_pos,
            query_key_padding_mask=key_padding_mask,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reference_points=reference_points,
            task=task 
        )
        
        # 转回
        memory = memory.transpose(0, 1)
        
        start_idx = level_start_index[-1]
        h, w = spatial_shapes[-1]
        start = start_idx.item()
        end = start + h * w
        last_lvl_feat = memory[:, start : end, :]
        
        x = last_lvl_feat.mean(dim=1)
        cls_score = self.fc(x)
        
        # 返回 List[np.array] 以兼容 MMCls 接口
        if isinstance(cls_score, torch.Tensor):
            cls_score = cls_score.detach().cpu().numpy()
        result = list(cls_score)
        
        return result

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from mmcv.runner import BaseModule
# from mmcv.utils import print_log
# from mmdet.models.builder import HEADS
# from mmdet.models.utils import SinePositionalEncoding

# def get_reference_points(spatial_shapes, valid_ratios, device):
#     """
#     手动生成 Reference Points (B, Total_L, n_levels, 2)
#     """
#     reference_points_list = []
#     for lvl, (H, W) in enumerate(spatial_shapes):
#         ref_y, ref_x = torch.meshgrid(
#             torch.linspace(0.5, H - 0.5, H, dtype=torch.float32, device=device),
#             torch.linspace(0.5, W - 0.5, W, dtype=torch.float32, device=device)
#         )
#         ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H)
#         ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W)
#         ref = torch.stack((ref_x, ref_y), -1)
#         reference_points_list.append(ref)
    
#     reference_points = torch.cat(reference_points_list, 1)
#     reference_points = reference_points[:, :, None] * valid_ratios[:, None]
#     return reference_points

# @HEADS.register_module()
# class SlvlClsHead(BaseModule):
#     def __init__(self,
#                  in_channels,
#                  num_classes,
#                  encoder_in_channels=256, 
#                  use_shared=False,
#                  loss=None,
#                  init_cfg=dict(type='Normal', layer='Linear', std=0.01),
#                  **kwargs):
#         super(SlvlClsHead, self).__init__(init_cfg)
        
#         self.target_dim = 256 
#         self.encoder_in_channels = 256
#         self.in_channels = in_channels
#         self.num_classes = num_classes
#         self.use_shared = use_shared
#         self.criterion = nn.CrossEntropyLoss()

#         if self.use_shared:
#             self.fc = nn.Linear(self.target_dim, num_classes)
#         else:
#             self.fc = nn.Linear(self.target_dim, num_classes)
            
#         self.input_proj = nn.ModuleList()
#         channels_list = in_channels if isinstance(in_channels, (list, tuple)) else [in_channels]
        
#         for ch in channels_list:
#             self.input_proj.append(
#                 nn.Sequential(
#                     nn.Conv2d(ch, self.target_dim, kernel_size=1),
#                     nn.GroupNorm(32, self.target_dim),
#                 )
#             )

#         self.positional_encoding = SinePositionalEncoding(
#             num_feats=self.target_dim // 2, normalize=True)

#     def get_valid_ratio(self, mask):
#         _, H, W = mask.shape
#         valid_H = torch.sum(~mask[:, :, 0], 1)
#         valid_W = torch.sum(~mask[:, 0, :], 1)
#         valid_ratio_h = valid_H.float() / H
#         valid_ratio_w = valid_W.float() / W
#         valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
#         return valid_ratio

#     def _prepare_encoder_inputs(self, neck_feature, img_metas):
#         projected_feats = []
#         for i, feat in enumerate(neck_feature):
#             if i < len(self.input_proj):
#                 layer = self.input_proj[i]
#             else:
#                 layer = self.input_proj[-1]
#             projected_feats.append(layer(feat))

#         if 'batch_input_shape' not in img_metas[0]:
#              max_h = max([m['img_shape'][0] for m in img_metas])
#              max_w = max([m['img_shape'][1] for m in img_metas])
#              pad_h = int((max_h + 31) / 32) * 32
#              pad_w = int((max_w + 31) / 32) * 32
#              input_img_h, input_img_w = pad_h, pad_w
#         else:
#              input_img_h, input_img_w = img_metas[0]['batch_input_shape']

#         batch_size = projected_feats[0].size(0)
#         img_masks = projected_feats[0].new_ones((batch_size, input_img_h, input_img_w))
#         for img_id in range(batch_size):
#             img_h, img_w, _ = img_metas[img_id]['img_shape']
#             img_masks[img_id, :img_h, :img_w] = 0
            
#         mlvl_masks = []
#         mlvl_pos_embeds = []
#         for feat in projected_feats:
#             mlvl_masks.append(
#                 F.interpolate(img_masks[None], size=feat.shape[-2:]).to(torch.bool).squeeze(0))
#             mlvl_pos_embeds.append(self.positional_encoding(mlvl_masks[-1]))

#         feat_flatten = []
#         mask_flatten = []
#         lvl_pos_embed_flatten = []
#         spatial_shapes = []
        
#         for lvl, (feat, mask, pos_embed) in enumerate(zip(projected_feats, mlvl_masks, mlvl_pos_embeds)):
#             bs, c, h, w = feat.shape
#             spatial_shapes.append((h, w))
            
#             feat = feat.flatten(2).transpose(1, 2) # (B, L, C)
#             mask = mask.flatten(1)
#             pos_embed = pos_embed.flatten(2).transpose(1, 2)
            
#             lvl_pos_embed = pos_embed 
                
#             lvl_pos_embed_flatten.append(lvl_pos_embed)
#             feat_flatten.append(feat)
#             mask_flatten.append(mask)
            
#         feat_flatten = torch.cat(feat_flatten, 1)
#         mask_flatten = torch.cat(mask_flatten, 1)
#         lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        
#         spatial_shapes = torch.as_tensor(
#             spatial_shapes, dtype=torch.long, device=feat_flatten.device)
        
#         level_start_index = torch.cat(
#             (spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
#         valid_ratios = torch.stack([self.get_valid_ratio(m) for m in mlvl_masks], 1)
        
#         return feat_flatten, mask_flatten, lvl_pos_embed_flatten, spatial_shapes, level_start_index, valid_ratios

#     def forward_train(self,
#                       neck_feature,
#                       backbone_feature,
#                       gt_label,
#                       shared_encoder,
#                       img_metas, 
#                       task=None,
#                       **kwargs):
        
#         query, key_padding_mask, query_pos, spatial_shapes, level_start_index, valid_ratios = \
#             self._prepare_encoder_inputs(neck_feature, img_metas)

#         reference_points = get_reference_points(spatial_shapes, valid_ratios, device=query.device)

#         # --- [核心修复] 手动转置为 (Length, Batch, Channel) ---
#         # 因为 MMCV Attention 默认 batch_first=False，它期待 (L, B, C)
#         query = query.transpose(0, 1)          
#         query_pos = query_pos.transpose(0, 1)
#         # 注意：reference_points 和 mask 不需要转置，MMCV 内部处理好了
#         # -------------------------------------------------

#         memory = shared_encoder(
#             query=query,
#             key=None,
#             value=None,
#             query_pos=query_pos,
#             query_key_padding_mask=key_padding_mask,
#             spatial_shapes=spatial_shapes,
#             level_start_index=level_start_index,
#             valid_ratios=valid_ratios,
#             reference_points=reference_points,
#             task=task 
#         )
        
#         # memory 出来是 (L, B, C) (如果输入是 L,B,C)
#         # 我们需要转回 (B, L, C) 以便后续处理
#         if memory.shape[0] != key_padding_mask.shape[0]: # 检查 dim 0 是否等于 Batch
#              memory = memory.transpose(0, 1)
        
#         start_idx = level_start_index[-1]
#         h, w = spatial_shapes[-1]
#         start = start_idx.item()
#         end = start + h * w
#         last_lvl_feat = memory[:, start : end, :] 
        
#         x = last_lvl_feat.mean(dim=1)
#         cls_score = self.fc(x) 

#         losses = dict()
#         loss = self.criterion(cls_score, gt_label)
#         losses['loss_cls'] = loss
#         return losses

#     def simple_test(self,
#                     neck_feature,
#                     backbone_feature,
#                     shared_encoder,
#                     img_metas=None,
#                     task=None,
#                     **kwargs):
        
#         query, key_padding_mask, query_pos, spatial_shapes, level_start_index, valid_ratios = \
#             self._prepare_encoder_inputs(neck_feature, img_metas)
            
#         reference_points = get_reference_points(spatial_shapes, valid_ratios, device=query.device)

#         # --- [核心修复] 转置 ---
#         query = query.transpose(0, 1)
#         query_pos = query_pos.transpose(0, 1)
#         # --------------------

#         memory = shared_encoder(
#             query=query,
#             key=None,
#             value=None,
#             query_pos=query_pos,
#             query_key_padding_mask=key_padding_mask,
#             spatial_shapes=spatial_shapes,
#             level_start_index=level_start_index,
#             valid_ratios=valid_ratios,
#             reference_points=reference_points,
#             task=task 
#         )
        
#         if memory.shape[0] != key_padding_mask.shape[0]:
#              memory = memory.transpose(0, 1)
        
#         start_idx = level_start_index[-1]
#         h, w = spatial_shapes[-1]
#         start = start_idx.item()
#         end = start + h * w
#         last_lvl_feat = memory[:, start : end, :]
        
#         x = last_lvl_feat.mean(dim=1)
#         cls_score = self.fc(x)
        
#         pred = cls_score.argmax(dim=1)
#         return {'pred_class': pred, 'cls_score': cls_score}


# from typing import List

# from torch import Tensor

# from mmcls.models import GlobalAveragePooling, LinearClsHead, HEADS


# @HEADS.register_module()
# class SlvlClsHead(LinearClsHead):
#     def __init__(self, *args, **kwargs):
#         super(SlvlClsHead, self).__init__(*args, **kwargs)
#         self.avg_pool = GlobalAveragePooling()

#     def pre_logits(self, x: List[Tensor]) -> Tensor:
#         cls_token = self.avg_pool(tuple(x))
#         if isinstance(cls_token, (tuple, list)):
#             cls_token = cls_token[-1]
#         return cls_token

#     def forward_train(self, neck_feature, backbone_feature, gt_label,
#                       shared_encoder, **kwargs):
#         return super(SlvlClsHead, self).forward_train(
#             backbone_feature, gt_label, **kwargs)

#     def simple_test(self, neck_feature, backbone_feature,
#                     shared_encoder, softmax=True, post_process=True):
#         return super(SlvlClsHead, self).simple_test(
#             backbone_feature, softmax, post_process)
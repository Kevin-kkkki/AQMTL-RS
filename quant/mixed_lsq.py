import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ._quan_base_plus import _Conv2dQ, _LinearQ, _ActQ, grad_scale, round_pass, Qmodes

class MixedConv2dLSQ(_Conv2dQ):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, 
                 num_tasks=3, nbits_high=8, nbits_low=4, nbits_a=4, **kwargs):
        super(MixedConv2dLSQ, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,
            nbits=nbits_high, mode=Qmodes.kernel_wise, **kwargs)
        
        self.nbits_high = nbits_high
        self.nbits_low = nbits_low
        
        # [Fix 1] 使用 torch.ones 初始化，防止 garbage value 导致 NaN
        if self.q_mode == Qmodes.kernel_wise:
            self.alpha_high = nn.Parameter(torch.ones(out_channels))
            self.alpha_low = nn.Parameter(torch.ones(out_channels))
        else:
            self.alpha_high = nn.Parameter(torch.ones(1))
            self.alpha_low = nn.Parameter(torch.ones(1))
            
        self.register_buffer('weight_mask', None) 
        self.register_buffer('init_state_mixed', torch.zeros(1))
        
        self.act = _ActQ(in_features=in_channels, num_tasks=num_tasks, nbits=nbits_a)

    def forward(self, x):
        if self.weight_mask is None:
            return F.conv2d(x, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)

        Qn_h, Qp_h = -2**(self.nbits_high-1), 2**(self.nbits_high-1)-1
        Qn_l, Qp_l = -2**(self.nbits_low-1), 2**(self.nbits_low-1)-1

        if self.training and self.init_state_mixed == 0:
            mask_bool = self.weight_mask.bool()
            w_h = self.weight[mask_bool]
            w_l = self.weight[~mask_bool]
            
            # [Fix 2] 即使部分为空，也用整体均值兜底初始化，防止 alpha 保持为 1.0 可能过大
            mean_abs = self.weight.abs().mean()
            if w_h.numel() > 0:
                self.alpha_high.data.copy_(2 * w_h.abs().mean() / math.sqrt(Qp_h))
            else:
                self.alpha_high.data.copy_(2 * mean_abs / math.sqrt(Qp_h)) # Fallback
                
            if w_l.numel() > 0:
                self.alpha_low.data.copy_(2 * w_l.abs().mean() / math.sqrt(Qp_l))
            else:
                self.alpha_low.data.copy_(2 * mean_abs / math.sqrt(Qp_l)) # Fallback

            self.init_state_mixed.fill_(1)

        g_h = 1.0 / math.sqrt(self.weight.numel() * Qp_h)
        g_l = 1.0 / math.sqrt(self.weight.numel() * Qp_l)
        
        # [Fix 3] 强力 Clamp，防止除以 0 或极小值
        alpha_h = grad_scale(self.alpha_high, g_h).view(-1, 1, 1, 1).abs().clamp(min=1e-5)
        alpha_l = grad_scale(self.alpha_low, g_l).view(-1, 1, 1, 1).abs().clamp(min=1e-5)

        w_q_h = round_pass((self.weight / alpha_h).clamp(Qn_h, Qp_h)) * alpha_h
        w_q_l = round_pass((self.weight / alpha_l).clamp(Qn_l, Qp_l)) * alpha_l
        
        w_mixed = w_q_h * self.weight_mask + w_q_l * (1 - self.weight_mask)

        x = F.conv2d(x, w_mixed, self.bias, self.stride, self.padding, self.dilation, self.groups)
        x = self.act(x)
        return x

class MixedLinearLSQ(_LinearQ):
    def __init__(self, in_features, out_features, bias=True, 
                 num_tasks=3, nbits_high=8, nbits_low=4, nbits_a=4, **kwargs):
        super(MixedLinearLSQ, self).__init__(
            in_features, out_features, bias, nbits=nbits_high, mode=Qmodes.kernel_wise, **kwargs)
        
        self.nbits_high = nbits_high
        self.nbits_low = nbits_low
        
        # [Fix 1] 使用 torch.ones 初始化
        if self.q_mode == Qmodes.kernel_wise:
            self.alpha_high = nn.Parameter(torch.ones(out_features))
            self.alpha_low = nn.Parameter(torch.ones(out_features))
        else:
            self.alpha_high = nn.Parameter(torch.ones(1))
            self.alpha_low = nn.Parameter(torch.ones(1))
            
        self.register_buffer('weight_mask', None)
        self.register_buffer('init_state_mixed', torch.zeros(1))
        self.act = _ActQ(in_features=out_features, num_tasks=num_tasks, nbits=nbits_a)

    def forward(self, x):
        if self.weight_mask is None:
            return F.linear(x, self.weight, self.bias)

        Qn_h, Qp_h = -2**(self.nbits_high-1), 2**(self.nbits_high-1)-1
        Qn_l, Qp_l = -2**(self.nbits_low-1), 2**(self.nbits_low-1)-1

        if self.training and self.init_state_mixed == 0:
            mask_bool = self.weight_mask.bool()
            w_h = self.weight[mask_bool]
            w_l = self.weight[~mask_bool]
            
            # [Fix 2] Fallback 初始化
            mean_abs = self.weight.abs().mean()
            if w_h.numel() > 0:
                self.alpha_high.data.copy_(2 * w_h.abs().mean() / math.sqrt(Qp_h))
            else:
                self.alpha_high.data.copy_(2 * mean_abs / math.sqrt(Qp_h))

            if w_l.numel() > 0:
                self.alpha_low.data.copy_(2 * w_l.abs().mean() / math.sqrt(Qp_l))
            else:
                self.alpha_low.data.copy_(2 * mean_abs / math.sqrt(Qp_l))
                
            self.init_state_mixed.fill_(1)

        g_h = 1.0 / math.sqrt(self.weight.numel() * Qp_h)
        g_l = 1.0 / math.sqrt(self.weight.numel() * Qp_l)
        
        # [Fix 3] 强力 Clamp
        alpha_h = grad_scale(self.alpha_high, g_h).view(-1, 1).abs().clamp(min=1e-5)
        alpha_l = grad_scale(self.alpha_low, g_l).view(-1, 1).abs().clamp(min=1e-5)
        
        w_q_h = round_pass((self.weight / alpha_h).clamp(Qn_h, Qp_h)) * alpha_h
        w_q_l = round_pass((self.weight / alpha_l).clamp(Qn_l, Qp_l)) * alpha_l
        
        w_mixed = w_q_h * self.weight_mask + w_q_l * (1 - self.weight_mask)
        
        x = F.linear(x, w_mixed, self.bias)
        x = self.act(x)
        return x
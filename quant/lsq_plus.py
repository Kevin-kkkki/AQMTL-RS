import torch
import torch.nn as F
import math
from ._quan_base_plus import _Conv2dQ, Qmodes, _LinearQ, _ActQ, _ActQ_conv, grad_scale, round_pass

__all__ = ['Conv2dLSQ', 'LinearLSQ', 'ActLSQ', 'QuantActLSQ']

class Conv2dLSQ(_Conv2dQ):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, 
                 num_tasks=3, nbits_w=4, nbits_a=4, **kwargs):
        super(Conv2dLSQ, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
            nbits=nbits_w, mode=Qmodes.kernel_wise)
        self.act = _ActQ_conv(in_features=out_channels, num_tasks=num_tasks, nbits=nbits_a)

    def forward(self, x):
        if self.alpha is None:
            return torch.nn.functional.conv2d(x, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        Qn = -2 ** (self.nbits - 1)
        Qp = 2 ** (self.nbits - 1) - 1
        if self.training and self.init_state == 0:
            self.alpha.data.copy_(2 * self.weight.abs().mean() / math.sqrt(Qp))
            self.init_state.fill_(1)
        g = 1.0 / math.sqrt(self.weight.numel() * Qp)
        alpha = grad_scale(self.alpha, g)
        alpha = alpha.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        w_q = round_pass((self.weight / alpha).clamp(Qn, Qp)) * alpha
        x = torch.nn.functional.conv2d(x, w_q, self.bias, self.stride, self.padding, self.dilation, self.groups)
        x = self.act(x)
        return x

class LinearLSQ(_LinearQ):
    def __init__(self, in_features, out_features, bias=True, 
                 num_tasks=3, nbits_w=4, nbits_a=4, **kwargs):
        super(LinearLSQ, self).__init__(in_features=in_features,
                                        out_features=out_features, bias=bias, nbits=nbits_w, mode=Qmodes.kernel_wise)
        self.act = _ActQ(in_features=out_features, num_tasks=num_tasks, nbits=nbits_a)

    def forward(self, x):
        if self.alpha is None: return torch.nn.functional.linear(x, self.weight, self.bias)
        Qn = -2 ** (self.nbits - 1)
        Qp = 2 ** (self.nbits - 1) - 1
        if self.training and self.init_state == 0:
            self.alpha.data.copy_(2 * self.weight.abs().mean() / math.sqrt(Qp))
            self.init_state.fill_(1)
        g = 1.0 / math.sqrt(self.weight.numel() * Qp)
        alpha = grad_scale(self.alpha, g)
        alpha = alpha.unsqueeze(1)
        w_q = round_pass((self.weight / alpha).clamp(Qn, Qp)) * alpha
        x = torch.nn.functional.linear(x, w_q, self.bias)
        x = self.act(x)
        return x

class ActLSQ(_ActQ):
    def __init__(self, in_features, num_tasks=3, nbits_a=4, mode=Qmodes.layer_wise, **kwargs):
        super(ActLSQ, self).__init__(in_features=in_features, num_tasks=num_tasks, nbits=nbits_a, mode=mode, **kwargs)

# --- [核心新增] 带激活函数的量化层 ---
class QuantActLSQ(ActLSQ):
    def __init__(self, activation_cls, in_features, num_tasks=3, nbits_a=4, mode=Qmodes.layer_wise, **kwargs):
        super().__init__(in_features, num_tasks, nbits_a, mode, **kwargs)
        self.activation = activation_cls()

    def forward(self, x):
        # 先做激活 (找回丢失的非线性能力！)
        x = self.activation(x)
        # 再做量化
        return super().forward(x)

# import torch
# import torch.nn.functional as F
# from torch.nn.modules.linear import Linear
# import math
# from torch.nn.parameter import Parameter
# from ._quan_base_plus import _Conv2dQ, Qmodes, _LinearQ, _ActQ, _LinearQ_v2, _ActQ_conv


# __all__ = ['Conv2dLSQ', 'LinearLSQ', 'ActLSQ']


# class FunLSQ(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, weight, alpha, g, Qn, Qp):
#         assert alpha > 0, 'alpha = {}'.format(alpha)
#         ctx.save_for_backward(weight, alpha)
#         ctx.other = g, Qn, Qp
#         q_w = (weight / alpha).round().clamp(Qn, Qp)
#         w_q = q_w * alpha
#         return w_q

#     @staticmethod
#     def backward(ctx, grad_weight):
#         weight, alpha = ctx.saved_tensors
#         g, Qn, Qp = ctx.other
#         q_w = weight / alpha
#         indicate_small = (q_w < Qn).float()
#         indicate_big = (q_w > Qp).float()

#         indicate_middle = 1.0 - indicate_small - indicate_big
#         grad_alpha = ((indicate_small * Qn + indicate_big * Qp + indicate_middle * (
#             -q_w + q_w.round())) * grad_weight * g).sum().unsqueeze(dim=0)
#         grad_weight = indicate_middle * grad_weight
#         return grad_weight, grad_alpha, None, None, None


# def grad_scale(x, scale):
#     y = x
#     y_grad = x * scale
#     return y.detach() - y_grad.detach() + y_grad


# def round_pass(x):
#     y = x.round()
#     y_grad = x
#     return y.detach() - y_grad.detach() + y_grad


# class Conv2dLSQ(_Conv2dQ):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1,
#                  padding=0, dilation=1, groups=1, bias=True, nbits_w=4, mode=Qmodes.kernel_wise, **kwargs):
#         super(Conv2dLSQ, self).__init__(
#             in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
#             stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
#             nbits=nbits_w, mode=mode)
#         self.act = ActLSQ_conv(in_features=in_channels, nbits_a=nbits_w)

#     def forward(self, x):
#         if self.alpha is None:
#             return F.conv2d(x, self.weight, self.bias, self.stride,
#                             self.padding, self.dilation, self.groups)
#         Qn = -2 ** (self.nbits - 1)
#         Qp = 2 ** (self.nbits - 1) - 1
#         if self.training and self.init_state == 0:
#             self.alpha.data.copy_(2 * self.weight.abs().mean() / math.sqrt(Qp))
#             self.init_state.fill_(1)

#         g = 1.0 / math.sqrt(self.weight.numel() * Qp)

#         alpha = grad_scale(self.alpha, g)
        
#         alpha = alpha.unsqueeze(1).unsqueeze(2).unsqueeze(3)
#         w_q = round_pass((self.weight / alpha).clamp(Qn, Qp)) * alpha

#         x = self.act(x)
        
#         return F.conv2d(x, w_q, self.bias, self.stride,
#                         self.padding, self.dilation, self.groups)
    
# class LinearLSQ(_LinearQ):
#     def __init__(self, in_features, out_features, bias=True, nbits_w=4, **kwargs):
#         super(LinearLSQ, self).__init__(in_features=in_features,
#                                         out_features=out_features, bias=bias, nbits=nbits_w, mode=Qmodes.kernel_wise)
#         self.act = ActLSQ(in_features=in_features, nbits_a=nbits_w)


#     def qw(self, weight):
#         Qn = -2 ** (self.nbits - 1)
#         Qp = 2 ** (self.nbits - 1) - 1
#         if self.training and self.init_state == 0:
#             self.alpha.data.copy_(2 * self.weight.abs().mean() / math.sqrt(Qp))
#             self.init_state.fill_(1)
#         g = 1.0 / math.sqrt(self.weight.numel() * Qp)

#         # Method1:
#         alpha = grad_scale(self.alpha, g)
        
#         alpha = alpha.unsqueeze(1)
#         w_q = round_pass((self.weight / alpha).clamp(Qn, Qp)) * alpha
#         return w_q

#     def forward(self, x, task):
#         if self.alpha is None:
#             return F.linear(x, self.weight, self.bias)

#         w_q = self.qw(self.weight)

#         x = self.act(x, task)
        
#         return F.linear(x, w_q, self.bias)

# class ActLSQ_conv(_ActQ_conv):
#     def __init__(self, in_features, nbits_a=4, mode=Qmodes.layer_wise, **kwargs):
#         super(ActLSQ_conv, self).__init__(in_features=in_features, nbits=nbits_a, mode=mode)
#         # print(self.alpha.shape, self.zero_point.shape)
#     def forward(self, x):
#         if self.alpha is None:
#             return x

#         if self.training and self.init_state == 0:
#             if x.min() < -1e-5:
#                 self.signed.data.fill_(1)
#             # print(self.signed)
#             if self.signed == 1:
#                 Qn = -2 ** (self.nbits - 1)
#                 Qp = 2 ** (self.nbits - 1) - 1
#             else:
#                 Qn = 0
#                 Qp = 2 ** self.nbits - 1
            
#             self.alpha.data.copy_((x.max() - x.min()) / (Qp - Qn))
            
#             self.zero_point.data.copy_(self.zero_point.data * 0.9 + 0.1 * (torch.min(x.detach()) - self.alpha.data * Qn))
#             self.init_state.fill_(1)

#         if self.signed == 1:
#             Qn = -2 ** (self.nbits - 1)
#             Qp = 2 ** (self.nbits - 1) - 1
#         else:
#             Qn = 0
#             Qp = 2 ** self.nbits - 1

#         g = 1.0 / math.sqrt(x.numel() * Qp)

#         # Method1:
#         zero_point = (self.zero_point.round() - self.zero_point).detach() + self.zero_point
#         alpha = grad_scale(self.alpha, g)
#         zero_point = grad_scale(zero_point, g)
#         if len(x.shape)==2:
#             alpha = alpha.unsqueeze(0)
#             zero_point = zero_point.unsqueeze(0)
#         elif len(x.shape)==3:
#             if x.shape[0] == alpha.shape[0]:
#                 alpha = alpha.unsqueeze(1).unsqueeze(2)
#                 zero_point = zero_point.unsqueeze(1).unsqueeze(2)
#             elif x.shape[1] == alpha.shape[0]:
#                 alpha = alpha.unsqueeze(0).unsqueeze(2)
#                 zero_point = zero_point.unsqueeze(0).unsqueeze(2)
#             elif x.shape[2] == alpha.shape[0]:
#                 alpha = alpha.unsqueeze(0).unsqueeze(0)
#                 zero_point = zero_point.unsqueeze(0).unsqueeze(0)
#         elif len(x.shape)==4:
#             # A, B, C, D = x.shape
#             if x.shape[0] == alpha.shape[0]:
#                 alpha = alpha.unsqueeze(1).unsqueeze(2).unsqueeze(3)
#                 zero_point = zero_point.unsqueeze(1).unsqueeze(2).unsqueeze(3)
#             elif x.shape[1] == alpha.shape[0]:
#                 alpha = alpha.unsqueeze(0).unsqueeze(2).unsqueeze(3)
#                 zero_point = zero_point.unsqueeze(0).unsqueeze(2).unsqueeze(3)
#             elif x.shape[2] == alpha.shape[0]:
#                 alpha = alpha.unsqueeze(0).unsqueeze(0).unsqueeze(3)
#                 zero_point = zero_point.unsqueeze(0).unsqueeze(0).unsqueeze(3)
#             elif x.shape[3] == alpha.shape[0]:
#                 alpha = alpha.unsqueeze(0).unsqueeze(0).unsqueeze(0)
#                 zero_point = zero_point.unsqueeze(0).unsqueeze(0).unsqueeze(0)

#         x = round_pass((x / alpha + zero_point).clamp(Qn, Qp))
#         x = (x - zero_point) * alpha
        
#         return x

# # for qipt
# class ActLSQ(_ActQ):
#     def __init__(self, in_features, nbits_a=4, mode=Qmodes.layer_wise, **kwargs):
#         super(ActLSQ, self).__init__(in_features=in_features, nbits=nbits_a, mode=mode)
#     def forward(self, x, task):
#         if self.alpha is None:
#             return x

#         if self.training and self.init_state[task] == 0:
#             if x.min() < -1e-5:
#                 self.signed.data[task] = 1
            
#             self.init_state[task] = 1
        
#         alpha = self.alpha[task]
#         zero_point = self.zero_point[task]
        
#         signed = self.signed[task]

#         if signed == 1:
#             Qn = -2 ** (self.nbits - 1)
#             Qp = 2 ** (self.nbits - 1) - 1
#         else:
#             Qn = 0
#             Qp = 2 ** self.nbits - 1

#         g = 1.0 / math.sqrt(x.numel() * Qp)

#         # Method1:
#         zero_point = (zero_point.round() - zero_point).detach() + zero_point
#         alpha = grad_scale(alpha, g)
#         zero_point = grad_scale(zero_point, g)
#         # x = round_pass((x / alpha).clamp(Qn, Qp)) * alpha
#         if len(x.shape)==2:
#             alpha = alpha.unsqueeze(0)
#             zero_point = zero_point.unsqueeze(0)
#         elif len(x.shape)==3:
#             if x.shape[0] == alpha.shape[0]:
#                 alpha = alpha.unsqueeze(1).unsqueeze(2)
#                 zero_point = zero_point.unsqueeze(1).unsqueeze(2)
#             elif x.shape[1] == alpha.shape[0]:
#                 alpha = alpha.unsqueeze(0).unsqueeze(2)
#                 zero_point = zero_point.unsqueeze(0).unsqueeze(2)
#             elif x.shape[2] == alpha.shape[0]:
#                 alpha = alpha.unsqueeze(0).unsqueeze(0)
#                 zero_point = zero_point.unsqueeze(0).unsqueeze(0)
#         elif len(x.shape)==4:
#             # A, B, C, D = x.shape
#             if x.shape[0] == alpha.shape[0]:
#                 alpha = alpha.unsqueeze(1).unsqueeze(2).unsqueeze(3)
#                 zero_point = zero_point.unsqueeze(1).unsqueeze(2).unsqueeze(3)
#             elif x.shape[1] == alpha.shape[0]:
#                 alpha = alpha.unsqueeze(0).unsqueeze(2).unsqueeze(3)
#                 zero_point = zero_point.unsqueeze(0).unsqueeze(2).unsqueeze(3)
#             elif x.shape[2] == alpha.shape[0]:
#                 alpha = alpha.unsqueeze(0).unsqueeze(0).unsqueeze(3)
#                 zero_point = zero_point.unsqueeze(0).unsqueeze(0).unsqueeze(3)
#             elif x.shape[3] == alpha.shape[0]:
#                 alpha = alpha.unsqueeze(0).unsqueeze(0).unsqueeze(0)
#                 zero_point = zero_point.unsqueeze(0).unsqueeze(0).unsqueeze(0)

#         x = round_pass((x / alpha + zero_point).clamp(Qn, Qp))
#         x = (x - zero_point) * alpha

#         return x

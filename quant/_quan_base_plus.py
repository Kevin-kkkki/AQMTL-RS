"""
    Quantized modules: the base class
"""
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import math
from enum import Enum

__all__ = ['Qmodes', '_Conv2dQ', '_LinearQ', '_ActQ', '_ActQ_conv',
           'truncation', 'get_sparsity_mask', 'FunStopGradient', 'round_pass', 'grad_scale']

class Qmodes(Enum):
    layer_wise = 1
    kernel_wise = 2

def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return y.detach() - y_grad.detach() + y_grad

def round_pass(x):
    y = x.round()
    y_grad = x
    return y.detach() - y_grad.detach() + y_grad

def get_default_kwargs_q(kwargs_q, layer_type):
    default = {'nbits': 4}
    if 'mode' not in kwargs_q:
        if isinstance(layer_type, (_Conv2dQ, _LinearQ)):
            default['mode'] = Qmodes.layer_wise
        else:
            default['mode'] = Qmodes.layer_wise
    for k, v in default.items():
        if k not in kwargs_q:
            kwargs_q[k] = v
    return kwargs_q

class _Conv2dQ(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, **kwargs_q):
        super(_Conv2dQ, self).__init__(in_channels, out_channels, kernel_size, stride=stride,
                                       padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.kwargs_q = get_default_kwargs_q(kwargs_q, layer_type=self)
        self.nbits = kwargs_q['nbits']
        if self.nbits < 0:
            self.register_parameter('alpha', None)
            return
        self.q_mode = kwargs_q['mode']
        if self.q_mode == Qmodes.kernel_wise:
            self.alpha = Parameter(torch.Tensor(out_channels))
        else:
            self.alpha = Parameter(torch.Tensor(1))
        self.register_buffer('init_state', torch.zeros(1))

    def extra_repr(self):
        s_prefix = super(_Conv2dQ, self).extra_repr()
        if self.alpha is None:
            return '{}, fake'.format(s_prefix)
        return '{}, {}'.format(s_prefix, self.kwargs_q)

class _LinearQ(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, **kwargs_q):
        super(_LinearQ, self).__init__(in_features=in_features, out_features=out_features, bias=bias)
        self.kwargs_q = get_default_kwargs_q(kwargs_q, layer_type=self)
        self.nbits = kwargs_q['nbits']
        if self.nbits < 0:
            self.register_parameter('alpha', None)
            return
        self.q_mode = kwargs_q['mode']
        self.alpha = Parameter(torch.Tensor(1))
        if self.q_mode == Qmodes.kernel_wise:
            self.alpha = Parameter(torch.Tensor(out_features))
        self.register_buffer('init_state', torch.zeros(1))

    def extra_repr(self):
        s_prefix = super(_LinearQ, self).extra_repr()
        if self.alpha is None:
            return '{}, fake'.format(s_prefix)
        return '{}, {}'.format(s_prefix, self.kwargs_q)

class _ActQ(nn.Module):
    def __init__(self, in_features, num_tasks=3, **kwargs_q):
        super(_ActQ, self).__init__()
        self.kwargs_q = get_default_kwargs_q(kwargs_q, layer_type=self)
        self.nbits = kwargs_q['nbits']
        if self.nbits < 0:
            self.register_parameter('alpha', None)
            self.register_parameter('zero_point', None)
            return
        self.q_mode = kwargs_q['mode']
        self.num_tasks = num_tasks
        self.alpha = Parameter(torch.ones(num_tasks, 1))
        self.zero_point = Parameter(torch.zeros(num_tasks, 1))
        if self.q_mode == Qmodes.kernel_wise:
            self.alpha = Parameter(torch.ones(num_tasks, in_features))
            self.zero_point = Parameter(torch.zeros(num_tasks, in_features))
        self.register_buffer('init_state', torch.zeros(num_tasks, 1))
        self.register_buffer('signed', torch.zeros(num_tasks, 1))
        self.register_buffer('task_buffer', torch.tensor(0, dtype=torch.int))

    def change_buffer(self, task_id):
        self.task_buffer.fill_(task_id)

    def forward(self, x):
        if self.alpha is None: return x
        task = self.task_buffer.item()
        if task >= self.num_tasks: task = 0

        if self.training and self.init_state[task] == 0:
            min_val, max_val = x.min(), x.max()
            val_range = max_val - min_val
            if val_range == 0: val_range = 1e-5 # 防除零
            self.alpha.data[task] = val_range / (2 ** self.nbits - 1)
            self.zero_point.data[task] = min_val - self.alpha.data[task] * (-2 ** (self.nbits - 1))
            self.init_state[task] = 1

        alpha = self.alpha[task]
        zero_point = self.zero_point[task]
        Qn = -2 ** (self.nbits - 1)
        Qp = 2 ** (self.nbits - 1) - 1
        g = 1.0 / math.sqrt(x.numel() * Qp)
        zero_point = (zero_point.round() - zero_point).detach() + zero_point
        alpha = grad_scale(alpha, g)
        zero_point = grad_scale(zero_point, g)
        if x.dim() == 4 and alpha.ndim == 1:
             alpha = alpha.view(1, -1, 1, 1)
             zero_point = zero_point.view(1, -1, 1, 1)
        x = round_pass((x / alpha + zero_point).clamp(Qn, Qp))
        x = (x - zero_point) * alpha
        return x

class _ActQ_conv(_ActQ):
    """卷积专用ActQ，逻辑一致，复用代码"""
    pass

# import torch
# import torch.nn as nn
# from torch.nn.parameter import Parameter

# import math
# from enum import Enum

# __all__ = ['Qmodes',  '_Conv2dQ', '_LinearQ', '_LinearQ_v2', '_ActQ',
#            'truncation', 'get_sparsity_mask', 'FunStopGradient', 'round_pass', 'grad_scale']


# class Qmodes(Enum):
#     layer_wise = 1
#     kernel_wise = 2


# def grad_scale(x, scale):
#     y = x
#     y_grad = x * scale
#     return y.detach() - y_grad.detach() + y_grad


# def get_sparsity_mask(param, sparsity):
#     bottomk, _ = torch.topk(param.abs().view(-1), int(sparsity * param.numel()), largest=False, sorted=True)
#     threshold = bottomk.data[-1]
#     return torch.gt(torch.abs(param), threshold).type(param.type())


# def round_pass(x):
#     y = x.round()
#     y_grad = x
#     return y.detach() - y_grad.detach() + y_grad


# class FunStopGradient(torch.autograd.Function):

#     @staticmethod
#     def forward(ctx, weight, stopGradientMask):
#         ctx.save_for_backward(stopGradientMask)
#         return weight

#     @staticmethod
#     def backward(ctx, grad_outputs):
#         stopGradientMask, = ctx.saved_tensors
#         grad_inputs = grad_outputs * stopGradientMask
#         return grad_inputs, None


# def log_shift(value_fp):
#     value_shift = 2 ** (torch.log2(value_fp).ceil())
#     return value_shift


# def clamp(input, min, max, inplace=False):
#     if inplace:
#         input.clamp_(min, max)
#         return input
#     return torch.clamp(input, min, max)


# def get_quantized_range(num_bits, signed=True):
#     if signed:
#         n = 2 ** (num_bits - 1)
#         return -n, n - 1
#     return 0, 2 ** num_bits - 1


# def linear_quantize(input, scale_factor, inplace=False):
#     if inplace:
#         input.mul_(scale_factor).round_()
#         return input
#     return torch.round(scale_factor * input)


# def linear_quantize_clamp(input, scale_factor, clamp_min, clamp_max, inplace=False):
#     output = linear_quantize(input, scale_factor, inplace)
#     return clamp(output, clamp_min, clamp_max, inplace)


# def linear_dequantize(input, scale_factor, inplace=False):
#     if inplace:
#         input.div_(scale_factor)
#         return input
#     return input / scale_factor


# def truncation(fp_data, nbits=8):
#     il = torch.log2(torch.max(fp_data.max(), fp_data.min().abs())) + 1
#     il = math.ceil(il - 1e-5)
#     qcode = nbits - il
#     scale_factor = 2 ** qcode
#     clamp_min, clamp_max = get_quantized_range(nbits, signed=True)
#     q_data = linear_quantize_clamp(fp_data, scale_factor, clamp_min, clamp_max)
#     q_data = linear_dequantize(q_data, scale_factor)
#     return q_data, qcode


# def get_default_kwargs_q(kwargs_q, layer_type):
#     default = {
#         'nbits': 4
#     }
#     if isinstance(layer_type, _Conv2dQ):
#         default.update({
#             'mode': Qmodes.layer_wise})
#     elif isinstance(layer_type, _LinearQ):
#         pass
#     elif isinstance(layer_type, _ActQ):
#         pass
#         # default.update({
#         #     'signed': 'Auto'})
#     else:
#         assert NotImplementedError
#         return
#     for k, v in default.items():
#         if k not in kwargs_q:
#             kwargs_q[k] = v
#     return kwargs_q


# class _Conv2dQ(nn.Conv2d):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1,
#                  padding=0, dilation=1, groups=1, bias=True, **kwargs_q):
#         super(_Conv2dQ, self).__init__(in_channels, out_channels, kernel_size, stride=stride,
#                                        padding=padding, dilation=dilation, groups=groups, bias=bias)
#         self.kwargs_q = get_default_kwargs_q(kwargs_q, layer_type=self)
#         self.nbits = kwargs_q['nbits']
#         if self.nbits < 0:
#             self.register_parameter('alpha', None)
#             return
#         self.q_mode = kwargs_q['mode']
#         if self.q_mode == Qmodes.kernel_wise:
#             self.alpha = Parameter(torch.Tensor(out_channels))
#         else:  # layer-wise quantization
#             self.alpha = Parameter(torch.Tensor(1))
#         self.register_buffer('init_state', torch.zeros(1))

#     def add_param(self, param_k, param_v):
#         self.kwargs_q[param_k] = param_v

#     def set_bit(self, nbits):
#         self.kwargs_q['nbits'] = nbits

#     def extra_repr(self):
#         s_prefix = super(_Conv2dQ, self).extra_repr()
#         if self.alpha is None:
#             return '{}, fake'.format(s_prefix)
#         return '{}, {}'.format(s_prefix, self.kwargs_q)


# class _LinearQ(nn.Linear):
#     def __init__(self, in_features, out_features, bias=True, **kwargs_q):
#         super(_LinearQ, self).__init__(in_features=in_features, out_features=out_features, bias=bias)
#         self.kwargs_q = get_default_kwargs_q(kwargs_q, layer_type=self)
#         self.nbits = kwargs_q['nbits']
#         if self.nbits < 0:
#             self.register_parameter('alpha', None)
#             return
#         self.q_mode = kwargs_q['mode']
#         self.alpha = Parameter(torch.Tensor(1))
#         if self.q_mode == Qmodes.kernel_wise:
#             self.alpha = Parameter(torch.Tensor(out_features))
#         self.register_buffer('init_state', torch.zeros(1))

#     def add_param(self, param_k, param_v):
#         self.kwargs_q[param_k] = param_v

#     def extra_repr(self):
#         s_prefix = super(_LinearQ, self).extra_repr()
#         if self.alpha is None:
#             return '{}, fake'.format(s_prefix)
#         return '{}, {}'.format(s_prefix, self.kwargs_q)

# class _LinearQ_v2(nn.Linear):
#     def __init__(self, in_features, out_features, bias=True, **kwargs_q):
#         super(_LinearQ_v2, self).__init__(in_features=in_features, out_features=out_features, bias=bias)
#         self.kwargs_q = get_default_kwargs_q(kwargs_q, layer_type=self)
#         self.nbits = kwargs_q['nbits']
#         if self.nbits < 0:
#             self.register_parameter('alpha', None)
#             return
#         self.q_mode = kwargs_q['mode']
#         self.alpha = Parameter(torch.Tensor(1))
#         self.beta = Parameter(torch.Tensor(1))
#         if self.q_mode == Qmodes.kernel_wise:
#             self.alpha = Parameter(torch.Tensor(out_features))
#             self.beta = Parameter(torch.Tensor(in_features))
#         self.register_buffer('init_state', torch.zeros(1))

#     def add_param(self, param_k, param_v):
#         self.kwargs_q[param_k] = param_v

#     def extra_repr(self):
#         s_prefix = super(_LinearQ_v2, self).extra_repr()
#         if self.alpha is None:
#             return '{}, fake'.format(s_prefix)
#         return '{}, {}'.format(s_prefix, self.kwargs_q)


# class _ActQ_conv(nn.Module):
#     def __init__(self, in_features, num_tasks=3, **kwargs_q):
#         super(_ActQ_conv, self).__init__()
#         self.kwargs_q = get_default_kwargs_q(kwargs_q, layer_type=self)
#         self.nbits = kwargs_q['nbits']
#         if self.nbits < 0:
#             self.register_parameter('alpha', None)
#             self.register_parameter('zero_point', None)
#             return
#         self.q_mode = kwargs_q['mode']
#         self.num_tasks = num_tasks

#         # self.alpha = Parameter(torch.Tensor(1))
#         # self.zero_point = Parameter(torch.Tensor([0]))
#         self.alpha = Parameter(torch.ones(num_tasks, 1))
#         self.zero_point = Parameter(torch.zeros(num_tasks, 1))
#         # if self.q_mode == Qmodes.kernel_wise:
#         #     self.alpha = Parameter(torch.Tensor(in_features))
#         #     self.zero_point = Parameter(torch.Tensor(in_features))
#         #     torch.nn.init.zeros_(self.zero_point)
#         if self.q_mode == Qmodes.kernel_wise:
#             # 如果是通道级量化，形状变为 [num_tasks, in_features]
#             self.alpha = Parameter(torch.ones(num_tasks, in_features))
#             self.zero_point = Parameter(torch.zeros(num_tasks, in_features))
#         # self.register_buffer('init_state', torch.zeros(1))
#         # self.register_buffer('signed', torch.zeros(1))
#         self.register_buffer('init_state', torch.zeros(num_tasks, 1))
#         self.register_buffer('signed', torch.zeros(num_tasks, 1))
#         self.register_buffer('task_buffer', torch.tensor(0, dtype=torch.int))

#     def change_buffer(self, task_id):
#         self.task_buffer.fill_(task_id)

#     def add_param(self, param_k, param_v):
#         self.kwargs_q[param_k] = param_v

#     def set_bit(self, nbits):
#         self.kwargs_q['nbits'] = nbits

#     def extra_repr(self):
#         # s_prefix = super(_ActQ, self).extra_repr()
#         if self.alpha is None:
#             return 'fake'
#         return '{}'.format(self.kwargs_q)


# class _ActQ(nn.Module):
#     def __init__(self, in_features, num_tasks=3, **kwargs_q):
#         super(_ActQ, self).__init__()
#         self.kwargs_q = get_default_kwargs_q(kwargs_q, layer_type=self)
#         self.nbits = kwargs_q['nbits']
#         if self.nbits < 0:
#             self.register_parameter('alpha', None)
#             self.register_parameter('zero_point', None)
#             return
#         self.q_mode = kwargs_q['mode']
#         self.num_tasks = num_tasks
        
#         # self.alpha = Parameter(torch.ones(6,1))
#         # self.zero_point = Parameter(torch.zeros(6,1))
#         # self.alpha1 = Parameter(torch.ones(1))
#         # self.alpha2 = Parameter(torch.ones(1))
#         # self.alpha3 = Parameter(torch.ones(1))
#         # self.alpha4 = Parameter(torch.ones(1))
#         # self.alpha5 = Parameter(torch.ones(1))
#         # self.alpha6 = Parameter(torch.ones(1))
#         # self.alpha = [self.alpha1,self.alpha2,self.alpha3,self.alpha4,self.alpha5,self.alpha6]
#         # self.zero_point1 = Parameter(torch.zeros(1))
#         # self.zero_point2 = Parameter(torch.zeros(1))
#         # self.zero_point3 = Parameter(torch.zeros(1))
#         # self.zero_point4 = Parameter(torch.zeros(1))
#         # self.zero_point5 = Parameter(torch.zeros(1))
#         # self.zero_point6 = Parameter(torch.zeros(1))
#         # self.zero_point = [self.zero_point1,self.zero_point1,self.zero_point1,self.zero_point1,self.zero_point1,self.zero_point1]
#         # if self.q_mode == Qmodes.kernel_wise:
#         #     self.alpha = Parameter(torch.Tensor(in_features))
#         #     self.zero_point = Parameter(torch.Tensor(in_features))
#         #     torch.nn.init.zeros_(self.zero_point)
#         # self.register_buffer('init_state', torch.zeros(6, 1))
#         # self.register_buffer('signed', torch.zeros(6, 1))
#         self.alpha = Parameter(torch.ones(num_tasks, 1)) 
#         self.zero_point = Parameter(torch.zeros(num_tasks, 1))
#         if self.q_mode == Qmodes.kernel_wise:
#             self.alpha = Parameter(torch.ones(num_tasks, in_features))
#             self.zero_point = Parameter(torch.zeros(num_tasks, in_features))
#         self.register_buffer('init_state', torch.zeros(num_tasks, 1))
#         self.register_buffer('signed', torch.zeros(num_tasks, 1))
#         self.register_buffer('task_buffer', torch.tensor(0, dtype=torch.int))

#     def change_buffer(self, task_id):
#         """被 set_task 调用，用于更新当前模块的任务ID"""
#         self.task_buffer.fill_(task_id)

#     def add_param(self, param_k, param_v):
#         self.kwargs_q[param_k] = param_v

#     def set_bit(self, nbits):
#         self.kwargs_q['nbits'] = nbits

#     def extra_repr(self):
#         if self.alpha is None:
#             return 'fake'
#         return '{}'.format(self.kwargs_q)

#     def forward(self, x):
#         # 1. 获取当前任务 ID
#         task = self.task_buffer.item()
        
#         # 2. 确保 ID 不越界 (防御性编程)
#         if task >= self.num_tasks:
#             # 如果出现这种情况，说明 set_task 没调用，或者 task_map 有误
#             # 默认回退到任务 0，或者报错
#             task = 0 
            
#         # ... (量化初始化逻辑: init_state[task]) ...
        
#         # 3. 获取对应任务的参数
#         # alpha 的形状是 [num_tasks, 1]，alpha[task] 得到 [1]
#         # LSQ 计算时会自动广播
#         alpha = self.alpha[task]
#         zero_point = self.zero_point[task]
        
#         # ... (执行 LSQ 量化计算: (x / alpha + zp).round()...) ...
        
#         return x_q
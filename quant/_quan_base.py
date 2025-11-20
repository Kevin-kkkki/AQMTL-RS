# """
#     Quantized modules: the base class
# """
# import torch
# import torch.nn as nn
# from torch.nn.parameter import Parameter

# import math
# from enum import Enum

# __all__ = ['Qmodes',  '_Conv2dQ', '_LinearQ', '_ActQ',
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
#     threshold = bottomk.data[-1]  # This is the largest element from the group of elements that we prune away
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


# class _ActQ(nn.Module):
#     def __init__(self, in_features, **kwargs_q):
#         super(_ActQ, self).__init__()
#         self.kwargs_q = get_default_kwargs_q(kwargs_q, layer_type=self)
#         self.nbits = kwargs_q['nbits']
#         if self.nbits < 0:
#             self.register_parameter('alpha', None)
#             self.register_parameter('zero_point', None)
#             return
#         # self.signed = kwargs_q['signed']
#         self.q_mode = kwargs_q['mode']
#         self.alpha = Parameter(torch.Tensor(1))
#         self.zero_point = Parameter(torch.Tensor([0]))
#         if self.q_mode == Qmodes.kernel_wise:
#             self.alpha = Parameter(torch.Tensor(in_features))
#             self.zero_point = Parameter(torch.Tensor(in_features))
#             torch.nn.init.zeros_(self.zero_point)
#         # self.zero_point = Parameter(torch.Tensor([0]))
#         self.register_buffer('init_state', torch.zeros(1))
#         self.register_buffer('signed', torch.zeros(1))

#     def add_param(self, param_k, param_v):
#         self.kwargs_q[param_k] = param_v

#     def set_bit(self, nbits):
#         self.kwargs_q['nbits'] = nbits

#     def extra_repr(self):
#         # s_prefix = super(_ActQ, self).extra_repr()
#         if self.alpha is None:
#             return 'fake'
#         return '{}'.format(self.kwargs_q)





"""
    量化模块：基础类
"""
# 导入PyTorch相关模块
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

import math
from enum import Enum  # 用于创建枚举类

# 定义模块公开接口，指定可被外部导入的类和函数
__all__ = ['Qmodes',  '_Conv2dQ', '_LinearQ', '_ActQ',
           'truncation', 'get_sparsity_mask', 'FunStopGradient', 'round_pass', 'grad_scale']


# 定义量化模式枚举类，用于指定量化粒度
class Qmodes(Enum):
    layer_wise = 1  # 层级量化：整个层共享一个量化参数
    kernel_wise = 2  # 核级量化：每个卷积核/输出通道有独立的量化参数


# 梯度缩放函数：调整量化参数的梯度大小
def grad_scale(x, scale):
    y = x  # 前向传播返回原始值
    y_grad = x * scale  # 反向传播使用缩放后的梯度
    # 通过detach分离计算图，实现前向和反向路径的分离
    return y.detach() - y_grad.detach() + y_grad


# 获取稀疏化掩码：根据指定的稀疏度筛选出需要保留的参数
def get_sparsity_mask(param, sparsity):
    '''
    结合模型剪枝和量化，生成稀疏化掩码筛选出需要保留的参数。
    比如：计算出参数中绝对值最小的sparsity 比例的元素（例如 sparsity=0.5 表示筛选出 50% 最小的参数）。
    以这些最小元素中的最大值作为阈值，生成掩码：大于阈值的参数标记为 1（保留），小于等于阈值的标记为 0（剪枝）。
    '''
    # 找到绝对值最小的sparsity比例的参数
    bottomk, _ = torch.topk(param.abs().view(-1), int(sparsity * param.numel()), largest=False, sorted=True)
    # 确定阈值：被剪枝参数中最大的那个值
    threshold = bottomk.data[-1]
    # 返回掩码：大于阈值的参数保留（1），否则剪枝（0）
    return torch.gt(torch.abs(param), threshold).type(param.type())


# 带直通估计器的取整函数：解决量化中取整操作不可导问题
def round_pass(x):
    y = x.round()  # 前向传播执行取整
    y_grad = x  # 反向传播直接使用输入的梯度
    # 分离计算图，实现前向取整、反向直通
    return y.detach() - y_grad.detach() + y_grad


# 自定义停止梯度函数：根据掩码控制哪些参数不更新梯度
'''
用于控制参数梯度更新范围的工具
实现 “前向保留全部参数、反向仅更新指定参数” 的功能
'''
class FunStopGradient(torch.autograd.Function):

    @staticmethod
    def forward(ctx, weight, stopGradientMask):
        # 保存掩码用于反向传播
        ctx.save_for_backward(stopGradientMask)
        return weight  # 前向传播返回原始权重

    @staticmethod
    def backward(ctx, grad_outputs):
        # 恢复掩码
        stopGradientMask, = ctx.saved_tensors
        # 只保留需要更新的参数的梯度
        grad_inputs = grad_outputs * stopGradientMask
        return grad_inputs, None


# 对数移位函数：将缩放系数调整为最接近的2的幂次方
def log_shift(value_fp):
    value_shift = 2 **(torch.log2(value_fp).ceil())
    return value_shift


# 封装clamp函数：限制量化后整数的取值范围，确保量化结果符合指定位数的数值边界
'''
限制量化后整数的取值范围，确保量化结果符合指定位数的数值边界
'''
def clamp(input, min, max, inplace=False):
    if inplace:
        input.clamp_(min, max)  # 原地操作
        return input
    return torch.clamp(input, min, max)  # 返回新张量


# 获取量化范围：根据位数和是否有符号计算量化的最小值和最大值
'''
根据量化位数和符号类型，确定量化过程中整数取值的范围。
'''
def get_quantized_range(num_bits, signed=True):
    if signed:  # 有符号量化（对称量化）
        n = 2** (num_bits - 1)
        return -n, n - 1
    # 无符号量化
    return 0, 2 ** num_bits - 1


# 线性量化：将浮点值缩放到整数范围并取整
'''
将浮点值映射到整数的过程
f_quant = round(scale_factor * f)
'''
def linear_quantize(input, scale_factor, inplace=False):
    if inplace:
        input.mul_(scale_factor).round_()  # 原地缩放并取整
        return input
    return torch.round(scale_factor * input)  # 返回缩放取整后的新张量


# 带截断的线性量化：量化后截断到指定范围
def linear_quantize_clamp(input, scale_factor, clamp_min, clamp_max, inplace=False):
    output = linear_quantize(input, scale_factor, inplace)  # 先量化
    return clamp(output, clamp_min, clamp_max, inplace)  # 再截断


# 线性反量化：将量化后的整数转换回浮点数
def linear_dequantize(input, scale_factor, inplace=False):
    if inplace:
        input.div_(scale_factor)  # 原地除以缩放因子
        return input
    return input / scale_factor  # 返回反量化后的新张量


# 截断量化：将浮点数据截断到指定位数的量化范围内
def truncation(fp_data, nbits=8):
    # 计算整数位宽度
    il = torch.log2(torch.max(fp_data.max(), fp_data.min().abs())) + 1
    il = math.ceil(il - 1e-5)  # 向上取整
    # 计算小数位宽度
    qcode = nbits - il
    '''
    根据输入的数据，计算出相应的缩放因子
    '''
    scale_factor = 2 ** qcode  # 缩放因子
    # 获取量化范围
    clamp_min, clamp_max = get_quantized_range(nbits, signed=True)
    # 量化并截断
    q_data = linear_quantize_clamp(fp_data, scale_factor, clamp_min, clamp_max)
    # 反量化回浮点数
    q_data = linear_dequantize(q_data, scale_factor)
    return q_data, qcode


# 获取量化的默认参数：为不同类型的层提供默认量化配置
def get_default_kwargs_q(kwargs_q, layer_type):
    default = {
        'nbits': 4  # 默认量化位数为4位
    }
    # 为卷积层设置默认模式
    if isinstance(layer_type, _Conv2dQ):
        default.update({
            'mode': Qmodes.layer_wise})  # 默认层级量化
    elif isinstance(layer_type, _LinearQ):
        pass  # 线性层使用基础默认值
    elif isinstance(layer_type, _ActQ):
        pass  # 激活函数层使用基础默认值
    else:
        assert NotImplementedError  # 不支持的层类型
        return
    # 补充缺失的默认参数（将默认参数和用户传入的参数进行合并或者覆盖）
    for k, v in default.items():
        if k not in kwargs_q:
            kwargs_q[k] = v
    return kwargs_q


# 量化卷积层的基类，继承自PyTorch的Conv2d
'''
后续的量化卷积层会继承Conv2dQ,并在其基础上实现具体的量化逻辑
因此这个类时量化卷积的底层支撑，既包含基础配置，也为量化功能提供了必要的参数和接口
'''
class _Conv2dQ(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, **kwargs_q):
        # 调用父类构造函数初始化卷积层基本参数
        super(_Conv2dQ, self).__init__(in_channels, out_channels, kernel_size, stride=stride,
                                       padding=padding, dilation=dilation, groups=groups, bias=bias)
        # 获取量化默认参数
        self.kwargs_q = get_default_kwargs_q(kwargs_q, layer_type=self)
        self.nbits = kwargs_q['nbits']  # 量化位数
        # 如果位数为负数，不进行量化（不注册alpha参数）
        if self.nbits < 0:
            self.register_parameter('alpha', None)
            return
        self.q_mode = kwargs_q['mode']  # 量化模式
        # 根据量化模式初始化缩放因子alpha
        if self.q_mode == Qmodes.kernel_wise:
            # 核级量化：每个输出通道一个alpha
            self.alpha = Parameter(torch.Tensor(out_channels))
        else:  # 层级量化：整个层共享一个alpha
            self.alpha = Parameter(torch.Tensor(1))
        # 注册初始化状态缓冲区，用于标记量化相关参数是否完成初始化（0表示未初始化，1表示已初始化）
        self.register_buffer('init_state', torch.zeros(1))

    # 添加量化参数
    def add_param(self, param_k, param_v):
        self.kwargs_q[param_k] = param_v

    # 设置量化位数
    def set_bit(self, nbits):
        self.kwargs_q['nbits'] = nbits

    # 自定义打印信息
    def extra_repr(self):
        s_prefix = super(_Conv2dQ, self).extra_repr()  # 父类的打印信息
        if self.alpha is None:
            return '{}, fake'.format(s_prefix)  # 标记为伪量化（不实际量化）
        return '{}, {}'.format(s_prefix, self.kwargs_q)  # 包含量化参数


# 量化线性层的基类，继承自PyTorch的Linear
class _LinearQ(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, **kwargs_q):
        # 调用父类构造函数初始化线性层基本参数
        super(_LinearQ, self).__init__(in_features=in_features, out_features=out_features, bias=bias)
        # 获取量化默认参数
        self.kwargs_q = get_default_kwargs_q(kwargs_q, layer_type=self)
        self.nbits = kwargs_q['nbits']  # 量化位数
        # 如果位数为负数，不进行量化
        if self.nbits < 0:
            self.register_parameter('alpha', None)
            return
        self.q_mode = kwargs_q['mode']  # 量化模式
        # 根据量化模式初始化缩放因子alpha
        self.alpha = Parameter(torch.Tensor(1))  # 默认一个alpha
        if self.q_mode == Qmodes.kernel_wise:
            # 核级量化：每个输出特征一个alpha
            self.alpha = Parameter(torch.Tensor(out_features))
        # 注册初始化状态缓冲区
        self.register_buffer('init_state', torch.zeros(1))

    # 添加量化参数
    def add_param(self, param_k, param_v):
        self.kwargs_q[param_k] = param_v

    # 自定义打印信息
    def extra_repr(self):
        s_prefix = super(_LinearQ, self).extra_repr()  # 父类的打印信息
        if self.alpha is None:
            return '{}, fake'.format(s_prefix)  # 标记为伪量化
        return '{}, {}'.format(s_prefix, self.kwargs_q)  # 包含量化参数


# 量化激活函数的基类
class _ActQ(nn.Module):
    def __init__(self, in_features, **kwargs_q):
        super(_ActQ, self).__init__()
        # 获取量化默认参数
        self.kwargs_q = get_default_kwargs_q(kwargs_q, layer_type=self)
        self.nbits = kwargs_q['nbits']  # 量化位数
        # 如果位数为负数，不进行量化
        if self.nbits < 0:
            self.register_parameter('alpha', None)
            self.register_parameter('zero_point', None)
            return
        self.q_mode = kwargs_q['mode']  # 量化模式
        # 初始化缩放因子alpha和零点zero_point（用于非对称量化）
        self.alpha = Parameter(torch.Tensor(1))  # 默认一个alpha
        self.zero_point = Parameter(torch.Tensor([0]))  # 默认零点为0
        if self.q_mode == Qmodes.kernel_wise:
            # 核级量化：每个输入特征有独立的alpha和zero_point
            self.alpha = Parameter(torch.Tensor(in_features))
            self.zero_point = Parameter(torch.Tensor(in_features))
            torch.nn.init.zeros_(self.zero_point)  # 零点初始化为0
        # 注册初始化状态缓冲区和符号标记（是否有符号量化）
        self.register_buffer('init_state', torch.zeros(1))
        self.register_buffer('signed', torch.zeros(1))

    # 添加量化参数
    def add_param(self, param_k, param_v):
        self.kwargs_q[param_k] = param_v

    # 设置量化位数
    def set_bit(self, nbits):
        self.kwargs_q['nbits'] = nbits

    # 自定义打印信息
    def extra_repr(self):
        if self.alpha is None:
            return 'fake'  # 标记为伪量化
        return '{}'.format(self.kwargs_q)  # 包含量化参数

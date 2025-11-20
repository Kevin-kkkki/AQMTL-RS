# import torch
# import torch.nn as nn 
# from torch import Tensor
# import torch.nn.functional as F
# from torch.nn.modules.linear import Linear
# import math
# from torch.nn.parameter import Parameter
# from model._quan_base import _Conv2dQ, Qmodes, _LinearQ, _ActQ


# __all__ = ['Conv2dQ', 'LinearQ', 'ActQ']


# class FunQ(torch.autograd.Function):
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
#         # indicate_middle = torch.ones(indicate_small.shape).to(indicate_small.device) - indicate_small - indicate_big
#         indicate_middle = 1.0 - indicate_small - indicate_big  # Thanks to @haolibai
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


# class Conv2dQ(_Conv2dQ):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1,
#                  padding=0, dilation=1, groups=1, bias=True, nbits_w=8, mode=Qmodes.kernel_wise, **kwargs):
#         super(Conv2dQ, self).__init__(
#             in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
#             stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
#             nbits=nbits_w, mode=mode)
#         self.act = ActQ(in_features=in_channels, nbits_a=nbits_w)

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

#         # Method1: 31GB GPU memory (AlexNet w4a4 bs 2048) 17min/epoch
#         alpha = grad_scale(self.alpha, g)
#         alpha = alpha.unsqueeze(1).unsqueeze(2).unsqueeze(3)
#         w_q = round_pass((self.weight / alpha).clamp(Qn, Qp)) * alpha

#         x = self.act(x)
       
#         return F.conv2d(x, w_q, self.bias, self.stride,
#                         self.padding, self.dilation, self.groups)


# class LinearQ(_LinearQ):
#     def __init__(self, in_features, out_features, bias=True, nbits_w=4, **kwargs):
#         super(LinearQ, self).__init__(in_features=in_features,
#                                         out_features=out_features, bias=bias, nbits=nbits_w, mode=Qmodes.kernel_wise)
#         self.act = ActQ(in_features=in_features, nbits_a=nbits_w)

#     def forward(self, x):
#         if self.alpha is None:
#             return F.linear(x, self.weight, self.bias)
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

#         x = self.act(x)

#         return F.linear(x, w_q, self.bias)


# class ActQ(_ActQ):
#     def __init__(self, in_features, nbits_a=4, mode=Qmodes.kernel_wise, **kwargs):
#         super(ActQ, self).__init__(in_features=in_features, nbits=nbits_a, mode=mode)
#         # print(self.alpha.shape, self.zero_point.shape)
#     def forward(self, x):
#         if self.alpha is None:
#             return x

#         if self.training and self.init_state == 0:
#             if x.min() < -1e-5:
#                 self.signed.data.fill_(1)
#             if self.signed == 1:
#                 Qn = -2 ** (self.nbits - 1)
#                 Qp = 2 ** (self.nbits - 1) - 1
#             else:
#                 Qn = 0
#                 Qp = 2 ** self.nbits - 1
#             self.alpha.data.copy_(2 * x.abs().mean() / math.sqrt(Qp))
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
#         # x = round_pass((x / alpha).clamp(Qn, Qp)) * alpha
#         if len(x.shape)==2:
#             alpha = alpha.unsqueeze(0)
#             zero_point = zero_point.unsqueeze(0)
#         elif len(x.shape)==4:
#             alpha = alpha.unsqueeze(0).unsqueeze(2).unsqueeze(3)
#             zero_point = zero_point.unsqueeze(0).unsqueeze(2).unsqueeze(3)

#         x = round_pass((x / alpha + zero_point).clamp(Qn, Qp))
#         x = (x - zero_point) * alpha

#         return x



# class Q_attention(nn.MultiheadAttention):

#     def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False,
#                  kdim=None, vdim=None, batch_first=False, device=None, dtype=None) -> None:
#         factory_kwargs = {'device': device, 'dtype': dtype}
#         super(Q_attention, self).__init__()
#         self.out_proj = LinearQ(embed_dim, embed_dim, bias=bias, nbits_w=4, **factory_kwargs)




# 导入PyTorch相关模块
import torch
import torch.nn as nn 
from torch import Tensor
import torch.nn.functional as F
from torch.nn.modules.linear import Linear
import math
from torch.nn.parameter import Parameter
# 从自定义模块导入量化基础类
from model._quan_base import _Conv2dQ, Qmodes, _LinearQ, _ActQ

# 定义模块公开接口，只暴露指定的类
__all__ = ['Conv2dQ', 'LinearQ', 'ActQ']


# 定义自定义量化操作的自动求导函数
class FunQ(torch.autograd.Function):
    # 前向传播：执行量化操作
    @staticmethod
    def forward(ctx, weight, alpha, g, Qn, Qp):
        # 确保缩放因子alpha为正数
        assert alpha > 0, 'alpha = {}'.format(alpha)
        # 保存用于反向传播的张量
        ctx.save_for_backward(weight, alpha)
        # 保存其他参数
        ctx.other = g, Qn, Qp
        # 量化步骤：缩放->取整->截断到量化范围
        q_w = (weight / alpha).round().clamp(Qn, Qp)
        # 反量化：将量化值转回原范围
        w_q = q_w * alpha
        return w_q

    # 反向传播：计算梯度
    @staticmethod
    def backward(ctx, grad_weight):
        # 恢复保存的张量和参数
        weight, alpha = ctx.saved_tensors
        g, Qn, Qp = ctx.other
        # 计算缩放后的权重
        q_w = weight / alpha
        # 标记小于量化范围最小值的部分
        indicate_small = (q_w < Qn).float()
        # 标记大于量化范围最大值的部分
        indicate_big = (q_w > Qp).float()
        # 标记在量化范围内的部分
        indicate_middle = 1.0 - indicate_small - indicate_big
        
        # 计算缩放因子alpha的梯度
        grad_alpha = ((indicate_small * Qn + indicate_big * Qp + indicate_middle * (
            -q_w + q_w.round())) * grad_weight * g).sum().unsqueeze(dim=0)
        # 计算权重的梯度（只保留量化范围内的梯度）
        grad_weight = indicate_middle * grad_weight
        
        return grad_weight, grad_alpha, None, None, None


# 定义梯度缩放函数，用于调整量化参数的梯度
def grad_scale(x, scale):
    y = x  # 前向传播时返回原张量
    y_grad = x * scale  # 反向传播时使用缩放后的梯度
    # 通过detach操作分离计算图，实现前向和反向路径的分离
    return y.detach() - y_grad.detach() + y_grad


# 定义带直通估计器的取整函数，解决量化中取整操作不可导问题
def round_pass(x):
    y = x.round()  # 前向传播时执行取整
    y_grad = x  # 反向传播时直接使用原输入的梯度
    # 分离计算图，实现前向取整、反向直通
    return y.detach() - y_grad.detach() + y_grad


# 定义量化卷积层，继承自基础量化卷积类
class Conv2dQ(_Conv2dQ):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, nbits_w=8, mode=Qmodes.kernel_wise, **kwargs):
        # 调用父类构造函数初始化基础卷积参数和量化配置
        super(Conv2dQ, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
            nbits=nbits_w, mode=mode)
        # 初始化量化激活函数，与卷积层共享量化位数
        self.act = ActQ(in_features=in_channels, nbits_a=nbits_w)

    def forward(self, x):
        # 如果没有量化参数alpha，直接使用普通卷积
        if self.alpha is None:
            return F.conv2d(x, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        
        # 计算量化范围（有符号量化）
        Qn = -2 **(self.nbits - 1)
        Qp = 2** (self.nbits - 1) - 1
        
        # 训练模式下且未初始化时，初始化缩放因子alpha
        if self.training and self.init_state == 0:
            self.alpha.data.copy_(2 * self.weight.abs().mean() / math.sqrt(Qp))
            self.init_state.fill_(1)
            
        # 计算梯度缩放因子g
        g = 1.0 / math.sqrt(self.weight.numel() * Qp)

        # 对alpha进行梯度缩放
        alpha = grad_scale(self.alpha, g)
        # 调整alpha形状以匹配权重
        alpha = alpha.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        # 执行权重量化：缩放->截断->取整->反量化
        w_q = round_pass((self.weight / alpha).clamp(Qn, Qp)) * alpha

        # 对输入进行激活量化
        x = self.act(x)
       
        # 使用量化后的权重执行卷积操作
        return F.conv2d(x, w_q, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


# 定义量化线性层，继承自基础量化线性类
class LinearQ(_LinearQ):
    def __init__(self, in_features, out_features, bias=True, nbits_w=4, **kwargs):
        # 调用父类构造函数初始化基础线性层参数和量化配置
        super(LinearQ, self).__init__(in_features=in_features,
                                        out_features=out_features, bias=bias, nbits=nbits_w, mode=Qmodes.kernel_wise)
        # 初始化量化激活函数
        self.act = ActQ(in_features=in_features, nbits_a=nbits_w)

    def forward(self, x):
        # 如果没有量化参数alpha，直接使用普通线性变换
        if self.alpha is None:
            return F.linear(x, self.weight, self.bias)
        
        # 计算量化范围（有符号量化）
        Qn = -2 **(self.nbits - 1)
        Qp = 2** (self.nbits - 1) - 1
        
        # 训练模式下且未初始化时，初始化缩放因子alpha
        if self.training and self.init_state == 0:
            self.alpha.data.copy_(2 * self.weight.abs().mean() / math.sqrt(Qp))
            self.init_state.fill_(1)
            
        # 计算梯度缩放因子g
        g = 1.0 / math.sqrt(self.weight.numel() * Qp)

        # 对alpha进行梯度缩放
        alpha = grad_scale(self.alpha, g)
        # 调整alpha形状以匹配权重
        alpha = alpha.unsqueeze(1)
        # 执行权重量化
        w_q = round_pass((self.weight / alpha).clamp(Qn, Qp)) * alpha

        # 对输入进行激活量化
        x = self.act(x)

        # 使用量化后的权重执行线性变换
        return F.linear(x, w_q, self.bias)


# 定义量化激活函数类，继承自基础量化激活类
class ActQ(_ActQ):
    def __init__(self, in_features, nbits_a=4, mode=Qmodes.kernel_wise, **kwargs):
        # 调用父类构造函数初始化量化参数
        super(ActQ, self).__init__(in_features=in_features, nbits=nbits_a, mode=mode)
        # 打印alpha和zero_point的形状（调试用）
        # print(self.alpha.shape, self.zero_point.shape)
    
    def forward(self, x):
        # 如果没有量化参数alpha，直接返回输入
        if self.alpha is None:
            return x

        # 训练模式下且未初始化时，初始化量化参数
        if self.training and self.init_state == 0:
            # 判断激活值是否为有符号（存在负值）
            if x.min() < -1e-5:
                self.signed.data.fill_(1)
            # 根据是否有符号设置量化范围
            if self.signed == 1:
                Qn = -2 **(self.nbits - 1)
                Qp = 2** (self.nbits - 1) - 1
            else:
                Qn = 0
                Qp = 2 ** self.nbits - 1
            # 初始化缩放因子alpha
            self.alpha.data.copy_(2 * x.abs().mean() / math.sqrt(Qp))
            # 初始化零点zero_point（非对称量化）
            self.zero_point.data.copy_(self.zero_point.data * 0.9 + 0.1 * (torch.min(x.detach()) - self.alpha.data * Qn))
            # 标记为已初始化
            self.init_state.fill_(1)

        # 根据是否有符号设置量化范围
        if self.signed == 1:
            Qn = -2 **(self.nbits - 1)
            Qp = 2** (self.nbits - 1) - 1
        else:
            Qn = 0
            Qp = 2 ** self.nbits - 1

        # 计算梯度缩放因子g
        g = 1.0 / math.sqrt(x.numel() * Qp)

        # 对零点zero_point应用取整直通操作
        zero_point = (self.zero_point.round() - self.zero_point).detach() + self.zero_point
        # 对alpha和zero_point进行梯度缩放
        alpha = grad_scale(self.alpha, g)
        zero_point = grad_scale(zero_point, g)
        
        # 根据输入维度调整alpha和zero_point的形状，确保广播正确
        if len(x.shape)==2:  # 2D张量（如线性层输入）
            alpha = alpha.unsqueeze(0)
            zero_point = zero_point.unsqueeze(0)
        elif len(x.shape)==4:  # 4D张量（如卷积层输入）
            alpha = alpha.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            zero_point = zero_point.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        # 执行激活量化：缩放+零点偏移->截断->取整->反量化
        x = round_pass((x / alpha + zero_point).clamp(Qn, Qp))
        x = (x - zero_point) * alpha

        return x


# 定义量化注意力机制类，继承自PyTorch的多头注意力
class Q_attention(nn.MultiheadAttention):

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False,
                 kdim=None, vdim=None, batch_first=False, device=None, dtype=None) -> None:
        # 准备设备和数据类型参数
        factory_kwargs = {'device': device, 'dtype': dtype}
        # 调用父类构造函数初始化注意力机制
        super(Q_attention, self).__init__()
        # 将输出投影层替换为量化线性层，使用4位量化
        self.out_proj = LinearQ(embed_dim, embed_dim, bias=bias, nbits_w=4, **factory_kwargs)
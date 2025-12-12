# 1. 继承全精度（FP32）的基础配置
_base_ = ['./MTL_slvlcls_swin-t-p4-w7_1x1_resisc&dior&potsdam.py']

# 2. 开启量化开关
quantize = True
model = dict(quantize=True)

# 3. 调整训练计划 (恢复到标准长周期)
# 将总迭代次数设定为 30万
runner = dict(type='IterBasedRunner', max_iters=300000)

# 调整学习率衰减点 (按比例扩展: 15w时的12w/14w -> 30w时的24w/28w)
lr_config = dict(policy='step', step=[240000, 280000])

# 4. 调整保存和评估频率 (由于总时间变长，适当放宽间隔以节省时间和磁盘)
# 每 20,000 次保存一次权重 (原 10,000)
checkpoint_config = dict(interval=20000)

# 每 10,000 次评估一次验证集 (原 5,000)
evaluation = dict(interval=10000)
# 1. 继承全精度（FP32）的基础配置
# 这样 backbone, head, data 等所有不需要改的参数都会自动读进来
_base_ = ['./MTL_slvlcls_swin-t-p4-w7_1x1_resisc&dior&potsdam.py']

# 2. 开启量化开关
# 对应 tools/train.py 中的 if cfg.get('quantize', False):
quantize = True

# 对应 models/multi/multitask_learner.py 中的 self.quantize
model = dict(quantize=True)

# 3. 调整训练计划 (针对微调缩短时间)
# 将总迭代次数从 30万 减少到 15万
runner = dict(type='IterBasedRunner', max_iters=150000)

# 调整学习率衰减点 (按比例提前，例如在 12w 和 14w 处衰减)
lr_config = dict(policy='step', step=[120000, 140000])

# 4. 加密保存和评估频率 (因为总时间变短了)
checkpoint_config = dict(interval=10000)
evaluation = dict(interval=5000)
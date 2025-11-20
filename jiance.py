import torch
import sys

# --- 请修改这里 ---
checkpoint_path = '/media/at03/4tb/jiangrongkai_workspace/RSCoTr-master/chickpoint/iter_300000.pth'  # 替换为你刚才训练好的权重路径
# ----------------

try:
    print(f"正在加载权重: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['state_dict']
except Exception as e:
    print(f"加载失败: {e}")
    sys.exit(1)

has_nan = False
print("开始检查参数数值...")

for key, param in state_dict.items():
    # 检查是否包含 NaN (非数字)
    if torch.isnan(param).any():
        print(f"❌ 发现 NaN: {key}")
        has_nan = True
        # 不需要打印所有，发现一个就足以说明文件坏了
        break 
    
    # 检查是否包含 Inf (无穷大)
    if torch.isinf(param).any():
        print(f"❌ 发现 Inf: {key}")
        has_nan = True
        break

    # 检查数值是否过大 (这也可能是梯度爆炸的前兆)
    if param.abs().max() > 1e4: # 阈值可调，通常权重不会这么大
        print(f"⚠️ 发现异常大值 (Max={param.abs().max().item():.2f}): {key}")

if has_nan:
    print("\n【结论】: 这个权重文件已经损坏（包含无效数值）。")
    print("原因：在 FP32 预热训练期间，梯度可能已经爆炸了，但没报错退出，只是把坏数值存进来了。")
else:
    print("\n【结论】: 这个权重文件是健康的！数值正常。")
    print("原因：问题出在 QAT 训练开始后的瞬间（学习率太高或配置问题），导致模型瞬间崩溃。")
"""
Mini-GPT 训练脚本

训练的核心循环只有三步：
1. 前向传播：把数据喂给模型，得到预测
2. 算误差：预测和正确答案差多少（loss）
3. 反向传播：根据误差调整参数

重复几千次，模型就学会了。
"""

import sys
import torch
import torch.nn.functional as F
from model.tokenizer import BPETokenizer
from model.gpt import MiniGPT


# ============================================================
# 第一步：准备数据
# ============================================================

# 读取训练文本（可通过第二个参数指定文件，默认用千字文）
data_file = sys.argv[2] if len(sys.argv) > 2 else 'data/sample.txt'
with open(data_file, 'r') as f:
    text = f.read()
print(f'训练数据: {data_file}')

# 创建 BPE tokenizer，从文本学习分词规则
tokenizer = BPETokenizer.train([text], target_vocab_size=8000)
data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

print(f'训练文本: {len(text)} 个字符')
print(f'词表大小: {tokenizer.vocab_size}')
print(f'编码后: {len(data)} 个 token')


# ============================================================
# 第二步：创建模型
# ============================================================

# 超参数（可以调的旋钮）
d_model = 256         # 向量维度（64→256，信息容量大4倍）
n_heads = 8           # 注意力头数（4→8，更多视角）
n_layers = 8          # 变换器块层数（4→8，理解更深）
context_length = 128  # 上下文窗口（64→128，看更远）
learning_rate = 1e-3  # 学习率：每步调整参数的幅度
max_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 2000  # 命令行指定步数，默认 2000
print_every = max(1, max_steps // 20)  # 自动调整打印频率

# 选设备：优先用 Mac GPU（MPS），否则用 CPU
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f'设备: {device}')

# 创建模型
model = MiniGPT(
    vocab_size=tokenizer.vocab_size,
    d_model=d_model,
    n_heads=n_heads,
    n_layers=n_layers,
    context_length=context_length,
).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f'模型参数量: {total_params:,}')

# 优化器：负责根据梯度调整参数
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# ============================================================
# 第三步：准备训练数据的获取方式
# ============================================================

def get_batch(batch_size=4):
    """
    从训练数据中随机取一批样本。

    每个样本是一段连续文本：
    - 输入 x: 前 context_length 个字
    - 目标 y: 后 context_length 个字（就是 x 整体往后移一位）

    比如 context_length=4，文本是"天地玄黄宇宙洪荒"：
    - x = "天地玄黄"
    - y = "地玄黄宇"
    模型要学的就是：看到"天"预测"地"，看到"天地"预测"玄"，以此类推。
    """
    # 随机选 batch_size 个起始位置
    max_start = len(data) - context_length - 1
    starts = torch.randint(0, max_start, (batch_size,))

    # 取出输入和目标
    x = torch.stack([data[s:s + context_length] for s in starts])
    y = torch.stack([data[s + 1:s + 1 + context_length] for s in starts])

    return x.to(device), y.to(device)


# ============================================================
# 第四步：训练循环（核心！）
# ============================================================

print(f'\n开始训练（共 {max_steps} 步）...\n')

for step in range(max_steps):

    # --- 1. 前向传播 ---
    x, y = get_batch()              # 取一批数据
    logits = model(x)               # 模型预测，输出 (B, T, vocab_size) 的分数

    # --- 2. 算误差（loss） ---
    # 把形状调整一下，才能计算
    B, T, V = logits.shape
    logits = logits.view(B * T, V)  # (B*T, vocab_size)
    y = y.view(B * T)               # (B*T,)

    loss = F.cross_entropy(logits, y)  # 交叉熵：预测分数和正确答案之间的差距

    # --- 3. 反向传播 + 更新参数 ---
    optimizer.zero_grad()  # 清除上一步的梯度
    loss.backward()        # 反向传播：计算每个参数的梯度
    optimizer.step()       # 更新参数：参数 = 参数 - 学习率 × 梯度

    # 打印进度
    if step % print_every == 0:
        print(f'步骤 {step:>4d}/{max_steps} | loss = {loss.item():.4f}')

print(f'\n训练完成！最终 loss = {loss.item():.4f}')

# 保存模型
save_path = f'checkpoints/mini-gpt-{max_steps}steps.pt'
torch.save({
    'model': model.state_dict(),
    'tokenizer_type': 'bpe',
    'tokenizer_data': tokenizer.save_vocab(),
    'config': {
        'vocab_size': tokenizer.vocab_size,
        'd_model': d_model,
        'n_heads': n_heads,
        'n_layers': n_layers,
        'context_length': context_length,
    }
}, save_path)
print(f'模型已保存到 {save_path}')


# ============================================================
# 第五步：生成文本（看看模型学到了什么）
# ============================================================

def generate(prompt, max_new_tokens=100):
    """
    给一个开头，让模型续写。

    过程：
    1. 把开头文字编码成数字
    2. 喂给模型，得到下一个字的预测分数
    3. 从分数中选一个字（按概率采样）
    4. 把这个字加到序列后面
    5. 重复 2-4
    """
    model.eval()  # 切换到评估模式

    # 编码输入
    tokens = tokenizer.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():  # 生成时不需要算梯度
        for _ in range(max_new_tokens):
            # 如果序列太长，只取最后 context_length 个
            input_tokens = tokens[:, -context_length:]

            # 前向传播
            logits = model(input_tokens)

            # 只看最后一个位置的预测（下一个字）
            next_logits = logits[:, -1, :]  # (1, vocab_size)

            # 转成概率
            probs = F.softmax(next_logits, dim=-1)

            # 按概率随机选一个字（不总是选最高的，增加多样性）
            next_token = torch.multinomial(probs, num_samples=1)

            # 加到序列后面
            tokens = torch.cat([tokens, next_token], dim=1)

    # 解码回文字
    result = tokenizer.decode(tokens[0].tolist())
    return result


# 试试生成效果
print('\n===== 生成测试 =====\n')

prompts = ['嵌套学习', '注意力', '模型']
for p in prompts:
    result = generate(p, max_new_tokens=20)
    print(f'输入: "{p}"')
    print(f'输出: "{result}"')
    print()

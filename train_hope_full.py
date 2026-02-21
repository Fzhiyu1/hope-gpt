"""
完整 Hope 模型训练脚本（Self-Modifying Titans + CMS）

支持从头训练和加载 checkpoint 继续训练。

用法：
  python3 train_hope_full.py                                      # 从头训练，千字文，2000步
  python3 train_hope_full.py 5000                                  # 从头训练，5000步
  python3 train_hope_full.py 5000 data/poetry_tang.txt             # 从头训练，指定数据
  python3 train_hope_full.py 2000 data/sample.txt checkpoint.pt    # 加载 checkpoint 继续训练

参数可以任意顺序：
  - 数字 → 训练步数
  - .txt 文件 → 训练数据
  - .pt 文件 → 加载的 checkpoint（继续训练）
"""

import sys
import os
import torch
import torch.nn.functional as F
from model.tokenizer import BPETokenizer, load_tokenizer_from_checkpoint
from model.hope import HopeGPT
from model.m3_optimizer import M3Optimizer


# ============================================================
# 第一步：解析参数
# ============================================================

max_steps = 2000
data_file = 'data/sample.txt'
resume_path = None

for arg in sys.argv[1:]:
    if arg.endswith('.pt'):
        resume_path = arg
    elif arg.endswith('.txt'):
        data_file = arg
    else:
        try:
            max_steps = int(arg)
        except ValueError:
            print(f'未知参数: {arg}')

if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'
print(f'设备: {device}')


# ============================================================
# 第二步：准备数据和模型
# ============================================================

# 超参数
d_model = 256
n_layers = 8
d_memory = 256          # Titans 记忆模块内部维度
chunk_size = 16         # Titans 分块大小
context_length = 128
learning_rate = 3e-4    # 比 Hope-Attention 的 1e-3 更小（DGD 需要更稳的外层学习率）
cms_chunk_sizes = [1, 16, 128, 1024]

if resume_path:
    # ---- 继续训练：从 checkpoint 加载 ----
    print(f'\n加载 checkpoint: {resume_path}')
    checkpoint = torch.load(resume_path, weights_only=False)
    config = checkpoint['config']
    total_steps_so_far = checkpoint.get('total_steps', 0)

    tokenizer = load_tokenizer_from_checkpoint(checkpoint)

    with open(data_file, 'r') as f:
        text = f.read()
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    model = HopeGPT(**config).to(device)
    model.load_state_dict(checkpoint['model'])

    optimizer = M3Optimizer(model.parameters(), lr=learning_rate, slow_interval=chunk_size)
    if 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])

    print(f'已加载模型（{sum(p.numel() for p in model.parameters()):,} 参数）')
    print(f'已完成步数: {total_steps_so_far}')
    print(f'本次训练: {max_steps} 步（步数从 {total_steps_so_far} 继续）')

else:
    # ---- 从头训练 ----
    total_steps_so_far = 0

    with open(data_file, 'r', encoding='utf-8') as f:
        text = f.read()

    tokenizer = BPETokenizer.train([text], target_vocab_size=8000)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    model = HopeGPT(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        d_memory=d_memory,
        chunk_size=chunk_size,
        context_length=context_length,
        cms_chunk_sizes=cms_chunk_sizes,
    ).to(device)

    optimizer = M3Optimizer(model.parameters(), lr=learning_rate, slow_interval=chunk_size)

    print(f'从头开始训练')
    print(f'模型参数量: {sum(p.numel() for p in model.parameters()):,}')

print(f'训练数据: {data_file}（{len(text)} 字符，{len(data)} tokens）')
print(f'词表大小: {tokenizer.vocab_size}')
print(f'Titans chunk_size: {chunk_size}')
print(f'CMS 频率层级: {cms_chunk_sizes}')


# ============================================================
# 第三步：训练
# ============================================================

def get_batch(batch_size=4):
    max_start = len(data) - context_length - 1
    starts = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([data[s:s + context_length] for s in starts])
    y = torch.stack([data[s + 1:s + 1 + context_length] for s in starts])
    return x.to(device), y.to(device)


warmup_steps = min(100, max_steps // 10)
print_every = max(1, max_steps // 20)
print(f'\n开始训练（本次 {max_steps} 步，全局步数 {total_steps_so_far} → '
      f'{total_steps_so_far + max_steps}，warmup {warmup_steps} 步）...\n')

for step in range(max_steps):
    global_step = total_steps_so_far + step

    # 学习率 warmup（M3 前几步偏差校正会放大更新，需要小 lr 稳定）
    if step < warmup_steps:
        warmup_factor = (step + 1) / warmup_steps
        for pg in optimizer.param_groups:
            pg['lr'] = learning_rate * warmup_factor
    elif step == warmup_steps:
        for pg in optimizer.param_groups:
            pg['lr'] = learning_rate

    # CMS 频率调度
    model.set_active_levels(global_step)

    x, y = get_batch()
    logits = model(x)
    B, T, V = logits.shape
    loss = F.cross_entropy(logits.view(B * T, V), y.view(B * T))

    optimizer.zero_grad()
    loss.backward()

    # 梯度裁剪（DGD 的记忆模块初始值梯度可能很大）
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()

    if step % print_every == 0:
        active_count = sum(1 for cs in cms_chunk_sizes if global_step % cs == 0)
        print(f'步骤 {global_step:>6d}（本次 {step:>5d}/{max_steps}）| '
              f'loss = {loss.item():.4f} | 活跃CMS: {active_count}/{len(cms_chunk_sizes)}')

total_steps_so_far += max_steps
print(f'\n训练完成！最终 loss = {loss.item():.4f}')
print(f'累计训练步数: {total_steps_so_far}')


# ============================================================
# 第四步：保存模型
# ============================================================

os.makedirs('checkpoints', exist_ok=True)
save_path = f'checkpoints/hope-full-{total_steps_so_far}steps.pt'

if isinstance(tokenizer, BPETokenizer):
    tok_save = {
        'tokenizer_type': 'bpe',
        'tokenizer_data': tokenizer.save_vocab(),
    }
else:
    tok_save = {
        'tokenizer_text': checkpoint['tokenizer_text'] if resume_path else text,
    }

save_data = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    **tok_save,
    'total_steps': total_steps_so_far,
    'config': {
        'vocab_size': tokenizer.vocab_size,
        'd_model': d_model,
        'n_layers': n_layers,
        'd_memory': d_memory,
        'chunk_size': chunk_size,
        'context_length': context_length,
        'cms_chunk_sizes': cms_chunk_sizes,
    },
    'model_type': 'hope-full',
}
torch.save(save_data, save_path)
print(f'模型已保存到 {save_path}')

latest_path = 'checkpoints/hope-full-latest.pt'
torch.save(save_data, latest_path)
print(f'最新模型: {latest_path}')
print(f'\n下次继续训练: python3 train_hope_full.py 2000 {data_file} {latest_path}')


# ============================================================
# 第五步：生成文本
# ============================================================

def generate(prompt, max_new_tokens=100):
    model.eval()
    tokens = tokenizer.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            input_tokens = tokens[:, -context_length:]
            logits = model(input_tokens)
            next_logits = logits[:, -1, :]
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_token], dim=1)

    return tokenizer.decode(tokens[0].tolist())


print('\n===== 生成测试 =====\n')
prompts = ['天地', '明月', '春風']
for p in prompts:
    result = generate(p, max_new_tokens=20)
    print(f'输入: "{p}"')
    print(f'输出: "{result}"')
    print()

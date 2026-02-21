"""
Hope-Attention 训练脚本

支持从头训练和加载 checkpoint 继续训练。

用法：
  python3 train_hope.py                                      # 从头训练，千字文，2000步
  python3 train_hope.py 5000                                  # 从头训练，5000步
  python3 train_hope.py 5000 data/nested_learning.txt         # 从头训练，指定数据
  python3 train_hope.py 2000 data/sample.txt checkpoint.pt    # 加载 checkpoint 继续训练

参数可以任意顺序：
  - 数字 → 训练步数
  - .txt 文件 → 训练数据
  - .pt 文件 → 加载的 checkpoint（继续训练）
"""

import sys
import torch
import torch.nn.functional as F
from model.tokenizer import BPETokenizer, load_tokenizer_from_checkpoint
from model.hope_attention import HopeAttentionGPT


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

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f'设备: {device}')


# ============================================================
# 第二步：准备数据和模型
# ============================================================

# 超参数
d_model = 256
n_heads = 8
n_layers = 8
context_length = 128
learning_rate = 1e-3
chunk_sizes = [1, 16, 128, 1024]

if resume_path:
    # ---- 继续训练：从 checkpoint 加载 ----
    print(f'\n加载 checkpoint: {resume_path}')
    checkpoint = torch.load(resume_path, weights_only=False)
    config = checkpoint['config']
    total_steps_so_far = checkpoint.get('total_steps', 0)

    # 用 checkpoint 重建 tokenizer（自动识别 BPE 或 Char）
    tokenizer = load_tokenizer_from_checkpoint(checkpoint)

    # 读取训练数据并编码（BPE 有 <UNK>，不会崩溃）
    with open(data_file, 'r') as f:
        text = f.read()
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    # 重建模型并加载参数
    model = HopeAttentionGPT(**config).to(device)
    model.load_state_dict(checkpoint['model'])

    # 重建优化器并加载状态（如果有）
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])

    print(f'已加载模型（{sum(p.numel() for p in model.parameters()):,} 参数）')
    print(f'已完成步数: {total_steps_so_far}')
    print(f'本次训练: {max_steps} 步（步数从 {total_steps_so_far} 继续）')

else:
    # ---- 从头训练 ----
    total_steps_so_far = 0

    with open(data_file, 'r') as f:
        text = f.read()

    tokenizer = BPETokenizer.train([text], target_vocab_size=8000)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    model = HopeAttentionGPT(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        context_length=context_length,
        chunk_sizes=chunk_sizes,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(f'从头开始训练')
    print(f'模型参数量: {sum(p.numel() for p in model.parameters()):,}')

print(f'训练数据: {data_file}（{len(text)} 字符，{len(data)} tokens）')
print(f'词表大小: {tokenizer.vocab_size}')
print(f'CMS 频率层级: {chunk_sizes}')


# ============================================================
# 第三步：训练
# ============================================================

def get_batch(batch_size=4):
    max_start = len(data) - context_length - 1
    starts = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([data[s:s + context_length] for s in starts])
    y = torch.stack([data[s + 1:s + 1 + context_length] for s in starts])
    return x.to(device), y.to(device)


print_every = max(1, max_steps // 20)
print(f'\n开始训练（本次 {max_steps} 步，全局步数 {total_steps_so_far} → '
      f'{total_steps_so_far + max_steps}）...\n')

for step in range(max_steps):
    global_step = total_steps_so_far + step

    # CMS 核心：根据全局步数设置活跃层级
    model.set_active_levels(global_step)

    x, y = get_batch()
    logits = model(x)
    B, T, V = logits.shape
    loss = F.cross_entropy(logits.view(B * T, V), y.view(B * T))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % print_every == 0:
        active_count = sum(1 for cs in chunk_sizes if global_step % cs == 0)
        print(f'步骤 {global_step:>6d}（本次 {step:>5d}/{max_steps}）| '
              f'loss = {loss.item():.4f} | 活跃层级: {active_count}/{len(chunk_sizes)}')

total_steps_so_far += max_steps
print(f'\n训练完成！最终 loss = {loss.item():.4f}')
print(f'累计训练步数: {total_steps_so_far}')


# ============================================================
# 第四步：保存模型
# ============================================================

save_path = f'checkpoints/hope-attention-{total_steps_so_far}steps.pt'

# 构建 tokenizer 保存数据
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
        'n_heads': n_heads,
        'n_layers': n_layers,
        'context_length': context_length,
        'chunk_sizes': chunk_sizes,
    },
    'model_type': 'hope-attention',
}
torch.save(save_data, save_path)
print(f'模型已保存到 {save_path}')

# 同时保存一个 latest，方便下次继续
latest_path = 'checkpoints/hope-attention-latest.pt'
torch.save(save_data, latest_path)
print(f'最新模型: {latest_path}')
print(f'\n下次继续训练: python3 train_hope.py 2000 {data_file} {latest_path}')


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
prompts = ['天地', '嵌套学习', '模型']
for p in prompts:
    try:
        result = generate(p, max_new_tokens=20)
        print(f'输入: "{p}"')
        print(f'输出: "{result}"')
        print()
    except KeyError as e:
        print(f'跳过 "{p}"（包含模型不认识的字: {e}）\n')

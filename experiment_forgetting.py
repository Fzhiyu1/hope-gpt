"""
抗遗忘实验：mini-GPT vs Hope-Attention（混合训练版）

实验目的：
  验证 CMS（连续记忆系统）在混合训练场景下的抗遗忘效果。

实验设计：
  第一阶段：两个模型都只训练唐诗（建立基线）
  第二阶段：训练宋词，同时按比例混入唐诗（模拟真实连续学习）
  第三阶段：测量唐诗遗忘程度

  混合比例：测试多个旧数据混入比例（0%、5%、20%）
  - 0% = 纯宋词（硬切换，之前的实验方式）
  - 5% = 95%宋词 + 5%唐诗（CMS 的优势场景）
  - 20% = 80%宋词 + 20%唐诗（充分混合）

预期结果：
  - 0% 混合：两者都严重遗忘，差别不大
  - 5% 混合：Hope-Attention 显著优于 mini-GPT（CMS 低频FFN利用少量旧数据巩固记忆）
  - 20% 混合：两者遗忘都不严重，差距缩小

用法：
  python3 experiment_forgetting.py
  python3 experiment_forgetting.py 1500    # 指定每阶段训练步数
"""

import sys
import os
import time
import torch
import torch.nn.functional as F
from model.tokenizer import BPETokenizer
from model.gpt import MiniGPT
from model.hope_attention import HopeAttentionGPT


# ============================================================
# 配置
# ============================================================

d_model = 256
n_heads = 8
n_layers = 8
context_length = 128
chunk_sizes = [1, 16, 128, 1024]
learning_rate = 1e-3
steps_per_phase = int(sys.argv[1]) if len(sys.argv) > 1 else 1500

# 混合比例：第二阶段中旧数据（唐诗）的占比
mix_ratios = [0.0, 0.05, 0.20]

device = 'mps' if torch.backends.mps.is_available() else 'cpu'

report = []
def log(msg):
    print(msg)
    report.append(msg)

log(f'设备: {device}')
log(f'每阶段训练步数: {steps_per_phase}')
log(f'混合比例: {mix_ratios}')


# ============================================================
# 准备数据
# ============================================================

with open('data/poetry_tang.txt', 'r') as f:
    text_tang = f.read()
with open('data/poetry_songci.txt', 'r') as f:
    text_songci = f.read()

# BPE 词表缓存
vocab_cache = 'checkpoints/bpe-vocab-tang-songci.pt'
if os.path.exists(vocab_cache):
    tokenizer = BPETokenizer.load_from_file(vocab_cache)
else:
    tokenizer = BPETokenizer.train([text_tang, text_songci], target_vocab_size=8000)
    os.makedirs('checkpoints', exist_ok=True)
    tokenizer.save_to_file(vocab_cache)

data_tang = torch.tensor(tokenizer.encode(text_tang), dtype=torch.long)
data_songci = torch.tensor(tokenizer.encode(text_songci), dtype=torch.long)

log(f'唐诗: {len(text_tang)} 字 → {len(data_tang)} tokens')
log(f'宋词: {len(text_songci)} 字 → {len(data_songci)} tokens')
log(f'BPE 词表: {tokenizer.vocab_size}')


# ============================================================
# 工具函数
# ============================================================

def get_batch(data, batch_size=4):
    """从数据中随机取一批样本"""
    max_start = len(data) - context_length - 1
    starts = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([data[s:s + context_length] for s in starts])
    y = torch.stack([data[s + 1:s + 1 + context_length] for s in starts])
    return x.to(device), y.to(device)


def get_mixed_batch(data_new, data_old, mix_ratio, batch_size=4):
    """
    混合取批次：mix_ratio 是旧数据的比例。

    比如 mix_ratio=0.05, batch_size=4 时：
    大部分 batch 全部来自新数据，偶尔有一个来自旧数据。
    """
    # 每个样本独立决定来自新数据还是旧数据
    x_list, y_list = [], []
    for _ in range(batch_size):
        if torch.rand(1).item() < mix_ratio:
            data = data_old
        else:
            data = data_new
        max_start = len(data) - context_length - 1
        start = torch.randint(0, max_start, (1,)).item()
        x_list.append(data[start:start + context_length])
        y_list.append(data[start + 1:start + 1 + context_length])
    x = torch.stack(x_list).to(device)
    y = torch.stack(y_list).to(device)
    return x, y


def eval_loss(model, data, num_batches=50):
    """在指定数据上测量 loss"""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for _ in range(num_batches):
            x, y = get_batch(data)
            logits = model(x)
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(B * T, V), y.view(B * T))
            total_loss += loss.item()
    model.train()
    return total_loss / num_batches


def generate(model, prompt, max_new=30):
    """用模型生成文本"""
    model.eval()
    tokens = tokenizer.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        for _ in range(max_new):
            input_tokens = tokens[:, -context_length:]
            logits = model(input_tokens)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_token], dim=1)
    model.train()
    return tokenizer.decode(tokens[0].tolist())


def train_model_mixed(model, data_new, data_old, mix_ratio, num_steps,
                      optimizer, model_name, is_hope=False, step_offset=0):
    """
    混合训练：每个 batch 按 mix_ratio 混入旧数据。
    mix_ratio=0 等价于纯新数据训练（硬切换）。
    """
    print_every = max(1, num_steps // 10)

    for step in range(num_steps):
        global_step = step + step_offset

        if is_hope:
            model.set_active_levels(global_step)

        if mix_ratio > 0:
            x, y = get_mixed_batch(data_new, data_old, mix_ratio)
        else:
            x, y = get_batch(data_new)

        logits = model(x)
        B, T, V = logits.shape
        loss = F.cross_entropy(logits.view(B * T, V), y.view(B * T))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % print_every == 0:
            print(f'  [{model_name}] 步骤 {step:>5d}/{num_steps} | '
                  f'loss = {loss.item():.4f}')

    return loss.item()


# ============================================================
# 第一阶段：训练唐诗（只需训练一次，保存 checkpoint）
# ============================================================

log(f'\n{"="*60}')
log(f'第一阶段：训练唐诗（{steps_per_phase} 步）')
log(f'{"="*60}')

t1 = time.time()

model_gpt = MiniGPT(
    vocab_size=tokenizer.vocab_size,
    d_model=d_model, n_heads=n_heads,
    n_layers=n_layers, context_length=context_length,
).to(device)

model_hope = HopeAttentionGPT(
    vocab_size=tokenizer.vocab_size,
    d_model=d_model, n_heads=n_heads,
    n_layers=n_layers, context_length=context_length,
    chunk_sizes=chunk_sizes,
).to(device)

gpt_params = sum(p.numel() for p in model_gpt.parameters())
hope_params = sum(p.numel() for p in model_hope.parameters())
log(f'mini-GPT 参数量:       {gpt_params:>12,}')
log(f'Hope-Attention 参数量: {hope_params:>12,}')

opt_gpt = torch.optim.Adam(model_gpt.parameters(), lr=learning_rate)
opt_hope = torch.optim.Adam(model_hope.parameters(), lr=learning_rate)

log(f'\n--- mini-GPT ---')
train_model_mixed(model_gpt, data_tang, data_tang, 0, steps_per_phase,
                  opt_gpt, 'mini-GPT')

log(f'\n--- Hope-Attention ---')
train_model_mixed(model_hope, data_tang, data_tang, 0, steps_per_phase,
                  opt_hope, 'Hope', is_hope=True)

loss_gpt_phase1 = eval_loss(model_gpt, data_tang)
loss_hope_phase1 = eval_loss(model_hope, data_tang)

log(f'\n第一阶段结束 (耗时 {time.time()-t1:.0f}s)，唐诗 loss:')
log(f'  mini-GPT:       {loss_gpt_phase1:.4f}')
log(f'  Hope-Attention: {loss_hope_phase1:.4f}')

# 保存 checkpoint（后面每个混合比例都从这里加载）
os.makedirs('checkpoints', exist_ok=True)
torch.save({
    'model': model_gpt.state_dict(),
    'optimizer': opt_gpt.state_dict(),
}, 'checkpoints/experiment-gpt-phase1.pt')

torch.save({
    'model': model_hope.state_dict(),
    'optimizer': opt_hope.state_dict(),
}, 'checkpoints/experiment-hope-phase1.pt')

log('Checkpoint 已保存')

# 生成测试
log(f'\n--- 第一阶段生成测试 ---')
for prompt in ['春風', '明月']:
    log(f'  mini-GPT     「{prompt}」→ {generate(model_gpt, prompt)}')
    log(f'  Hope-Attention「{prompt}」→ {generate(model_hope, prompt)}')
    log('')


# ============================================================
# 第二阶段：对每个混合比例，加载 checkpoint → 训练宋词+唐诗混合
# ============================================================

# 存储结果
results = []

for mix_ratio in mix_ratios:
    log(f'\n{"="*60}')
    log(f'第二阶段：训练宋词（混入 {mix_ratio:.0%} 唐诗，{steps_per_phase} 步）')
    log(f'{"="*60}')

    t2 = time.time()

    # 每次从 phase1 checkpoint 重新加载（公平对比）
    ckpt_gpt = torch.load('checkpoints/experiment-gpt-phase1.pt',
                           weights_only=False)
    model_gpt.load_state_dict(ckpt_gpt['model'])
    opt_gpt.load_state_dict(ckpt_gpt['optimizer'])

    ckpt_hope = torch.load('checkpoints/experiment-hope-phase1.pt',
                            weights_only=False)
    model_hope.load_state_dict(ckpt_hope['model'])
    opt_hope.load_state_dict(ckpt_hope['optimizer'])

    log(f'\n--- mini-GPT (混合 {mix_ratio:.0%}) ---')
    train_model_mixed(model_gpt, data_songci, data_tang, mix_ratio,
                      steps_per_phase, opt_gpt, 'mini-GPT')

    log(f'\n--- Hope-Attention (混合 {mix_ratio:.0%}) ---')
    train_model_mixed(model_hope, data_songci, data_tang, mix_ratio,
                      steps_per_phase, opt_hope, 'Hope', is_hope=True,
                      step_offset=steps_per_phase)

    # 测量遗忘
    loss_gpt_tang = eval_loss(model_gpt, data_tang)
    loss_hope_tang = eval_loss(model_hope, data_tang)
    loss_gpt_songci = eval_loss(model_gpt, data_songci)
    loss_hope_songci = eval_loss(model_hope, data_songci)

    forget_gpt = loss_gpt_tang - loss_gpt_phase1
    forget_hope = loss_hope_tang - loss_hope_phase1

    results.append({
        'mix': mix_ratio,
        'gpt_tang': loss_gpt_tang,
        'hope_tang': loss_hope_tang,
        'gpt_songci': loss_gpt_songci,
        'hope_songci': loss_hope_songci,
        'forget_gpt': forget_gpt,
        'forget_hope': forget_hope,
    })

    log(f'\n混合 {mix_ratio:.0%} 结果 (耗时 {time.time()-t2:.0f}s):')
    log(f'  {"":>18s} {"唐诗 loss":>10s}  {"宋词 loss":>10s}  {"遗忘程度":>10s}')
    log(f'  {"mini-GPT":>18s} {loss_gpt_tang:>10.4f}  '
        f'{loss_gpt_songci:>10.4f}  {forget_gpt:>+10.4f}')
    log(f'  {"Hope-Attention":>18s} {loss_hope_tang:>10.4f}  '
        f'{loss_hope_songci:>10.4f}  {forget_hope:>+10.4f}')

    # 生成测试
    log(f'\n--- 生成测试（混合 {mix_ratio:.0%}）---')
    for prompt in ['春風', '明月']:
        log(f'  mini-GPT     「{prompt}」→ {generate(model_gpt, prompt)}')
        log(f'  Hope-Attention「{prompt}」→ {generate(model_hope, prompt)}')
        log('')


# ============================================================
# 最终汇总
# ============================================================

log(f'\n{"="*60}')
log(f'最终汇总')
log(f'{"="*60}')

log(f'\n基线（第一阶段训完唐诗后）:')
log(f'  mini-GPT 唐诗 loss:       {loss_gpt_phase1:.4f}')
log(f'  Hope-Attention 唐诗 loss: {loss_hope_phase1:.4f}')

log(f'\n遗忘对比（数值越小越好）:')
log(f'  {"混合比例":>10s}  {"GPT遗忘":>10s}  {"Hope遗忘":>10s}  {"Hope/GPT":>10s}  {"判定":>6s}')
for r in results:
    ratio = r['forget_hope'] / max(r['forget_gpt'], 0.001)
    verdict = 'Hope胜' if r['forget_hope'] < r['forget_gpt'] else 'GPT胜'
    log(f'  {r["mix"]:>10.0%}  {r["forget_gpt"]:>+10.4f}  '
        f'{r["forget_hope"]:>+10.4f}  {ratio:>10.1%}  {verdict:>6s}')

log(f'\n宋词学习效果（数值越小越好）:')
log(f'  {"混合比例":>10s}  {"GPT宋词":>10s}  {"Hope宋词":>10s}')
for r in results:
    log(f'  {r["mix"]:>10.0%}  {r["gpt_songci"]:>10.4f}  '
        f'{r["hope_songci"]:>10.4f}')

log(f'\n解读:')
log(f'  如果 CMS 有效，预期在低混合比例（5%）时：')
log(f'  Hope-Attention 的遗忘显著少于 mini-GPT（Hope/GPT < 80%）')
log(f'  同时宋词学习效果接近（不牺牲新知识学习速度）')

# 保存报告
os.makedirs('outputs', exist_ok=True)
with open('outputs/forgetting_report.txt', 'w') as f:
    f.write('\n'.join(report))
log(f'\n报告已保存到 outputs/forgetting_report.txt')

log('\n实验完成！')

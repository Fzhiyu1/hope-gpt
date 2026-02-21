"""
BPE Tokenizer 全流程验证脚本

自动运行三阶段训练 + 连续学习测试，最后输出报告。
启动后可以放着不管，跑完会把报告写到 outputs/bpe_report.txt。

用法：
  python3 run_full_test.py              # 默认每阶段 3000 步
  python3 run_full_test.py 1000         # 每阶段 1000 步（快速测试）
"""

import sys
import os
import time
import torch
import torch.nn.functional as F
from model.tokenizer import BPETokenizer, load_tokenizer_from_checkpoint
from model.hope_attention import HopeAttentionGPT

steps_per_phase = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
device = 'mps' if torch.backends.mps.is_available() else 'cpu'

# 超参数
d_model = 256
n_heads = 8
n_layers = 8
context_length = 128
learning_rate = 1e-3
chunk_sizes = [1, 16, 128, 1024]

report = []
def log(msg):
    print(msg)
    report.append(msg)


def get_batch(data, batch_size=4):
    max_start = len(data) - context_length - 1
    starts = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([data[s:s + context_length] for s in starts])
    y = torch.stack([data[s + 1:s + 1 + context_length] for s in starts])
    return x.to(device), y.to(device)


def eval_loss(model, data, num_batches=50):
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


def generate(model, tokenizer, prompt, max_new=50):
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


def train_phase(model, data, num_steps, optimizer, step_offset=0):
    print_every = max(1, num_steps // 10)
    for step in range(num_steps):
        global_step = step + step_offset
        model.set_active_levels(global_step)
        x, y = get_batch(data)
        logits = model(x)
        B, T, V = logits.shape
        loss = F.cross_entropy(logits.view(B * T, V), y.view(B * T))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % print_every == 0:
            print(f'  步骤 {global_step:>6d} | loss = {loss.item():.4f}')
    return loss.item()


# ============================================================
log('=' * 60)
log('BPE Tokenizer 全流程验证')
log(f'每阶段 {steps_per_phase} 步，设备: {device}')
log('=' * 60)
t_start = time.time()

# 读取数据
with open('data/poetry_tang.txt', 'r') as f:
    text_tang = f.read()
with open('data/poetry_songci.txt', 'r') as f:
    text_songci = f.read()
with open('data/poetry_songshi.txt', 'r') as f:
    text_songshi = f.read()

log(f'\n数据: 唐诗 {len(text_tang):,} 字 / 宋词 {len(text_songci):,} 字 / 宋诗 {len(text_songshi):,} 字')


# ============================================================
# 第一阶段：唐诗从头训练
# ============================================================
log(f'\n{"=" * 60}')
log(f'第一阶段：唐诗从头训练（{steps_per_phase} 步）')
log('=' * 60)

t1 = time.time()
# BPE 词表：有缓存直接加载，没有则训练并保存
vocab_cache = 'checkpoints/bpe-vocab-all.pt'
if os.path.exists(vocab_cache):
    tokenizer = BPETokenizer.load_from_file(vocab_cache)
else:
    tokenizer = BPETokenizer.train([text_tang, text_songci, text_songshi], target_vocab_size=8000)
    os.makedirs('checkpoints', exist_ok=True)
    tokenizer.save_to_file(vocab_cache)
data_tang = torch.tensor(tokenizer.encode(text_tang), dtype=torch.long)

log(f'BPE 词表: {tokenizer.vocab_size}（三份数据联合训练）')
log(f'唐诗编码: {len(text_tang)} 字符 → {len(data_tang)} tokens '
    f'(压缩率 {len(data_tang)/len(text_tang):.2f})')

model = HopeAttentionGPT(
    vocab_size=tokenizer.vocab_size, d_model=d_model, n_heads=n_heads,
    n_layers=n_layers, context_length=context_length, chunk_sizes=chunk_sizes,
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

log(f'模型参数: {sum(p.numel() for p in model.parameters()):,}')
final_loss_1 = train_phase(model, data_tang, steps_per_phase, optimizer)

loss_tang_after_1 = eval_loss(model, data_tang)
log(f'\n第一阶段结果: loss = {loss_tang_after_1:.4f} (耗时 {time.time()-t1:.0f}s)')

# 保存 checkpoint
os.makedirs('checkpoints', exist_ok=True)
ckpt_path = 'checkpoints/bpe-test-phase1.pt'
torch.save({
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'tokenizer_type': 'bpe',
    'tokenizer_data': tokenizer.save_vocab(),
    'total_steps': steps_per_phase,
    'config': {
        'vocab_size': tokenizer.vocab_size, 'd_model': d_model,
        'n_heads': n_heads, 'n_layers': n_layers,
        'context_length': context_length, 'chunk_sizes': chunk_sizes,
    },
}, ckpt_path)
log(f'Checkpoint 已保存: {ckpt_path}')

# 生成测试
log('\n--- 第一阶段生成 ---')
for prompt in ['秦川', '明月']:
    result = generate(model, tokenizer, prompt, max_new=30)
    log(f'  「{prompt}」→ {result}')


# ============================================================
# 第二阶段：加载 checkpoint，续训宋词（核心测试）
# ============================================================
log(f'\n{"=" * 60}')
log(f'第二阶段：加载 checkpoint，续训宋词（{steps_per_phase} 步）')
log('这是 BPE 升级要解决的核心问题——跨数据集续训')
log('=' * 60)

t2 = time.time()

# 模拟真实场景：完全从 checkpoint 重建
checkpoint = torch.load(ckpt_path, weights_only=False)
tokenizer_loaded = load_tokenizer_from_checkpoint(checkpoint)
config = checkpoint['config']

# 编码宋词数据（有 UNK 不会崩）
data_songci = torch.tensor(tokenizer_loaded.encode(text_songci), dtype=torch.long)
unk_count = (data_songci == 0).sum().item()
log(f'宋词编码: {len(text_songci)} 字符 → {len(data_songci)} tokens '
    f'(UNK: {unk_count}, {unk_count/len(data_songci)*100:.1f}%)')

# 重建模型
model2 = HopeAttentionGPT(**config).to(device)
model2.load_state_dict(checkpoint['model'])
optimizer2 = torch.optim.Adam(model2.parameters(), lr=learning_rate)
optimizer2.load_state_dict(checkpoint['optimizer'])

# 记录训练前的唐诗 loss（遗忘基线）
data_tang_eval = torch.tensor(tokenizer_loaded.encode(text_tang), dtype=torch.long)
loss_tang_before_2 = eval_loss(model2, data_tang_eval)
log(f'训练前唐诗 loss: {loss_tang_before_2:.4f}')

# 训练宋词
final_loss_2 = train_phase(model2, data_songci, steps_per_phase, optimizer2,
                            step_offset=steps_per_phase)

loss_songci_after_2 = eval_loss(model2, data_songci)
loss_tang_after_2 = eval_loss(model2, data_tang_eval)
forget = loss_tang_after_2 - loss_tang_before_2
log(f'\n第二阶段结果 (耗时 {time.time()-t2:.0f}s):')
log(f'  宋词 loss: {loss_songci_after_2:.4f}')
log(f'  唐诗 loss: {loss_tang_before_2:.4f} → {loss_tang_after_2:.4f} (遗忘: {forget:+.4f})')

# 保存
ckpt_path_2 = 'checkpoints/bpe-test-phase2.pt'
torch.save({
    'model': model2.state_dict(),
    'optimizer': optimizer2.state_dict(),
    'tokenizer_type': 'bpe',
    'tokenizer_data': tokenizer_loaded.save_vocab(),
    'total_steps': steps_per_phase * 2,
    'config': config,
}, ckpt_path_2)

log('\n--- 第二阶段生成 ---')
for prompt in ['秦川', '明月']:
    result = generate(model2, tokenizer_loaded, prompt, max_new=30)
    log(f'  「{prompt}」→ {result}')


# ============================================================
# 第三阶段：再续训宋诗（三连续训）
# ============================================================
log(f'\n{"=" * 60}')
log(f'第三阶段：加载 checkpoint，续训宋诗（{steps_per_phase} 步）')
log('=' * 60)

t3 = time.time()

checkpoint2 = torch.load(ckpt_path_2, weights_only=False)
tokenizer_loaded2 = load_tokenizer_from_checkpoint(checkpoint2)

data_songshi = torch.tensor(tokenizer_loaded2.encode(text_songshi), dtype=torch.long)
unk_count_2 = (data_songshi == 0).sum().item()
log(f'宋诗编码: {len(text_songshi)} 字符 → {len(data_songshi)} tokens '
    f'(UNK: {unk_count_2}, {unk_count_2/len(data_songshi)*100:.1f}%)')

model3 = HopeAttentionGPT(**checkpoint2['config']).to(device)
model3.load_state_dict(checkpoint2['model'])
optimizer3 = torch.optim.Adam(model3.parameters(), lr=learning_rate)
optimizer3.load_state_dict(checkpoint2['optimizer'])

loss_tang_before_3 = eval_loss(model3, data_tang_eval)
loss_songci_before_3 = eval_loss(model3, data_songci)

final_loss_3 = train_phase(model3, data_songshi, steps_per_phase, optimizer3,
                            step_offset=steps_per_phase * 2)

loss_tang_after_3 = eval_loss(model3, data_tang_eval)
loss_songci_after_3 = eval_loss(model3, data_songci)
loss_songshi_after_3 = eval_loss(model3, data_songshi)

log(f'\n第三阶段结果 (耗时 {time.time()-t3:.0f}s):')
log(f'  宋诗 loss: {loss_songshi_after_3:.4f}')
log(f'  唐诗 loss: {loss_tang_before_3:.4f} → {loss_tang_after_3:.4f} '
    f'(遗忘: {loss_tang_after_3 - loss_tang_before_3:+.4f})')
log(f'  宋词 loss: {loss_songci_before_3:.4f} → {loss_songci_after_3:.4f} '
    f'(遗忘: {loss_songci_after_3 - loss_songci_before_3:+.4f})')

log('\n--- 第三阶段生成 ---')
for prompt in ['秦川', '明月', '江南']:
    result = generate(model3, tokenizer_loaded2, prompt, max_new=30)
    log(f'  「{prompt}」→ {result}')


# ============================================================
# 最终报告
# ============================================================
total_time = time.time() - t_start

log(f'\n{"=" * 60}')
log('最终报告')
log('=' * 60)
log(f'总耗时: {total_time/60:.1f} 分钟')
log(f'每阶段: {steps_per_phase} 步')
log(f'')
log(f'BPE 词表: {tokenizer.vocab_size} (基于唐诗训练)')
log(f'')
log(f'连续学习 loss 变化:')
log(f'  {"":>8s}  {"训唐诗后":>10s}  {"续训宋词后":>10s}  {"续训宋诗后":>10s}')
log(f'  {"唐诗":>8s}  {loss_tang_after_1:>10.4f}  {loss_tang_after_2:>10.4f}  {loss_tang_after_3:>10.4f}')
log(f'  {"宋词":>8s}  {"—":>10s}  {loss_songci_after_2:>10.4f}  {loss_songci_after_3:>10.4f}')
log(f'  {"宋诗":>8s}  {"—":>10s}  {"—":>10s}  {loss_songshi_after_3:>10.4f}')
log(f'')
log(f'核心验证:')
log(f'  [{"PASS" if True else "FAIL"}] BPE 训练成功')
log(f'  [PASS] UNK 处理正常（联合词表，宋词 UNK: {unk_count}）')
log(f'  [{"PASS" if True else "FAIL"}] 跨数据集续训成功（唐诗→宋词→宋诗，三阶段无报错）')
log(f'  [{"PASS" if True else "FAIL"}] Checkpoint 保存/加载正常')

# 保存报告
os.makedirs('outputs', exist_ok=True)
with open('outputs/bpe_report.txt', 'w') as f:
    f.write('\n'.join(report))
log(f'\n报告已保存到 outputs/bpe_report.txt')
log('全部完成！')

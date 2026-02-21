"""
和训练好的 Mini-GPT 对话（其实是让它续写文字）

用法：
  python3 chat.py                          # 加载默认模型
  python3 chat.py checkpoints/mini-gpt-50steps.pt   # 加载指定模型
"""

import sys
import torch
import torch.nn.functional as F
from model.tokenizer import load_tokenizer_from_checkpoint
from model.gpt import MiniGPT
from model.hope_attention import HopeAttentionGPT
from model.hope import HopeGPT

# 加载模型
model_path = sys.argv[1] if len(sys.argv) > 1 else None
if model_path is None:
    import glob
    files = sorted(glob.glob('checkpoints/mini-gpt-*.pt'))
    if not files:
        print('没有找到模型文件，请先运行 python3 train.py 训练')
        sys.exit(1)
    model_path = files[-1]  # 取最新的

print(f'加载模型: {model_path}')
checkpoint = torch.load(model_path, weights_only=False)
config = checkpoint['config']

# 重建 tokenizer（自动识别 BPE 或 Char）
tokenizer = load_tokenizer_from_checkpoint(checkpoint)

# 重建模型并加载参数（自动识别模型类型）
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'
model_type = checkpoint.get('model_type', '')
if model_type == 'hope-full':
    model = HopeGPT(**config).to(device)
elif 'chunk_sizes' in config:
    model = HopeAttentionGPT(**config).to(device)
else:
    model = MiniGPT(**config).to(device)
model.load_state_dict(checkpoint['model'])
model.eval()

print(f'模型已加载（{sum(p.numel() for p in model.parameters()):,} 参数）')
print(f'词表大小: {tokenizer.vocab_size}')
print(f'输入开头文字，模型会续写。输入 q 退出。\n')


def generate(prompt, max_new_tokens=50):
    tokens = tokenizer.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            input_tokens = tokens[:, -config['context_length']:]
            logits = model(input_tokens)
            next_logits = logits[:, -1, :]
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_token], dim=1)

    return tokenizer.decode(tokens[0].tolist())


while True:
    prompt = input('你: ')
    if prompt == 'q':
        break
    result = generate(prompt)
    print(f'模型: {result}\n')

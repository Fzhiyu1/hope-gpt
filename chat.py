"""
和训练好的 Mini-GPT 对话（其实是让它续写文字）

用法：
  python3 chat.py                          # 加载默认模型
  python3 chat.py checkpoints/mini-gpt-50steps.pt   # 加载指定模型
"""

import sys
import torch
import torch.nn.functional as F
from model.tokenizer import CharTokenizer
from model.gpt import MiniGPT

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

# 重建 tokenizer
tokenizer = CharTokenizer(checkpoint['tokenizer_text'])

# 重建模型并加载参数
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
model = MiniGPT(**config).to(device)
model.load_state_dict(checkpoint['model'])
model.eval()

print(f'模型已加载（{sum(p.numel() for p in model.parameters()):,} 参数）')
print(f'词表: {"".join(tokenizer.idx_to_char.values())}')
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
    try:
        result = generate(prompt)
        print(f'模型: {result}\n')
    except KeyError as e:
        print(f'模型不认识这个字: {e}\n')

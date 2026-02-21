"""
下载中文维基百科数据并转为训练用 txt 文件

用法：
  pip install datasets
  python3 download_wiki.py
"""

from datasets import load_dataset
import os

print('正在下载中文维基百科（约 524MB）...')
ds = load_dataset("pleisto/wikipedia-cn-20230720-filtered", split="train")

os.makedirs('data', exist_ok=True)
output_path = 'data/wiki_cn.txt'

with open(output_path, 'w', encoding='utf-8') as f:
    for i, item in enumerate(ds):
        f.write(item["text"].strip() + '\n\n')
        if (i + 1) % 50000 == 0:
            print(f'  已处理 {i + 1}/{len(ds)} 篇文章...')

file_size = os.path.getsize(output_path) / 1024 / 1024
print(f'\n完成! 共 {len(ds)} 篇文章')
print(f'保存到: {output_path}（{file_size:.1f} MB）')

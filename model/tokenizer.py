"""
字符级 Tokenizer（分词器）

机器不认识文字，只认识数字。
Tokenizer 的工作就是：文字 ↔ 数字，两个方向都能转换。

这是最简单的一种 Tokenizer：每个字符对应一个数字。
真实的大模型（GPT、Claude）用的是更复杂的 BPE 分词，
但原理是一样的：把文本切成小块，每块对应一个数字。
"""


class CharTokenizer:
    """字符级分词器：一个字符 = 一个 token"""

    def __init__(self, text):
        # 第一步：找出文本中所有不重复的字符
        chars = sorted(set(text))

        # vocab_size = 词表大小，也就是模型需要认识多少个"字"
        self.vocab_size = len(chars)

        # 第二步：建立映射表
        # char_to_idx: 字符 → 数字  （编码用）
        # idx_to_char: 数字 → 字符  （解码用）
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}

    def encode(self, text):
        """编码：把文字变成数字列表"""
        return [self.char_to_idx[ch] for ch in text]

    def decode(self, indices):
        """解码：把数字列表变回文字"""
        return ''.join(self.idx_to_char[i] for i in indices)


# ===== 运行看看效果 =====
if __name__ == '__main__':
    # 读取训练数据
    with open('data/sample.txt', 'r') as f:
        text = f.read()

    # 创建 tokenizer
    tokenizer = CharTokenizer(text)

    print(f'文本总长度: {len(text)} 个字符')
    print(f'词表大小: {tokenizer.vocab_size} 个不同的字符')
    print(f'词表内容: {"".join(tokenizer.idx_to_char.values())}')
    print()

    # 演示：编码
    sample = '天地玄黄'
    encoded = tokenizer.encode(sample)
    print(f'编码: "{sample}" → {encoded}')

    # 演示：解码
    decoded = tokenizer.decode(encoded)
    print(f'解码: {encoded} → "{decoded}"')

    # 验证：编码再解码，应该得到原文
    assert decoded == sample, '编码解码不一致！'
    print(f'验证通过：编码→解码 = 原文')

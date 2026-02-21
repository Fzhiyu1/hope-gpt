"""
Tokenizer（分词器）

机器不认识文字，只认识数字。
Tokenizer 的工作就是：文字 ↔ 数字，两个方向都能转换。

包含两种分词器：
- CharTokenizer：字符级，一个字 = 一个 token（向后兼容）
- BPETokenizer：BPE 分词，GPT-2/3/4 使用的方式，从字符出发反复合并常见对
"""

from collections import Counter


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


class BPETokenizer:
    """
    BPE（Byte Pair Encoding）分词器

    GPT-2/3/4 真正使用的分词方式。
    从单字符出发，反复合并最常见的相邻 token 对，
    让常见词组（如"江南"、"相思"）成为独立 token，提升编码效率。

    特点：
    - 有 <UNK>（ID=0），遇到未知字符不会崩溃
    - 支持多文本训练，天然解决词表兼容问题
    """

    UNK_TOKEN = '<UNK>'
    UNK_ID = 0

    def __init__(self):
        # token → ID
        self.token_to_id = {self.UNK_TOKEN: self.UNK_ID}
        # ID → token
        self.id_to_token = {self.UNK_ID: self.UNK_TOKEN}
        # 合并规则列表，按训练时的顺序排列（顺序不可变）
        # 每条规则是 (token_a, token_b)，表示 a+b → 新token
        self.merges = []
        self.vocab_size = 1  # 初始只有 <UNK>

    @classmethod
    def train(cls, texts, target_vocab_size=8000):
        """
        从多个文本训练 BPE 分词器。

        算法：
        1. 初始化：<UNK>(ID=0) + 所有出现过的字符(ID=1~N)
        2. 把文本按行分割，每行转成字符级 token 列表
        3. 循环：统计相邻 token 对频率 → 合并最频繁的对 → 新 token 加入词表
        4. 重复直到达到目标 vocab_size 或没有频率≥2的对

        Args:
            texts: 文本列表（多个文件的内容）
            target_vocab_size: 目标词表大小，默认 8000

        Returns:
            训练好的 BPETokenizer 实例
        """
        tokenizer = cls()

        # 第一步：收集所有字符，建立初始词表
        all_chars = set()
        for text in texts:
            all_chars.update(text)
        for ch in sorted(all_chars):
            token_id = tokenizer.vocab_size
            tokenizer.token_to_id[ch] = token_id
            tokenizer.id_to_token[token_id] = ch
            tokenizer.vocab_size += 1

        base_vocab_size = tokenizer.vocab_size

        # 根据数据量自适应调整目标词表大小
        # 小数据集不强行凑到 8000，避免合并只出现 2 次的无意义长串
        total_chars = sum(len(t) for t in texts)
        max_useful_merges = total_chars // 50  # 经验值：每 50 字符支撑 1 次有意义的合并
        adaptive_target = min(target_vocab_size,
                              base_vocab_size + max_useful_merges)
        if adaptive_target < target_vocab_size:
            print(f'BPE 训练：基础字符 {base_vocab_size - 1} 个（+<UNK>），'
                  f'数据量 {total_chars:,} 字符，'
                  f'自适应词表 {adaptive_target}（原目标 {target_vocab_size}）')
        else:
            print(f'BPE 训练：基础字符 {base_vocab_size - 1} 个（+<UNK>），'
                  f'目标词表 {target_vocab_size}')
        target_vocab_size = adaptive_target

        # 第二步：把所有文本按行分割，转成 token ID 列表
        # 不跨行合并，每行独立处理
        lines = []
        for text in texts:
            for line in text.split('\n'):
                if line:  # 跳过空行
                    token_ids = [tokenizer.token_to_id[ch] for ch in line]
                    lines.append(token_ids)

        # 第三步：反复合并最常见的相邻 token 对
        num_merges = target_vocab_size - tokenizer.vocab_size
        for i in range(num_merges):
            # 统计所有行中相邻 token 对的频率
            pair_counts = Counter()
            for line in lines:
                for j in range(len(line) - 1):
                    pair = (line[j], line[j + 1])
                    pair_counts[pair] += 1

            if not pair_counts:
                break

            # 找出最频繁的对
            best_pair, best_count = pair_counts.most_common(1)[0]
            if best_count < 2:
                break  # 没有频率≥2的对了

            # 创建新 token
            token_a = tokenizer.id_to_token[best_pair[0]]
            token_b = tokenizer.id_to_token[best_pair[1]]
            new_token = token_a + token_b
            new_id = tokenizer.vocab_size

            tokenizer.token_to_id[new_token] = new_id
            tokenizer.id_to_token[new_id] = new_token
            tokenizer.merges.append(best_pair)
            tokenizer.vocab_size += 1

            # 在所有行中执行合并
            for idx, line in enumerate(lines):
                new_line = []
                j = 0
                while j < len(line):
                    if j < len(line) - 1 and line[j] == best_pair[0] and line[j + 1] == best_pair[1]:
                        new_line.append(new_id)
                        j += 2
                    else:
                        new_line.append(line[j])
                        j += 1
                lines[idx] = new_line

            if (i + 1) % 500 == 0:
                print(f'  合并 {i + 1}/{num_merges}：'
                      f'"{new_token}" (出现{best_count}次)')

        actual_merges = tokenizer.vocab_size - base_vocab_size
        print(f'BPE 训练完成：{actual_merges} 次合并，'
              f'最终词表 {tokenizer.vocab_size}')

        return tokenizer

    def encode(self, text):
        """
        编码：把文字变成 token ID 列表。

        按训练时的合并顺序逐条应用规则。
        按行分割，不跨行合并。
        未知字符返回 <UNK>（ID=0）。
        """
        result = []
        for line in text.split('\n'):
            if result:  # 行之间加换行符的 token
                newline_id = self.token_to_id.get('\n', self.UNK_ID)
                result.append(newline_id)
            if not line:
                continue
            # 先转成字符级 token
            token_ids = [self.token_to_id.get(ch, self.UNK_ID) for ch in line]
            # 按合并顺序逐条应用规则
            for a, b in self.merges:
                merged_token = self.id_to_token[a] + self.id_to_token[b]
                new_id = self.token_to_id[merged_token]
                new_ids = []
                j = 0
                while j < len(token_ids):
                    if j < len(token_ids) - 1 and token_ids[j] == a and token_ids[j + 1] == b:
                        new_ids.append(new_id)
                        j += 2
                    else:
                        new_ids.append(token_ids[j])
                        j += 1
                token_ids = new_ids
            result.extend(token_ids)
        return result

    def decode(self, indices):
        """解码：把 token ID 列表变回文字"""
        return ''.join(self.id_to_token.get(i, self.UNK_TOKEN) for i in indices)

    def save_vocab(self):
        """序列化为可保存的字典"""
        return {
            'merges': self.merges,
            'vocab': self.token_to_id,
            'vocab_size': self.vocab_size,
        }

    @classmethod
    def load_vocab(cls, data):
        """从字典重建 BPETokenizer"""
        tokenizer = cls()
        tokenizer.token_to_id = data['vocab']
        tokenizer.id_to_token = {v: k for k, v in data['vocab'].items()}
        tokenizer.merges = [tuple(m) for m in data['merges']]
        tokenizer.vocab_size = data['vocab_size']
        return tokenizer

    def save_to_file(self, path):
        """保存词表到文件"""
        import torch
        torch.save(self.save_vocab(), path)
        print(f'词表已保存到 {path}')

    @classmethod
    def load_from_file(cls, path):
        """从文件加载词表"""
        import torch
        data = torch.load(path, weights_only=False)
        tokenizer = cls.load_vocab(data)
        print(f'词表已加载: {path}（{tokenizer.vocab_size} tokens）')
        return tokenizer


class HFBPETokenizer:
    """
    BPE 分词器 — HuggingFace tokenizers 引擎（Rust 实现，快 100 倍）

    接口与 BPETokenizer 完全一致，可无缝替换。
    需要安装：pip install tokenizers
    """

    UNK_TOKEN = '<UNK>'

    def __init__(self, hf_tokenizer):
        self._tok = hf_tokenizer
        self.vocab_size = hf_tokenizer.get_vocab_size()

    @classmethod
    def train(cls, texts, target_vocab_size=8000):
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import Sequence, UnicodeScripts, Digits

        tok = Tokenizer(BPE(unk_token=cls.UNK_TOKEN))
        tok.pre_tokenizer = Sequence([
            UnicodeScripts(),
            Digits(individual_digits=True),
        ])

        trainer = BpeTrainer(
            vocab_size=target_vocab_size,
            special_tokens=[cls.UNK_TOKEN],
            min_frequency=2,
        )

        tok.train_from_iterator(texts, trainer=trainer)

        print(f'HF BPE 训练完成：词表 {tok.get_vocab_size()}')
        return cls(tok)

    def encode(self, text):
        return self._tok.encode(text).ids

    def decode(self, indices):
        return self._tok.decode(indices)

    def save_vocab(self):
        return {
            'tokenizer_json': self._tok.to_str(),
            'vocab_size': self.vocab_size,
        }

    @classmethod
    def load_vocab(cls, data):
        from tokenizers import Tokenizer
        tok = Tokenizer.from_str(data['tokenizer_json'])
        return cls(tok)


def load_tokenizer_from_checkpoint(checkpoint):
    """
    根据 checkpoint 内容自动选择正确的 tokenizer。

    - tokenizer_type='hf_bpe' → HFBPETokenizer
    - tokenizer_type='bpe' → BPETokenizer
    - 旧格式（有 tokenizer_text）→ CharTokenizer
    """
    tok_type = checkpoint.get('tokenizer_type')
    if tok_type == 'hf_bpe':
        return HFBPETokenizer.load_vocab(checkpoint['tokenizer_data'])
    elif tok_type == 'bpe':
        return BPETokenizer.load_vocab(checkpoint['tokenizer_data'])
    else:
        return CharTokenizer(checkpoint['tokenizer_text'])


# ===== 运行看看效果 =====
if __name__ == '__main__':
    import os

    # 测试 CharTokenizer（向后兼容）
    print('===== CharTokenizer 测试 =====\n')
    with open('data/sample.txt', 'r') as f:
        text = f.read()

    char_tok = CharTokenizer(text)
    print(f'文本总长度: {len(text)} 个字符')
    print(f'词表大小: {char_tok.vocab_size} 个不同的字符')

    sample = '天地玄黄'
    encoded = char_tok.encode(sample)
    decoded = char_tok.decode(encoded)
    print(f'编码: "{sample}" → {encoded}')
    print(f'解码: {encoded} → "{decoded}"')
    assert decoded == sample, '编码解码不一致！'
    print(f'验证通过')

    # 测试 BPETokenizer
    print(f'\n===== BPETokenizer 测试 =====\n')

    # 用现有数据文件训练
    train_texts = []
    for fname in ['data/sample.txt']:
        if os.path.exists(fname):
            with open(fname, 'r') as f:
                train_texts.append(f.read())

    bpe_tok = BPETokenizer.train(train_texts, target_vocab_size=2000)
    print(f'\n词表大小: {bpe_tok.vocab_size}')

    # 测试编码解码
    for sample in ['天地玄黄', '宇宙洪荒', '这个字不存在吧xyz']:
        encoded = bpe_tok.encode(sample)
        decoded = bpe_tok.decode(encoded)
        print(f'编码: "{sample}" → {encoded[:10]}... ({len(encoded)} tokens)')
        print(f'解码: → "{decoded}"')
        print()

    # 测试 save/load
    data = bpe_tok.save_vocab()
    bpe_tok2 = BPETokenizer.load_vocab(data)
    test_text = '天地玄黄宇宙洪荒'
    assert bpe_tok.encode(test_text) == bpe_tok2.encode(test_text)
    print('save/load 验证通过')

    # 测试 load_tokenizer_from_checkpoint
    # 模拟新格式 checkpoint
    fake_ckpt = {'tokenizer_type': 'bpe', 'tokenizer_data': data}
    tok_new = load_tokenizer_from_checkpoint(fake_ckpt)
    assert tok_new.encode(test_text) == bpe_tok.encode(test_text)

    # 模拟旧格式 checkpoint
    fake_ckpt_old = {'tokenizer_text': text}
    tok_old = load_tokenizer_from_checkpoint(fake_ckpt_old)
    assert isinstance(tok_old, CharTokenizer)

    print('load_tokenizer_from_checkpoint 验证通过')

"""
Mini-GPT 模型

一个最小的 GPT 实现，用于学习 Transformer 的工作原理。
每个组件都有详细注释，方便理解。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# 组件 1：Embedding（嵌入层）
# ============================================================
# 问题：Tokenizer 给了我们数字（比如 67 代表"天"），
#       但一个数字太单薄了，模型需要更丰富的表示。
#
# 解决：把每个数字变成一个向量（一串数字）。
#       比如 67 → [0.12, -0.34, 0.56, ...]（64个数字）
#       这样模型就有更多维度来理解每个字的含义。
#
# 类比：一个字的编号就像一个人的身份证号——只是个编号。
#       而向量就像这个人的简历——包含了丰富的信息。
#       训练过程中，模型会自动学会：意思相近的字，向量也相近。


# ============================================================
# 组件 2：位置编码（Positional Encoding）
# ============================================================
# 问题："我吃饭"和"饭吃我"用了完全相同的字，
#       但意思截然不同。模型怎么知道字的顺序？
#
# 解决：给每个位置也分配一个向量，加到字的向量上。
#       位置0的向量 + "我"的向量 → 模型知道"我"在第一个位置
#       位置1的向量 + "吃"的向量 → 模型知道"吃"在第二个位置


# ============================================================
# 组件 3：Self-Attention（自注意力）
# ============================================================
# 这是 Transformer 的核心。
#
# 问题：理解一个字，需要看它周围的字。
#       "苹果很好吃"的"苹果"是水果，
#       "苹果发布新手机"的"苹果"是公司。
#       同一个字，含义取决于上下文。
#
# Attention 的做法：每个字向所有其他字"提问"，
#       收集相关信息，更新自己的理解。
#
# Q（Query，查询）：我在找什么信息？
# K（Key，键）：    我有什么信息可以提供？
# V（Value，值）：  我的具体信息内容是什么？
#
# 计算过程：
# 1. Q 和 K 做点积 → 得到"相关性分数"
# 2. 分数做 softmax → 变成权重（加起来=1）
# 3. 用权重对 V 加权求和 → 得到融合了上下文的新表示


class SelfAttention(nn.Module):
    """单头自注意力"""

    def __init__(self, d_model, head_dim):
        super().__init__()
        # 三个投影矩阵：把输入变成 Q、K、V
        self.query = nn.Linear(d_model, head_dim, bias=False)
        self.key = nn.Linear(d_model, head_dim, bias=False)
        self.value = nn.Linear(d_model, head_dim, bias=False)
        self.head_dim = head_dim

    def forward(self, x, mask):
        Q = self.query(x)   # 每个字的"提问"
        K = self.key(x)     # 每个字的"标签"
        V = self.value(x)   # 每个字的"内容"

        # Q 和 K 做点积，除以 sqrt(head_dim) 防止数值太大
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 应用 mask：GPT 不能看到未来的字（只能看前面的）
        scores = scores.masked_fill(mask == 0, float('-inf'))

        # softmax：把分数变成权重
        weights = F.softmax(scores, dim=-1)

        # 用权重对 V 加权求和
        out = weights @ V
        return out


# ============================================================
# 组件 4：Multi-Head Attention（多头注意力）
# ============================================================
# 为什么要"多头"？
#
# 一个 Attention 只能关注一种关系。
# 多头 = 多个 Attention 并行运行，各自关注不同的关系：
#   - 头1 可能关注语法关系（主语和动词）
#   - 头2 可能关注语义关系（近义词）
#   - 头3 可能关注位置关系（相邻的字）
#
# 最后把所有头的结果拼起来。


class MultiHeadAttention(nn.Module):
    """多头注意力：多个 SelfAttention 并行"""

    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        head_dim = d_model // n_heads

        # 创建多个注意力头
        self.heads = nn.ModuleList([
            SelfAttention(d_model, head_dim) for _ in range(n_heads)
        ])
        # 把多头的输出拼接后，投影回原来的维度
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask):
        # 每个头各自计算
        head_outputs = [head(x, mask) for head in self.heads]
        # 拼接所有头的输出
        concat = torch.cat(head_outputs, dim=-1)
        # 投影回 d_model 维
        return self.proj(concat)


# ============================================================
# 组件 5：Feed-Forward Network（前馈网络）
# ============================================================
# Attention 让每个字看到了上下文，但还没做"思考"。
# FFN 就是对每个字独立地做一次非线性变换——可以理解为"消化信息"。
#
# 结构很简单：放大 → 激活 → 缩回
# d_model → 4*d_model → d_model
#
# 你在论文里看到的"前馈层"就是这个东西。


class FeedForward(nn.Module):
    """前馈网络：两层线性变换 + 激活函数"""

    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),   # 放大 4 倍
            nn.GELU(),                          # 激活函数（引入非线性）
            nn.Linear(4 * d_model, d_model),   # 缩回原大小
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# 组件 6：Transformer Block（变换器块）
# ============================================================
# 把 Attention 和 FFN 组合起来，加上两个关键技巧：
#
# 1. Layer Norm（层归一化）：稳定训练，防止数值爆炸
# 2. Residual Connection（残差连接）：
#    output = input + 变换(input)
#    就是"在原来的基础上做修改"，而不是完全替换。
#    这让深层网络更容易训练。


class TransformerBlock(nn.Module):
    """一个 Transformer 块 = Attention + FFN + LayerNorm + 残差"""

    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)         # Attention 前的归一化
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)         # FFN 前的归一化
        self.ffn = FeedForward(d_model)

    def forward(self, x, mask):
        # Attention + 残差连接
        x = x + self.attn(self.ln1(x), mask)
        # FFN + 残差连接
        x = x + self.ffn(self.ln2(x))
        return x


# ============================================================
# 组件 7：完整的 GPT 模型
# ============================================================
# 把上面所有组件组装起来：
#
# 输入(数字) → 字嵌入 + 位置嵌入 → N个Transformer Block → 输出(预测下一个字)


class MiniGPT(nn.Module):
    """完整的 Mini-GPT 模型"""

    def __init__(self, vocab_size, d_model=64, n_heads=4, n_layers=4, context_length=128):
        super().__init__()

        # 字嵌入：数字 → 向量
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        # 位置嵌入：位置 → 向量
        self.position_embedding = nn.Embedding(context_length, d_model)

        # N 个 Transformer Block 堆叠
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads) for _ in range(n_layers)
        ])

        # 最后的归一化
        self.ln_final = nn.LayerNorm(d_model)
        # 输出层：向量 → 词表大小的分数（预测每个字的概率）
        self.output_proj = nn.Linear(d_model, vocab_size)

        self.context_length = context_length

    def forward(self, idx):
        B, T = idx.shape  # B=批次大小, T=序列长度

        # 第一步：嵌入
        tok_emb = self.token_embedding(idx)           # (B, T, d_model)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))  # (T, d_model)
        x = tok_emb + pos_emb                         # 字向量 + 位置向量

        # 创建 causal mask：每个位置只能看到自己和前面的位置
        mask = torch.tril(torch.ones(T, T, device=idx.device))  # 下三角矩阵

        # 第二步：通过所有 Transformer Block
        for block in self.blocks:
            x = block(x, mask)

        # 第三步：输出预测
        x = self.ln_final(x)
        logits = self.output_proj(x)  # (B, T, vocab_size) — 每个位置预测下一个字

        return logits


# ===== 测试模型能不能跑 =====
if __name__ == '__main__':
    # 模型参数
    vocab_size = 258     # 词表大小（来自 tokenizer）
    d_model = 64         # 每个字的向量维度
    n_heads = 4          # 注意力头数
    n_layers = 4         # Transformer 块数
    context_length = 128 # 上下文窗口大小

    # 创建模型
    model = MiniGPT(vocab_size, d_model, n_heads, n_layers, context_length)

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f'模型参数量: {total_params:,}')

    # 模拟输入：一个长度为 16 的随机序列
    fake_input = torch.randint(0, vocab_size, (1, 16))  # (batch=1, length=16)
    print(f'输入形状: {fake_input.shape}')

    # 前向传播
    output = model(fake_input)
    print(f'输出形状: {output.shape}')
    print(f'含义: 对 16 个位置，每个位置给出 {vocab_size} 个字的概率分数')

    print('\n模型结构:')
    print(model)

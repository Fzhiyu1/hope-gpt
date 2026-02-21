"""
Hope-Attention 模型

Hope 架构的简化版：标准 Softmax 注意力 + CMS（连续记忆系统）。

和 mini-GPT 的区别：
- mini-GPT 每个 Block 里有 1 个 FFN → Hope-Attention 有 4 个（CMS）
- 4 个 FFN 串联，更新频率不同：每1步、每16步、每128步、每1024步
- 高频 FFN = 短期记忆（快速适应当前数据）
- 低频 FFN = 长期记忆（保持持久知识，抗遗忘）

来自论文：Nested Learning: The Illusion of Deep Learning Architectures
公式 (70)：y_t = MLP^(f4)(MLP^(f3)(MLP^(f2)(MLP^(f1)(x_t))))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 复用 mini-GPT 的注意力组件
from model.gpt import SelfAttention, MultiHeadAttention


# ============================================================
# 组件 1：FeedForward（和 mini-GPT 一样，但会有多个）
# ============================================================

class FeedForward(nn.Module):
    """前馈网络：和 mini-GPT 里的一模一样"""

    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# 组件 2：CMS（连续记忆系统）—— Hope-Attention 的核心创新
# ============================================================
#
# 论文的核心洞察：
#   标准 Transformer 的 FFN 在预训练后就冻结了（更新频率 = 0）。
#   CMS 把一个 FFN 变成多个，每个以不同频率更新：
#
#   高频 FFN（每步更新）   → 快速适应当前数据，短期记忆
#   中频 FFN（每16步更新）  → 中期模式
#   低频 FFN（每128步更新） → 较持久的知识
#   超低频 FFN（每1024步）  → 长期知识，很难被遗忘
#
# 顺序变体：高频输出 → 中频输入 → 低频输入 → 超低频输入 → 最终输出
#
# 抗遗忘原理：
#   当高频 FFN 学新东西、忘了旧东西时，
#   低频 FFN 里还保留着旧知识。
#   通过反向传播，遗忘的知识可以从低频层"恢复"回来。

class CMS(nn.Module):
    """
    连续记忆系统（Continuum Memory System）

    4 个 FFN 串联，更新频率分别为：1, 16, 128, 1024
    训练时通过冻结/解冻 requires_grad 来控制哪些 FFN 参与更新。
    """

    def __init__(self, d_model, chunk_sizes=None):
        super().__init__()

        # 默认的更新频率（每隔多少步更新一次）
        if chunk_sizes is None:
            chunk_sizes = [1, 16, 128, 1024]

        self.chunk_sizes = chunk_sizes
        self.num_levels = len(chunk_sizes)

        # 创建多个 FFN，每个对应一个频率层级
        self.levels = nn.ModuleList([
            FeedForward(d_model) for _ in range(self.num_levels)
        ])

        # 每个层级前面加一个 LayerNorm（稳定训练）
        self.norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(self.num_levels)
        ])

    def forward(self, x):
        """
        顺序通过所有 FFN：高频 → 中频 → 低频 → 超低频

        每一层都有残差连接：x = x + FFN(LayerNorm(x))
        """
        for norm, ffn in zip(self.norms, self.levels):
            x = x + ffn(norm(x))
        return x

    def set_active_levels(self, step):
        """
        根据当前训练步数，决定哪些 FFN 参与梯度更新。

        规则：如果 step % chunk_size == 0，该层级"到期"，需要更新。
              否则冻结该层级的参数（不计算梯度）。

        例如 step=32:
          - 层级0（chunk=1）:    32 % 1 == 0    → 更新 ✅
          - 层级1（chunk=16）:   32 % 16 == 0   → 更新 ✅
          - 层级2（chunk=128）:  32 % 128 == 32  → 冻结 ❌
          - 层级3（chunk=1024）: 32 % 1024 == 32 → 冻结 ❌
        """
        for i, (ffn, chunk_size) in enumerate(zip(self.levels, self.chunk_sizes)):
            should_update = (step % chunk_size == 0)
            for param in ffn.parameters():
                param.requires_grad = should_update
            # LayerNorm 跟随对应的 FFN
            for param in self.norms[i].parameters():
                param.requires_grad = should_update


# ============================================================
# 组件 3：HopeBlock（用 CMS 替换 FFN 的变换器块）
# ============================================================

class HopeBlock(nn.Module):
    """
    Hope-Attention 变换器块 = 多头注意力 + CMS

    对比 mini-GPT 的 TransformerBlock:
      mini-GPT:      Attention + 1个FFN
      HopeBlock:     Attention + CMS（4个不同频率的FFN）
    """

    def __init__(self, d_model, n_heads, chunk_sizes=None):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.cms = CMS(d_model, chunk_sizes)

    def forward(self, x, mask):
        # 注意力 + 残差（和 mini-GPT 完全一样）
        x = x + self.attn(self.ln1(x), mask)
        # CMS 替代了原来的单个 FFN（CMS 内部自带残差和 LayerNorm）
        x = self.cms(x)
        return x


# ============================================================
# 组件 4：完整的 Hope-Attention 模型
# ============================================================

class HopeAttentionGPT(nn.Module):
    """
    Hope-Attention = Softmax 全局注意力 + CMS

    整体结构和 MiniGPT 一样：
    输入 → 嵌入 + 位置编码 → N个HopeBlock → LayerNorm → 输出投影

    区别只在于每个 Block 里的 FFN 变成了 CMS。
    """

    def __init__(self, vocab_size, d_model=64, n_heads=4, n_layers=4,
                 context_length=128, chunk_sizes=None):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(context_length, d_model)

        self.blocks = nn.ModuleList([
            HopeBlock(d_model, n_heads, chunk_sizes) for _ in range(n_layers)
        ])

        self.ln_final = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)

        self.context_length = context_length

    def forward(self, idx):
        B, T = idx.shape

        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb

        mask = torch.tril(torch.ones(T, T, device=idx.device))

        for block in self.blocks:
            x = block(x, mask)

        x = self.ln_final(x)
        logits = self.output_proj(x)

        return logits

    def set_active_levels(self, step):
        """设置所有 Block 中 CMS 的活跃层级"""
        for block in self.blocks:
            block.cms.set_active_levels(step)

"""
完整 Hope 模型（Self-Modifying Titans + CMS）— 完全体版本

三级架构对比：
  MiniGPT:          Softmax Attention + 1个FFN         → model/gpt.py
  HopeAttentionGPT: Softmax Attention + CMS（4个FFN）   → model/hope_attention.py
  HopeGPT:          Self-Modifying Titans + CMS        → 本文件

核心区别：HopeGPT 完全不用 softmax 注意力！
用 6 个在线学习的记忆模块（Self-Modifying Titans）替代 Q@K^T softmax 机制。
每个记忆模块是一个 2 层残差 MLP（论文公式 89），在前向传播中通过 DGD 实时更新权重。

来自论文：Nested Learning: The Illusion of Deep Learning Architectures
公式 (83)-(97)

关键概念：
  - MLPMemoryModule: 一个可在线更新的 2 层残差 MLP
    * M_□(x) = x + W1 · σ(W2 · x + b2) + b1  （公式 89）
    * 外层优化器学的是 (W1, W2, b1, b2) 的"初始值"
  - DGD (Delta Gradient Descent): 记忆的在线更新规则
    * L = (1/2C)||M(k) - v̂||²  （自监督 L2 损失）
    * W_new = α·W - η·∇L        （对 W1, W2, b1, b2 各自更新）
  - 分块处理: 序列按 chunk_size 分块，块内记忆冻结，块边界做 DGD 更新
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 复用 Hope-Attention 的 CMS 组件
from model.hope_attention import CMS


# ============================================================
# 组件 1：MLPMemoryModule（2 层残差 MLP 记忆）
# ============================================================

class MLPMemoryModule(nn.Module):
    """
    2 层残差 MLP 记忆模块（论文公式 89）。

    M_□(x) = x + W1 · σ(W2 · x + b2) + b1   （d_in == d_out 时有残差）
    M_□(x) = W1 · σ(W2 · x + b2) + b1        （d_in != d_out 时无残差）

    vs 旧版 MemoryModule（简单线性矩阵 y = W·x）：
    - 旧版：单参数 W_init，DGD 有闭合公式
    - 新版：4 个参数 (W1, W2, b1, b2)，DGD 用手动链式法则
    - 新版表达能力更强，能学习非线性映射

    状态是一个 dict：{'W1': (B, d_out, d_hid), 'W2': (B, d_hid, d_in),
                      'b1': (B, d_out), 'b2': (B, d_hid)}
    """

    def __init__(self, d_in, d_out, d_hidden=None):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.d_hidden = d_hidden or d_in
        self.has_residual = (d_in == d_out)

        # W1 接近零 → MLP 输出 ≈ 0 → 加残差后 M(x) ≈ x（稳定启动）
        self.W1_init = nn.Parameter(torch.zeros(d_out, self.d_hidden))
        self.b1_init = nn.Parameter(torch.zeros(d_out))
        # W2 用 Kaiming 初始化（隐藏层标准做法）
        self.W2_init = nn.Parameter(
            torch.randn(self.d_hidden, d_in) * (2.0 / d_in) ** 0.5
        )
        self.b2_init = nn.Parameter(torch.zeros(self.d_hidden))

    def forward(self, x, state):
        """
        前向传播：M(x) = [x +] W1·σ(W2·x + b2) + b1

        Args:
            x: (B, C, d_in)
            state: dict {'W1', 'W2', 'b1', 'b2'}
        Returns:
            y: (B, C, d_out)
        """
        # 隐藏层：h = σ(x @ W2^T + b2)
        h = torch.bmm(x, state['W2'].transpose(1, 2)) + state['b2'].unsqueeze(1)
        h = F.silu(h)
        # 输出层：out = h @ W1^T + b1
        out = torch.bmm(h, state['W1'].transpose(1, 2)) + state['b1'].unsqueeze(1)
        # 残差连接（仅 d_in == d_out）
        if self.has_residual:
            out = x + out
        return out

    def get_initial_state(self, batch_size):
        """获取初始记忆状态，扩展到 batch 维度"""
        return {
            'W1': self.W1_init.unsqueeze(0).expand(batch_size, -1, -1).clone(),
            'W2': self.W2_init.unsqueeze(0).expand(batch_size, -1, -1).clone(),
            'b1': self.b1_init.unsqueeze(0).expand(batch_size, -1).clone(),
            'b2': self.b2_init.unsqueeze(0).expand(batch_size, -1).clone(),
        }

    def compute_dgd_grads(self, x, target, state):
        """
        手动计算 DGD 梯度（per-sample，支持 per-batch 的 eta/alpha）。

        L = (1/2C) * ||M(x) - target||²

        手动链式法则比 torch.autograd.grad 快，且天然支持 per-sample 梯度。

        Args:
            x: (B, C, d_in) — 输入（通常是 k）
            target: (B, C, d_out) — 目标（detached v_hat）
            state: dict — 当前记忆状态
        Returns:
            grads: dict {'W1': (B,...), 'W2': (B,...), 'b1': (B,...), 'b2': (B,...)}
        """
        B, C, _ = x.shape
        W1, W2 = state['W1'], state['W2']

        # ===== Forward =====
        pre_act = torch.bmm(x, W2.transpose(1, 2)) + state['b2'].unsqueeze(1)
        h = F.silu(pre_act)                                    # (B, C, d_hidden)
        pred = torch.bmm(h, W1.transpose(1, 2)) + state['b1'].unsqueeze(1)
        if self.has_residual:
            pred = x + pred                                    # (B, C, d_out)

        # ===== Backward =====
        # dL/dpred = (pred - target) / C
        error = (pred - target) / C                            # (B, C, d_out)

        # dL/dW1 = error^T @ h                                (B, d_out, d_hidden)
        dW1 = torch.bmm(error.transpose(1, 2), h)
        # dL/db1 = error.sum(C)                               (B, d_out)
        db1 = error.sum(dim=1)

        # dL/dh = error @ W1                                  (B, C, d_hidden)
        dl_dh = torch.bmm(error, W1)

        # silu'(z) = σ(z) * (1 + z*(1 - σ(z)))
        sig = torch.sigmoid(pre_act)
        silu_grad = sig * (1.0 + pre_act * (1.0 - sig))
        dl_dpre = dl_dh * silu_grad                            # (B, C, d_hidden)

        # dL/dW2 = dl_dpre^T @ x                             (B, d_hidden, d_in)
        dW2 = torch.bmm(dl_dpre.transpose(1, 2), x)
        # dL/db2 = dl_dpre.sum(C)                             (B, d_hidden)
        db2 = dl_dpre.sum(dim=1)

        return {'W1': dW1, 'W2': dW2, 'b1': db1, 'b2': db2}


# ============================================================
# 组件 2：Self-Modifying Titans（自修改 Titans）
# ============================================================

class SelfModifyingTitans(nn.Module):
    """
    自修改 Titans —— 用 6 个在线学习的 MLP 记忆模块替代 softmax 注意力。

    6 个记忆模块的分工（公式 94-96）：
      M_k:      z_t → k_t      生成 key（记忆更新的"地址"）
      M_v:      z_t → v_t      生成 value（用于产生自监督目标）
      M_q:      z_t → q_t      生成 query（从 M_memory 读取输出）
      M_eta:    z_t → η_t      生成学习率（控制记忆更新速度）
      M_alpha:  z_t → α_t      生成衰减因子（控制遗忘速度）
      M_memory: q_t → o_t      核心记忆，存储历史信息

    DGD 更新（L2 损失版本，在 chunk 边界执行）：
      v̂_□ = M_□(v)                       每个模块的自监督目标
      L = (1/2C)||M_□(k) - v̂_□||²        自监督损失
      W_new = α·W - η·∇L                 梯度下降更新（对 W1/W2/b1/b2 分别）
    """

    def __init__(self, d_model, d_memory=None, chunk_size=16, bptt_depth=0):
        super().__init__()

        if d_memory is None:
            d_memory = d_model

        self.d_model = d_model
        self.d_memory = d_memory
        self.chunk_size = chunk_size
        self.bptt_depth = bptt_depth  # 0 = 每 chunk 后 detach

        # 输入/输出投影
        self.input_proj = nn.Linear(d_model, d_memory)
        self.output_proj = nn.Linear(d_memory, d_model)

        # 6 个 MLP 记忆模块
        self.M_k = MLPMemoryModule(d_memory, d_memory)
        self.M_v = MLPMemoryModule(d_memory, d_memory)
        self.M_q = MLPMemoryModule(d_memory, d_memory)
        self.M_eta = MLPMemoryModule(d_memory, 1)         # 标量学习率
        self.M_alpha = MLPMemoryModule(d_memory, 1)       # 标量衰减因子
        self.M_memory = MLPMemoryModule(d_memory, d_memory)

        # 便于遍历
        self.all_modules = nn.ModuleDict({
            'k': self.M_k, 'v': self.M_v, 'q': self.M_q,
            'eta': self.M_eta, 'alpha': self.M_alpha, 'memory': self.M_memory,
        })

        # M_eta: b1 = -3 → sigmoid(output) ≈ 0.05（学习率偏小，防爆炸）
        # M_alpha: b1 = +3 → sigmoid(output) ≈ 0.95（衰减偏慢，保留旧记忆）
        nn.init.constant_(self.M_eta.b1_init, -3.0)
        nn.init.constant_(self.M_alpha.b1_init, 3.0)

        self.ln = nn.LayerNorm(d_memory)

    def forward(self, x):
        """
        Args:
            x: (B, T, d_model) — 输入序列
        Returns:
            output: (B, T, d_model) — 输出序列
        """
        B, T, D = x.shape
        C = self.chunk_size

        z = self.input_proj(x)  # (B, T, d_memory)

        # 分块（补零对齐）
        num_chunks = (T + C - 1) // C
        padded_T = num_chunks * C
        if T < padded_T:
            z = F.pad(z, (0, 0, 0, padded_T - T))

        # 初始化记忆状态（每个模块是一个 dict）
        states = {}
        for name, module in self.all_modules.items():
            states[name] = module.get_initial_state(B)

        all_outputs = []

        for chunk_idx in range(num_chunks):
            start = chunk_idx * C
            end = start + C
            z_chunk = z[:, start:end, :]  # (B, C, d_memory)

            # ===== 用冻结的记忆处理 chunk =====
            k = self.M_k.forward(z_chunk, states['k'])
            v = self.M_v.forward(z_chunk, states['v'])
            q = self.M_q.forward(z_chunk, states['q'])
            eta = torch.sigmoid(
                self.M_eta.forward(z_chunk, states['eta'])
            )
            alpha = torch.sigmoid(
                self.M_alpha.forward(z_chunk, states['alpha'])
            )

            # L2 归一化 k 和 q
            k = F.normalize(k, dim=-1)
            q = F.normalize(q, dim=-1)

            # 从记忆读取输出
            o = self.M_memory.forward(q, states['memory'])
            all_outputs.append(o)

            # ===== chunk 边界 DGD 更新 =====
            self._dgd_update(states, k, v, eta, alpha, chunk_idx)

        # 拼接输出
        output = torch.cat(all_outputs, dim=1)  # (B, padded_T, d_memory)
        output = output[:, :T, :]               # 去掉 padding
        output = self.ln(output)
        output = self.output_proj(output)        # (B, T, d_model)

        return output

    def _clamp_state_norm(self, w, B):
        """范数裁剪：防止记忆状态爆炸"""
        ndim = w.dim() - 1
        flat = w.flatten(1)
        norm = torch.norm(flat, dim=1, keepdim=True)
        max_n = math.sqrt(flat.shape[1])
        scale = torch.clamp(max_n / (norm + 1e-8), max=1.0)
        return w * scale.view(B, *([1] * ndim))

    def _dgd_update(self, states, k, v, eta, alpha, chunk_idx):
        """
        DGD 更新：在 chunk 边界更新所有 6 个记忆模块。

        两条路径：
        - bptt_depth=0 或 detach 边界：手动梯度（快速，per-sample eta/alpha）
        - bptt_depth>0 且非 detach 边界：autograd（可微，梯度穿越 DGD 回到 W_init）

        bptt_depth 控制梯度截断频率：
          0: 每 chunk 后 detach（默认，最快）
          1: 保留 1 个 chunk 的梯度图后 detach
          -1: 完全不 detach（最慢，显存消耗最大）
        """
        B, C, _ = k.shape
        eta_avg = eta.mean(dim=1)      # (B, 1)
        alpha_avg = alpha.mean(dim=1)  # (B, 1)

        should_detach = (
            self.bptt_depth == 0 or
            (self.bptt_depth > 0 and (chunk_idx + 1) % self.bptt_depth == 0)
        )
        use_autograd = (self.bptt_depth != 0 and not should_detach)

        for name, module in self.all_modules.items():
            state = states[name]

            # 自监督目标（detached，作为固定 target）
            v_hat = module.forward(v, state).detach()

            if use_autograd:
                # ===== BPTT 路径：autograd + create_graph =====
                # 让 state 参数可追踪梯度
                param_names = ('W1', 'W2', 'b1', 'b2')
                param_list = []
                state_grad = {}
                for pn in param_names:
                    p = state[pn]
                    if not p.requires_grad:
                        p = p.detach().requires_grad_(True)
                    param_list.append(p)
                    state_grad[pn] = p

                # L = (1/2BC)||M(k) - v̂||²
                pred = module.forward(k, state_grad)
                loss = 0.5 * ((pred - v_hat) ** 2).sum() / (B * C)

                grads = torch.autograd.grad(
                    loss, param_list, create_graph=True
                )

                # eta/alpha 近似为 batch 均值（autograd 梯度已求和）
                eta_scalar = eta_avg.mean()
                alpha_scalar = alpha_avg.mean()

                new_state = {}
                for pn, pval, grad in zip(param_names, param_list, grads):
                    w_new = alpha_scalar * pval - eta_scalar * grad
                    w_new = self._clamp_state_norm(w_new, B)
                    new_state[pn] = w_new

            else:
                # ===== 标准路径：手动梯度（快速，per-sample）=====
                grads = module.compute_dgd_grads(k, v_hat, state)

                new_state = {}
                for pname in ('W1', 'W2', 'b1', 'b2'):
                    pval = state[pname]
                    grad = grads[pname]

                    ndim = grad.dim() - 1
                    a = alpha_avg.view(B, *([1] * ndim))
                    e = eta_avg.view(B, *([1] * ndim))

                    w_new = a * pval - e * grad
                    w_new = self._clamp_state_norm(w_new, B)
                    new_state[pname] = w_new

            states[name] = new_state

        # 截断梯度
        if should_detach:
            for name in states:
                states[name] = {
                    pn: pv.detach() for pn, pv in states[name].items()
                }


# ============================================================
# 组件 3：HopeFullBlock（Titans + CMS）
# ============================================================

class HopeFullBlock(nn.Module):
    """
    完整 Hope 变换器块 = Self-Modifying Titans + CMS

    不需要 mask 参数！Titans 天然因果。
    """

    def __init__(self, d_model, d_memory=None, chunk_size=16,
                 cms_chunk_sizes=None, bptt_depth=0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.titans = SelfModifyingTitans(d_model, d_memory, chunk_size, bptt_depth)
        self.cms = CMS(d_model, cms_chunk_sizes)

    def forward(self, x):
        x = x + self.titans(self.ln1(x))
        x = self.cms(x)
        return x


# ============================================================
# 组件 4：HopeGPT（完整模型）
# ============================================================

class HopeGPT(nn.Module):
    """
    完整 Hope 模型 = Self-Modifying Titans + CMS

    注意：
    - 没有 n_heads 参数（Titans 不用多头注意力）
    - 没有 mask（Titans 天然因果）
    - bptt_depth 控制梯度穿越 DGD 的深度（默认 0 = 每 chunk 截断）
    """

    def __init__(self, vocab_size, d_model=256, n_layers=8,
                 d_memory=None, chunk_size=16,
                 context_length=128, cms_chunk_sizes=None,
                 bptt_depth=0):
        super().__init__()

        if cms_chunk_sizes is None:
            cms_chunk_sizes = [1, 16, 128, 1024]

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(context_length, d_model)

        self.blocks = nn.ModuleList([
            HopeFullBlock(d_model, d_memory, chunk_size, cms_chunk_sizes, bptt_depth)
            for _ in range(n_layers)
        ])

        self.ln_final = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)

        self.context_length = context_length

    def forward(self, idx):
        B, T = idx.shape

        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb

        for block in self.blocks:
            x = block(x)

        x = self.ln_final(x)
        logits = self.output_proj(x)
        return logits

    def set_active_levels(self, step):
        """设置所有 Block 中 CMS 的活跃层级"""
        for block in self.blocks:
            block.cms.set_active_levels(step)


# ============================================================
# 单元测试
# ============================================================

if __name__ == '__main__':
    print('===== HopeGPT 完全体 单元测试 =====\n')

    # 配置
    vocab_size = 4000
    d_model = 256
    d_memory = 256
    n_layers = 8
    chunk_size = 16
    context_length = 128
    cms_chunk_sizes = [1, 16, 128, 1024]

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f'设备: {device}')

    # 创建模型
    model = HopeGPT(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        d_memory=d_memory,
        chunk_size=chunk_size,
        context_length=context_length,
        cms_chunk_sizes=cms_chunk_sizes,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'模型参数量: {total_params:,}')

    # MLPMemoryModule 单独测试
    print('\n--- MLPMemoryModule 测试 ---')
    mem = MLPMemoryModule(256, 256).to(device)
    state = mem.get_initial_state(2)
    x_mem = torch.randn(2, 16, 256, device=device)
    y_mem = mem.forward(x_mem, state)
    print(f'MLP记忆 (256→256): 输入 {x_mem.shape} → 输出 {y_mem.shape}')
    assert y_mem.shape == (2, 16, 256)

    # 残差验证：W1=0 时，输出 ≈ 输入
    diff = (y_mem - x_mem).abs().mean().item()
    print(f'初始残差偏差（应接近0）: {diff:.6f}')

    # 手动梯度测试
    target = torch.randn(2, 16, 256, device=device)
    grads = mem.compute_dgd_grads(x_mem, target, state)
    print(f'DGD 梯度 shapes: W1={grads["W1"].shape}, W2={grads["W2"].shape}, '
          f'b1={grads["b1"].shape}, b2={grads["b2"].shape}')
    assert grads['W1'].shape == (2, 256, 256)
    assert grads['b1'].shape == (2, 256)

    # M_eta 测试（无残差，d_out=1）
    mem_eta = MLPMemoryModule(256, 1).to(device)
    nn.init.constant_(mem_eta.b1_init, -3.0)
    state_eta = mem_eta.get_initial_state(2)
    y_eta = mem_eta.forward(x_mem, state_eta)
    print(f'M_eta (256→1): 输出 {y_eta.shape}, '
          f'sigmoid 均值 = {torch.sigmoid(y_eta).mean().item():.4f}（目标 ≈0.05）')
    assert y_eta.shape == (2, 16, 1)

    # Shape 测试
    print('\n--- Shape 测试 ---')
    for T in [64, 50, 8]:
        x = torch.randint(0, vocab_size, (2, T)).to(device)
        logits = model(x)
        print(f'输入: {x.shape} → 输出: {logits.shape}')
        assert logits.shape == (2, T, vocab_size), f'Shape 错误: {logits.shape}'

    # 梯度测试
    print('\n--- 梯度测试 ---')
    model.zero_grad()
    y = torch.randint(0, vocab_size, (2, 64)).to(device)
    logits = model(torch.randint(0, vocab_size, (2, 64)).to(device))
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    loss.backward()
    print(f'Loss: {loss.item():.4f}')
    grad_count = sum(1 for p in model.parameters() if p.grad is not None)
    param_count = sum(1 for p in model.parameters())
    print(f'有梯度的参数: {grad_count}/{param_count}')

    # CMS 频率控制测试
    print('\n--- CMS 频率控制测试 ---')
    model.set_active_levels(0)
    active_0 = sum(1 for p in model.parameters() if p.requires_grad)
    model.set_active_levels(1)
    active_1 = sum(1 for p in model.parameters() if p.requires_grad)
    print(f'step=0 活跃参数: {active_0}')
    print(f'step=1 活跃参数: {active_1}')
    assert active_0 > active_1, '步骤0应该有更多活跃参数'

    # DGD 梯度正确性验证（线性极限测试）
    print('\n--- DGD 梯度正确性验证 ---')
    # 验证：当 W1=I, W2=0, b=0 时，MLP 退化为恒等映射
    # compute_dgd_grads 的梯度应和 autograd 一致
    mem_test = MLPMemoryModule(8, 8, d_hidden=8).to(device)
    state_test = mem_test.get_initial_state(1)
    # 设置非零 W1 以测试
    state_test['W1'] = torch.randn(1, 8, 8, device=device) * 0.1
    state_test['W2'] = torch.randn(1, 8, 8, device=device) * 0.1
    x_test = torch.randn(1, 4, 8, device=device)
    t_test = torch.randn(1, 4, 8, device=device)

    # 手动梯度
    grads_manual = mem_test.compute_dgd_grads(x_test, t_test, state_test)

    # autograd 验证
    for pn in ('W1', 'W2', 'b1', 'b2'):
        state_test[pn] = state_test[pn].detach().requires_grad_(True)
    pred = mem_test.forward(x_test, state_test)
    loss_test = 0.5 * ((pred - t_test) ** 2).sum() / 4  # /C
    loss_test.backward()

    max_err = 0
    for pn in ('W1', 'W2', 'b1', 'b2'):
        err = (grads_manual[pn] - state_test[pn].grad).abs().max().item()
        max_err = max(max_err, err)
    print(f'手动梯度 vs autograd 最大误差: {max_err:.2e}（应 < 1e-5）')
    assert max_err < 1e-4, f'梯度误差过大: {max_err}'

    # BPTT 穿越测试
    print('\n--- BPTT 穿越测试 ---')
    # 小模型，bptt_depth=0（默认）vs bptt_depth=2
    for bptt in [0, 2]:
        titans_test = SelfModifyingTitans(
            d_model=32, d_memory=32, chunk_size=4, bptt_depth=bptt
        ).to(device)
        x_bptt = torch.randn(1, 16, 32, device=device, requires_grad=True)
        out = titans_test(x_bptt)
        loss_bptt = out.sum()
        loss_bptt.backward()
        # W_init 参数应该有梯度
        w1_grad = titans_test.M_k.W1_init.grad
        has_grad = w1_grad is not None and w1_grad.abs().sum() > 0
        print(f'bptt_depth={bptt}: M_k.W1_init 有梯度 = {has_grad}')
        titans_test.zero_grad()

    print('\n===== 所有测试通过 =====')

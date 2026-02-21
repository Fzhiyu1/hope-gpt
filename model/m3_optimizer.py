"""
M3 优化器（Multi-scale Momentum Muon）

来自论文：Nested Learning: The Illusion of Deep Learning Architectures（算法 1）

核心创新（vs Adam）：
  1. 双层动量：高频（每步更新）+ 低频（每 slow_interval 步更新）
     - 高频动量捕捉近期梯度趋势
     - 低频动量捕捉长期梯度结构
  2. Newton-Schulz 正交化：将动量映射到更"正交"的方向
     - 来自 Muon 优化器，计算极分解的正交因子
     - 效果：更均匀的参数更新，避免某些方向过度更新
  3. 方差缩放：类 Adam 的自适应学习率

用法：
    optimizer = M3Optimizer(model.parameters(), lr=3e-4, slow_interval=16)

Newton-Schulz 只对 2D 参数（权重矩阵）做正交化。
1D 参数（bias, LayerNorm）回退到标准 Adam 行为。
"""

import torch
from torch.optim import Optimizer


class M3Optimizer(Optimizer):
    """
    M3: Multi-scale Momentum Muon 优化器

    参数：
        params: 模型参数
        lr: 学习率（默认 3e-4）
        betas: (β1, β2) 高频动量衰减和方差衰减（默认 (0.9, 0.999)）
        beta_slow: β3 低频动量衰减（默认 0.99）
        slow_interval: Ĉ 低频动量更新间隔（默认 16，建议和 chunk_size 对齐）
        mix_alpha: α 高低频动量混合权重（默认 0.7，偏向高频）
        ns_steps: Newton-Schulz 迭代次数（默认 5）
        eps: 方差缩放的数值稳定项（默认 1e-8）
    """

    def __init__(self, params, lr=3e-4, betas=(0.9, 0.999),
                 beta_slow=0.99, slow_interval=16,
                 mix_alpha=0.7, ns_steps=5, eps=1e-8):
        defaults = dict(
            lr=lr, betas=betas, beta_slow=beta_slow,
            slow_interval=slow_interval, mix_alpha=mix_alpha,
            ns_steps=ns_steps, eps=eps,
        )
        super().__init__(params, defaults)

    @staticmethod
    def _newton_schulz(M, steps=5):
        """
        Newton-Schulz 正交化迭代（Muon 风格多项式版本）。

        计算极分解 M = U·P 中的正交因子 U 的近似。
        使用三次多项式迭代（系数来自 Muon 优化器），5 步即可收敛。

        Args:
            M: 2D 张量（d_out, d_in）
            steps: 迭代次数
        Returns:
            X: 正交化后的矩阵
        """
        if M.dim() != 2:
            return M

        norm = torch.norm(M)
        if norm < 1e-8:
            return M

        # Muon 风格三次多项式系数（收敛更快更稳定）
        a, b, c = 3.4445, -4.7750, 2.0315

        X = M / norm

        # 确保 rows >= cols，否则转置
        transposed = False
        if X.shape[0] < X.shape[1]:
            X = X.T
            transposed = True

        for _ in range(steps):
            A = X @ X.T                      # (rows, rows)
            B = b * A + c * A @ A            # 二次多项式
            X = a * X + B @ X               # 三次更新

        if transposed:
            X = X.T

        return X

    @torch.no_grad()
    def step(self, closure=None):
        """执行一步优化"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            beta_slow = group['beta_slow']
            slow_interval = group['slow_interval']
            mix_alpha = group['mix_alpha']
            ns_steps = group['ns_steps']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('M3 不支持稀疏梯度')

                state = self.state[p]

                # 初始化状态
                if len(state) == 0:
                    state['step'] = 0
                    state['m_fast'] = torch.zeros_like(p)   # 高频动量
                    state['m_slow'] = torch.zeros_like(p)   # 低频动量
                    state['v'] = torch.zeros_like(p)        # 方差（类 Adam）
                    state['grad_accum'] = torch.zeros_like(p)  # 低频梯度累积

                state['step'] += 1
                step = state['step']

                m_fast = state['m_fast']
                m_slow = state['m_slow']
                v = state['v']
                grad_accum = state['grad_accum']

                # 1. 更新高频动量：M^(1) = β1·M^(1) + (1-β1)·g
                m_fast.mul_(beta1).add_(grad, alpha=1 - beta1)

                # 2. 累积梯度用于低频动量
                grad_accum.add_(grad)

                # 3. 每 slow_interval 步更新低频动量
                if step % slow_interval == 0:
                    m_slow.mul_(beta_slow).add_(
                        grad_accum / slow_interval, alpha=1 - beta_slow
                    )
                    grad_accum.zero_()

                # 4. 更新方差：v = β2·v + (1-β2)·g²
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # 5. 偏差校正
                bc1 = 1 - beta1 ** step
                bc2 = 1 - beta2 ** step
                m_fast_hat = m_fast / bc1

                slow_steps = step // slow_interval
                if slow_steps > 0:
                    bc_slow = 1 - beta_slow ** slow_steps
                    m_slow_hat = m_slow / bc_slow
                else:
                    m_slow_hat = m_slow

                v_hat = v / bc2

                # 6. Newton-Schulz 正交化（只对 2D 参数）
                if p.dim() == 2:
                    m_fast_ns = self._newton_schulz(m_fast_hat, ns_steps)
                    m_slow_ns = self._newton_schulz(m_slow_hat, ns_steps)
                else:
                    m_fast_ns = m_fast_hat
                    m_slow_ns = m_slow_hat

                # 7. 混合：m = α·NS(M^(1)) + (1-α)·NS(M^(2))
                m_combined = mix_alpha * m_fast_ns + (1 - mix_alpha) * m_slow_ns

                # 8. 方差缩放 + 更新参数
                denom = v_hat.sqrt().add_(eps)
                p.addcdiv_(m_combined, denom, value=-lr)

        return loss


# ============================================================
# 单元测试
# ============================================================

if __name__ == '__main__':
    print('===== M3 优化器测试 =====\n')

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f'设备: {device}')

    # 简单优化问题：最小化 ||Wx - y||²
    torch.manual_seed(42)
    d = 32
    W_true = torch.randn(d, d, device=device)
    x_data = torch.randn(100, d, device=device)
    y_data = x_data @ W_true.T

    # 用 M3 优化
    W = torch.nn.Parameter(torch.randn(d, d, device=device))
    optimizer = M3Optimizer([W], lr=1e-2, slow_interval=4)

    for step in range(500):
        pred = x_data @ W.T
        loss = ((pred - y_data) ** 2).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if step % 100 == 0:
            print(f'步骤 {step:>4d} | loss = {loss.item():.6f}')

    print(f'\n最终 loss: {loss.item():.6f}')
    assert loss.item() < 1.0, f'M3 优化失败，loss 过高: {loss.item()}'
    print('M3 可以收敛！')

    # 对比 Adam
    print('\n--- Adam 对比 ---')
    W_adam = torch.nn.Parameter(torch.randn(d, d, device=device))
    opt_adam = torch.optim.Adam([W_adam], lr=1e-2)
    for step in range(500):
        pred = x_data @ W_adam.T
        loss_adam = ((pred - y_data) ** 2).mean()
        loss_adam.backward()
        opt_adam.step()
        opt_adam.zero_grad()
    print(f'Adam 最终 loss: {loss_adam.item():.6f}')
    print(f'M3   最终 loss: {loss.item():.6f}')

    # 1D 参数应正常工作
    print('\n--- 1D 参数测试 ---')
    bias = torch.nn.Parameter(torch.randn(d, device=device))
    opt_1d = M3Optimizer([bias], lr=1e-1, slow_interval=4)
    target = torch.randn(d, device=device)
    for _ in range(200):
        loss_1d = ((bias - target) ** 2).sum()
        loss_1d.backward()
        opt_1d.step()
        opt_1d.zero_grad()
    print(f'1D 优化最终 loss: {loss_1d.item():.6f}')
    assert loss_1d.item() < 1.0, f'1D 优化失败: {loss_1d.item()}'

    print('\n===== 所有测试通过 =====')

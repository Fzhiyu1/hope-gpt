# HopeGPT 过夜训练指南

本指南供 RTX 5070 机器上的 Claude Code 执行。

## 项目位置

```
C:\Users\24038\Documents\run\hope-gpt
```

## 前置条件检查

在开始前，依次确认：

```bash
# 1. PyTorch 支持 CUDA（必须是 nightly 版本，支持 sm_120/Blackwell）
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

# 2. tokenizers 库已安装
python -c "import tokenizers; print('tokenizers OK:', tokenizers.__version__)"

# 3. datasets 库已安装（下载维基数据用）
python -c "import datasets; print('datasets OK:', datasets.__version__)"
```

如果缺少任何库：
```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
pip install tokenizers datasets
```

## 步骤一：拉取最新代码

```bash
cd C:\Users\24038\Documents\run\hope-gpt
git pull
```

确认看到以下关键文件：
- `train_hope_full.py` — 训练脚本（已升级：100M 模型 + FP16 + Gradient Checkpointing）
- `download_wiki.py` — 维基数据下载脚本
- `model/hope.py` — 模型（已加 gradient_checkpointing 支持）

## 步骤二：下载中文维基百科数据

```bash
python download_wiki.py
```

预期输出：
- 下载约 524MB
- 生成 `data/wiki_cn.txt`
- 约 25 万篇文章

**如果下载失败**（网络问题），备选方案：用现有数据合并训练
```bash
# Windows PowerShell:
Get-Content data\xiyouji.txt, data\poetry_tang.txt, data\poetry_songci.txt, data\poetry_songshi.txt | Set-Content data\combined.txt
# 然后用 data/combined.txt 代替 data/wiki_cn.txt
```

## 步骤三：开始过夜训练

```bash
python train_hope_full.py 20000 data/wiki_cn.txt
```

### 预期行为

启动后应看到：
```
使用 HuggingFace tokenizers（Rust 加速）
设备: cuda
HF BPE 训练完成：词表 8000
从头开始训练
模型参数量: ~100,000,000（约 1 亿）
FP16 混合精度: 开启
Gradient Checkpointing: 开启
```

训练日志格式：
```
步骤   0（本次     0/20000）| loss = 9.xxxx | 活跃CMS: 4/4
步骤 1000（本次  1000/20000）| loss = 6.xxxx | 活跃CMS: 1/4
  → checkpoint 已保存: checkpoints/hope-full-2000steps.pt
```

### 正常 loss 范围

| 步数 | 预期 loss | 说明 |
|------|----------|------|
| 0 | ~9.0 | ln(8000)=8.97，随机水平 |
| 1000 | 6.5-7.5 | 开始学习 |
| 5000 | 5.0-6.0 | 有明显进展 |
| 10000 | 4.0-5.5 | 模型在学中文 |
| 20000 | 3.5-5.0 | 训练完成 |

### 异常处理

**CUDA out of memory：**
修改 `train_hope_full.py` 中 `get_batch` 的 batch_size：
```python
def get_batch(batch_size=8):   # 从 16 降到 8
```
如果还是不够，降到 4，或者降模型规模：
```python
d_model = 384
n_layers = 12
context_length = 192
```

**loss 爆炸（突然变成 nan 或 >20）：**
Ctrl+C 停止，从最近的 checkpoint 继续：
```bash
python train_hope_full.py 20000 data/wiki_cn.txt checkpoints/hope-full-latest.pt
```

**loss 不下降（前 500 步始终 >8.5）：**
学习率可能太大或太小，尝试修改 `train_hope_full.py`：
```python
learning_rate = 5e-4   # 从 1e-3 降一半
```

## 步骤四：训练完成后

### 4.1 查看结果

训练完成后会自动：
- 保存最终 checkpoint 到 `checkpoints/hope-full-20000steps.pt`
- 保存 `checkpoints/hope-full-latest.pt`
- 运行生成测试（用"天地"、"明月"、"春風"作为 prompt）

### 4.2 手动测试生成效果

```bash
python chat.py checkpoints/hope-full-latest.pt
```

输入一些中文短语看生成效果。如果生成的文字有一定连贯性（不是完全乱码），说明训练成功。

### 4.3 推送到 GitHub

```bash
git add checkpoints/hope-full-latest.pt
git add checkpoints/hope-full-20000steps.pt
git commit -m "过夜训练完成：100M HopeGPT on 中文维基，20000 步，最终 loss=X.XXXX"
git push
```

注意：
- commit message 中替换 `X.XXXX` 为实际最终 loss 值
- checkpoint 文件可能较大（~400MB），push 可能需要几分钟
- 如果 push 失败（文件太大），用 Git LFS：
  ```bash
  git lfs install
  git lfs track "*.pt"
  git add .gitattributes
  git add checkpoints/hope-full-latest.pt
  git commit -m "过夜训练完成：100M HopeGPT，最终 loss=X.XXXX"
  git push
  ```

## 显存监控

训练开始后可以用另一个终端监控显存：
```bash
nvidia-smi
```

预期显存占用：8-11GB（12GB 总量）。如果接近 12GB，有 OOM 风险，参考上面的异常处理。

## 关键配置汇总

| 参数 | 值 |
|------|-----|
| d_model | 512 |
| n_layers | 16 |
| d_memory | 512 |
| chunk_size | 16 |
| context_length | 256 |
| learning_rate | 1e-3 |
| batch_size | 16 |
| FP16 | 开启（CUDA） |
| Gradient Checkpointing | 开启（CUDA） |
| 自动保存间隔 | 每 2000 步 |
| 总训练步数 | 20000 |

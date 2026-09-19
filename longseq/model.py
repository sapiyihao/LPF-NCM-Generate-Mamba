"""面向真实时间轴的纯 PyTorch 选择性状态空间网络。"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SelectiveSSM(nn.Module):
    """输入相关的 Δ/B/C 与门控扫描；分块并行实现，避免512次Python循环。

    该层是可在Windows/PyTorch直接运行的 Mamba 风格实现，不等同于
    mamba-ssm 官方 fused CUDA kernel；算法/效率比较时须注明。
    """

    def __init__(self, d_model: int, d_state: int = 8, kernel_size: int = 5, chunk_size: int = 32):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.chunk_size = chunk_size
        self.in_proj = nn.Linear(d_model, d_model * 2)
        self.conv = nn.Conv1d(d_model, d_model, kernel_size, groups=d_model, padding=kernel_size - 1)
        self.delta_proj = nn.Linear(d_model, d_model)
        self.bc_proj = nn.Linear(d_model, 2 * d_state)
        self.a_log = nn.Parameter(torch.linspace(-2.0, -0.3, d_state).repeat(d_model, 1))
        self.skip = nn.Parameter(torch.ones(d_model))
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        u, gate = self.in_proj(x).chunk(2, dim=-1)
        u = self.conv(u.transpose(1, 2))[:, :, :length].transpose(1, 2)
        u = F.silu(u).float()
        # 扫描在float32中完成，避免AMP下前缀积和累积和溢出。
        delta = (0.01 + 0.09 * torch.sigmoid(self.delta_proj(u))).float()
        b_t, c_t = self.bc_proj(u).float().chunk(2, dim=-1)
        a_rate = self.a_log.float().exp()
        state = torch.zeros(batch, self.d_model, self.d_state, device=x.device, dtype=torch.float32)
        outputs = []
        for start in range(0, length, self.chunk_size):
            end = min(start + self.chunk_size, length)
            valid = mask[:, start:end, None, None]
            decay = torch.exp(-delta[:, start:end, :, None] * a_rate[None, None])
            decay = torch.where(valid, decay, torch.ones_like(decay))
            drive = delta[:, start:end, :, None] * b_t[:, start:end, None, :] * u[:, start:end, :, None]
            drive = drive * valid
            prefix = torch.cumprod(decay, dim=1)
            states = prefix * (state[:, None] + torch.cumsum(drive / prefix.clamp_min(1e-12), dim=1))
            state = states[:, -1]
            y = (states * c_t[:, start:end, None, :]).sum(dim=-1)
            y = y + self.skip.float()[None, None, :] * u[:, start:end]
            outputs.append(y)
        y = torch.cat(outputs, dim=1).to(gate.dtype) * F.silu(gate)
        return self.out_proj(y) * mask.unsqueeze(-1)


class MambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(d_model, d_state)
        self.dropout = nn.Dropout(dropout)
        self.ff_norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, 2 * d_model), nn.SiLU(),
                                nn.Dropout(dropout), nn.Linear(2 * d_model, d_model))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.ssm(self.norm(x), mask))
        x = x + self.dropout(self.ff(self.ff_norm(x)))
        return x * mask.unsqueeze(-1)


class LongSequenceMamba(nn.Module):
    """[B,T,7] -> 四类车辆故障logits。"""

    def __init__(self, d_model: int = 32, d_state: int = 8, layers: int = 2,
                 dropout: float = 0.1, in_features: int = 7, classes: int = 4):
        super().__init__()
        self.input = nn.Sequential(nn.Linear(in_features, d_model), nn.LayerNorm(d_model), nn.SiLU())
        self.blocks = nn.ModuleList([MambaBlock(d_model, d_state, dropout) for _ in range(layers)])
        self.pool_score = nn.Linear(d_model, 1)
        self.classifier = nn.Sequential(nn.LayerNorm(3 * d_model), nn.Linear(3 * d_model, d_model),
                                        nn.SiLU(), nn.Dropout(dropout), nn.Linear(d_model, classes))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.input(x) * mask.unsqueeze(-1)
        for block in self.blocks:
            h = block(h, mask)
        valid = mask.unsqueeze(-1)
        mean = (h * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        maximum = h.masked_fill(~valid, -1e4).max(dim=1).values
        attention = self.pool_score(h).squeeze(-1).masked_fill(~mask, -1e4).softmax(dim=1)
        pooled = (h * attention.unsqueeze(-1)).sum(dim=1)
        return self.classifier(torch.cat([mean, maximum, pooled], dim=-1))

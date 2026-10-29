import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

@dataclass
class ModelArgs:
    vocab_size: int = 16
    d_model: int = 128
    n_heads: int = 4
    d_ff: int = 512
    n_layers: int = 2
    dropout: float = 0.1
    max_seq_len: int = 64
    pad_id: int = 0


class MultiHeadAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        assert args.d_model % args.n_heads == 0

        self.n_heads = args.n_heads
        self.head_dim = args.d_model // args.n_heads
        self.wq = nn.Linear(args.d_model, args.d_model, bias=False)
        self.wk = nn.Linear(args.d_model, args.d_model, bias=False)
        self.wv = nn.Linear(args.d_model, args.d_model, bias=False)
        self.wo = nn.Linear(args.d_model, args.d_model, bias=False)
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)


    def forward(self, q, k, v, mask=None):
        # mask: bool matrix, True = 允许看
        B, Tq, C = q.shape # C = d_model
        Tk = k.size(1)

        xq = self.wq(q).view(B, Tq, self.n_heads, self.head_dim).transpose(1, 2)
        xk = self.wk(k).view(B, Tk, self.n_heads, self.head_dim).transpose(1, 2)
        xv = self.wv(v).view(B, Tk, self.n_heads, self.head_dim).transpose(1, 2)

        # xq: B, h, Tq, self.head_dim
        # xk: B, h, Tk, self.head_dim
        # xv: B, h, Tk, self.head_dim

        scores = xq @ xk.transpose(-2, -1) / math.sqrt(self.head_dim)  # (B, h, Tq, Tk)

        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))

        attn = F.softmax(scores.float(), dim=-1).type_as(xq)
        attn = self.attn_dropout(attn) # (B, h, Tq, Tk)
        out = attn @ xv # (B, h, Tq, hd)

        out = out.transpose(1, 2).contiguous().view(B, Tq, C)

        return self.resid_dropout(self.wo(out))

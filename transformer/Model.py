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



class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps


    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)

        return self.weight * (x - mean) / torch.sqrt(var + self.eps) + self.bias



class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.w1 = nn.Linear(args.d_model, args.d_ff, bias=False)
        self.w2 = nn.Linear(args.d_ff, args.d_model, bias=False)
        self.dropout = nn.Dropout(args.dropout)


    def forward(self, x):
        return self.dropout(self.w2(F.relu(self.w1(x))))



class EncoderLayer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__() 
        self.attn_norm = LayerNorm(args.d_model)
        self.attn = MultiHeadAttention(args)
        self.ffn_norm = LayerNorm(args.d_model)
        self.ffn = MLP(args)


    def forward(self, x, src_mask):
        h = self.attn_norm(x)
        x = x + self.attn(h, h, h, src_mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x

class Encoder(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(args) for _ in range(args.n_layers)])
        self.norm = LayerNorm(args.d_model)


    def forward(self, x, src_mask):
        for layer in self.layers:
            x = layer(x, src_mask)
        
        return self.norm(x)



class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.self_norm = LayerNorm(args.d_model)
        self.self_attn = MultiHeadAttention(args)
        self.cross_norm = LayerNorm(args.d_model)
        self.cross_attn = MultiHeadAttention(args)
        self.ffn_norm = LayerNorm(args.d_model)
        self.ffn = MLP(args)


    def forward(self, x, enc_out, tgt_mask, src_mask):
        h = self.self_norm(x)
        x = x + self.self_attn(h, h, h, tgt_mask)                  # 掩码自注意力
        h = self.cross_norm(x)
        x = x + self.cross_attn(h, enc_out, enc_out, src_mask)     # Q 来自 decoder，K/V 来自 encoder
        x = x + self.ffn(self.ffn_norm(x))
        return x



class Decoder(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(args) for _ in range(args.n_layers)])
        self.norm = LayerNorm(args.d_model)

    def forward(self, x, enc_out, tgt_mask, src_mask):
        for layer in self.layers:
            x = layer(x, enc_out, tgt_mask, src_mask)
        return self.norm(x)



class PositionalEncoding(nn.Module):
    pe: torch.Tensor
    
    def __init__(self, args: ModelArgs):
        super().__init__()
        pe = torch.zeros(args.max_seq_len, args.d_model)
        pos = torch.arange(args.max_seq_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, args.d_model, 2) * (-math.log(10000.0) / args.d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]
import torch
from torch import nn

from model import ModelArgs, MultiHeadAttention

torch.manual_seed(0)
args = ModelArgs(dropout=0.0)

mha = MultiHeadAttention(args)
x = torch.randn(2, 10, args.d_model)
assert mha(x, x, x).shape == (2, 10, args.d_model)

causal = torch.tril(torch.ones(10, 10, dtype=torch.bool))[None, None]
y1 = mha(x, x, x, causal)


x2 = x.clone()

x2[:, 5:] = torch.randn_like(x2[:, 5:])
y2 = mha(x2, x2, x2, causal)
assert torch.allclose(y1[:, :5], y2[:, :5], atol=1e-6)
assert not torch.allclose(y1[:, 5:], y2[:, 5:])
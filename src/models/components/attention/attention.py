import torch
from torch import Tensor
import torch.nn as nn


class Attention(nn.Module):

    def __init__(self, channels: int, n_heads: int = 4):
        super(Attention, self).__init__()
        self.mha = nn.MultiheadAttention(channels, n_heads, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x: Tensor, cond: Tensor | None = None):
        """
        x: shape [b, c, w*h]
        """
        x_ln = self.ln(x)
        # cond is None: self_attention else cross_attention
        if cond is None:
            cond = x_ln

        attention_value, _ = self.mha(x_ln, cond, cond)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value


if __name__ == "__main__":
    input = torch.randn(2, 256)
    cond = torch.randn(2, 256)
    ca = Attention(256, 4)
    output = ca(input, cond=cond)
    print(output.shape)

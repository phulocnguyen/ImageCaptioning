import math

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from pickle import load
import os.path as osp

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class Glove_Transformer(nn.Module):

    def __init__(
        self,
        num_tokens: int = 1741,
        dim_model: int = 200,
        text_features: int = 256,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dropout_p: float = 0.1, 
        dataset_dir: str = 'data/flickr8k'
    ):
        super().__init__()
        self.pos_encoder = PositionalEncoding(dim_model, dropout_p)
        self.embedding = nn.Embedding.from_pretrained(
            self.load_weight_embedding(dataset_dir),
            freeze=True,
            padding_idx=0)
        self.transformer = nn.Transformer(
            d_model=dim_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout_p,
            batch_first=True
        )
        self.dim_model = dim_model
        self.linear_image = nn.Linear(text_features, dim_model)
        self.out = nn.Linear(dim_model, num_tokens)
    
    def load_weight_embedding(self, dataset_dir: str = 'data/flickr8k'):
        embedding_matrix_path = osp.join(dataset_dir, 'embedding_matrix.pkl')

        if not osp.exists(embedding_matrix_path):
            raise ValueError(
                "weight_embedding_path is not exist. Please check path or run datamodule to prepare"
            )

        with open(embedding_matrix_path, "rb") as file:
            embedding_matrix = load(file)
        print('Embedding_matrix:', embedding_matrix.shape)
        return embedding_matrix
    
    def forward(self, src, tgt, tgt_mask=None, src_pad_mask=None, tgt_pad_mask=None):
        # Src size must be (batch_size, src sequence length)
        # Tgt size must be (batch_size, tgt sequence length)
        # Embedding + positional encoding - Out size = (batch_size, sequence length, dim_model)
        # src = self.embedding(src) * math.sqrt(self.dim_model)
        src = self.linear_image(src).unsqueeze(1).expand(-1, tgt.shape[1], -1)
        tgt = self.embedding(tgt) * math.sqrt(self.dim_model)
        # src = self.positional_encoder(src)
        tgt = self.pos_encoder(tgt.permute(1, 0, 2))
        
        # We could use the parameter batch_first=True, but our KDL version doesn't support it yet, so we permute
        # to obtain size (sequence length, batch_size, dim_model),
        # src = src.permute(1, 0, 2)
        tgt = tgt.permute(1, 0, 2)
        if tgt_mask is None:
            """Generate a square causal mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
            """
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(len(tgt))
        # Transformer blocks - Out size = (sequence length, batch_size, num_tokens)
        transformer_out = self.transformer(src, tgt, tgt_mask=None, src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=tgt_pad_mask)
        out = self.out(transformer_out)
        
        return out[:, -1]
    def get_tgt_mask(self, size) -> torch.tensor:
        # Generates a squeare matrix where the each row allows one word more to be seen
        mask = torch.tril(torch.ones(size, size) == 1) # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0
        
        # EX for size=5:
        # [[0., -inf, -inf, -inf, -inf],
        #  [0.,   0., -inf, -inf, -inf],
        #  [0.,   0.,   0., -inf, -inf],
        #  [0.,   0.,   0.,   0., -inf],
        #  [0.,   0.,   0.,   0.,   0.]]
        
        return mask
if __name__ == "__main__":
    net = Glove_Transformer()

    sequence = torch.randint(0, 100, (2, 20))
    out = net(sequence)
    
    print(out.shape)
    

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

class Glove_Transformer_Encoder(nn.Module):

    def __init__(self, ntoken: int = 1741, d_model: int = 200, nhead: int = 10, d_hid: int = 200,
                 nlayers: int = 6, dropout: float = 0.5, dataset_dir: str = 'data/flickr8k'):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embedding = nn.Embedding.from_pretrained(
            self.load_weight_embedding(dataset_dir),
            freeze=True,
            padding_idx=0)
        self.d_model = d_model
        self.linear = nn.Linear(d_model, ntoken)

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

    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        if src_mask is None:
            """Generate a square causal mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
            """
            src_mask = nn.Transformer.generate_square_subsequent_mask(len(src))
        output = self.transformer_encoder(src, src_mask)
        output = self.linear(output)
        return output[:, -1]

if __name__ == "__main__":
    net = Glove_Transformer_Encoder()

    sequence = torch.randint(0, 100, (2, 20))
    out = net(sequence)
    
    print(out.shape)
    

import torch
import torch.nn as nn
from torch import Tensor
from pickle import load
import os.path as osp


class Glove_RNN(nn.Module):

    def __init__(
        self,
        embed_dim: int = 200,
        drop_rate: float = 0.5,
        text_features: int = 256,
        n_layer_rnn: int = 1,
        dataset_dir: str = 'data/flickr8k',
    ) -> None:
        """_summary_

        Args:
            embed_dim (int, optional): _description_. Defaults to 200.
            drop_rate (float, optional): _description_. Defaults to 0.5.
            text_features (int, optional): _description_. Defaults to 256.
            n_layer_rnn (int, optional): _description_. Defaults to 1.
            dataset_dir (str, optional): _description_. Defaults to 'data/flickr8k'.
        """
        super().__init__()

        self.embed = nn.Embedding.from_pretrained(
            self.load_weight_embedding(dataset_dir),
            freeze=True,
            padding_idx=0)

        # self.dropout = nn.Dropout(p=drop_rate)
        self.rnn = nn.RNN(input_size=embed_dim,
                          hidden_size=text_features,
                          num_layers=n_layer_rnn,
                          batch_first=True)

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

    def forward(self, sequence: Tensor) -> Tensor:
        """_summary_

        Args:
            sequence (Tensor): (batch, max_length)

        Returns:
            Tensor: (batch, text_features)
        """

        out = self.embed(sequence)
        # out = self.dropout(out)
        out, _ = self.rnn(out)  # return output and hidden state
        return out[:, -1]  # only get the last


if __name__ == "__main__":
    net = Glove_RNN()

    x = torch.randint(0, 100, (2, 20))
    out = net(x)
    print(x.shape, out.shape)
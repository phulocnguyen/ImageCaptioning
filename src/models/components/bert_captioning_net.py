import torch
import rootutils
import os.path as osp
import torch.nn as nn
from torch import Tensor
from pickle import load
from torch.nn.utils.rnn import pad_sequence

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.components.image_embedding import InceptionNet, ResnetEncoder
from src.models.components.text_embedding import (
    Glove_RNN,
    DecoderWithAttention,
)


class BertCaptioningNet(nn.Module):
    def __init__(
        self,
        image_embed_net,
        text_embed_net,
        features: int = 256,
        dataset_dir: str = "data/flickr8k",
    ) -> None:
        """_summary_

        Args:
            image_embed_net (_type_): _description_
            text_embed_net (_type_): _description_
            features (int, optional): _description_. Defaults to 256.
            dataset_dir (str, optional): _description_. Defaults to 'data/flickr8k'.
        """
        super().__init__()

        self.text_embed_net = text_embed_net
        self.image_embed_net = image_embed_net

        self.id2word, self.word2id, self.max_length, vocab_size = self.prepare(
            dataset_dir
        )

        self.linear_1 = nn.Linear(features, features)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(features, vocab_size)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, image: Tensor, sequence: Tensor) -> Tensor:
        """_summary_

        Args:
            image (Tensor): (batch, c, w, h)
            sequence (Tensor): (batch, max_length)

        Returns:
            Tensor: (batch, vocab_size)
        """
        from IPython import embed

        embed()
        image_embed = self.image_embed_net(image)
        scores, encoded_captions, decode_lengths, alphas, sort_ind = self.text_embed_net(
            image_embed, sequence, torch.full((image.shape[0], 1), self.max_length)
        )
        return scores, encoded_captions, decode_lengths, alphas, sort_ind

    def prepare(self, dataset_dir: str):
        """_summary_

        Args:
            dataset_dir (str): _description_

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        id2word_path = osp.join(dataset_dir, "id2word.pkl")
        word2id_path = osp.join(dataset_dir, "word2id.pkl")
        max_length_path = osp.join(dataset_dir, "max_length.pkl")
        vocab_size_path = osp.join(dataset_dir, "vocab_size.pkl")

        if not osp.exists(vocab_size_path):
            raise ValueError(
                "vocab_size_path is not exist. Please check path or run datamodule to prepare"
            )

        with open(id2word_path, "rb") as file:
            id2word = load(file)

        with open(word2id_path, "rb") as file:
            word2id = load(file)

        with open(max_length_path, "rb") as file:
            max_length = load(file)

        with open(vocab_size_path, "rb") as file:
            vocab_size = load(file)

        return id2word, word2id, max_length, vocab_size

    def greedySearch(self, image: Tensor):
        """_summary_

        Args:
            image (Tensor): _description_

        Returns:
            _type_: _description_
        """
        in_text = "startseq"
        for i in range(self.max_length):
            sequence = [self.word2id[w] for w in in_text.split() if w in self.word2id]
            sequence = torch.nn.functional.pad(
                torch.tensor(sequence), (self.max_length - len(sequence), 0), value=0
            )

            sequence = sequence.unsqueeze(0).to(image.device)

            pred = self(image, sequence)
            print(pred)
            pred = torch.argmax(pred, dim=1)
            word = self.id2word[pred.cpu().item()]
            print(in_text + " " + word)
            in_text += " " + word
            if word == "endseq":
                break
        final = in_text.split()
        final = final[1:-1]
        final = " ".join(final)
        return final


if __name__ == "__main__":
    embed_dim = 256  # 512
    attention_dim = 256  # 512
    decoder_dim = 256  # 512
    vocab_size = 1741

    net = BertCaptioningNet(
        image_embed_net=ResnetEncoder(),
        text_embed_net=DecoderWithAttention(
            attention_dim=attention_dim,
            embed_dim=embed_dim,
            decoder_dim=decoder_dim,
            vocab_size=vocab_size,
        ),
    )

    sequences = torch.randint(0, 100, (2, 37))
    images = torch.randn(2, 3, 299, 299)
    out = net(images, sequences)
    print(out.shape)

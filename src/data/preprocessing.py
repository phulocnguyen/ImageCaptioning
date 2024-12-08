from typing import Optional, List

import torch
import imageio
import numpy as np
import os.path as osp
from pickle import dump, load
from torch.utils.data import Dataset
from torchvision import transforms as T


class PreprocessingDataset(Dataset):
    # glove: wget https://nlp.stanford.edu/data/glove.6B.zip
    glove_dir = 'data/glove'

    def __init__(self, dataset: Dataset = None, dataset_dir: str = None):

        captions = sum([dataset[i][1] for i in range(len(dataset))], [])
        max_length, _, word2id = prepare_dataset(captions,
                                                 dataset_dir,
                                                 glove_dir=self.glove_dir)

        self.preprocessed_dataset = []
        for i in range(len(dataset)):
            img_path, captions = dataset[i]
            caps = []
            for caption in captions:
                caption = [
                    word2id[word] for word in caption.split()
                    if word in word2id
                ]

                caption = caption + [0] * (max_length - len(caption))
                caps.append(caption)

            self.preprocessed_dataset.append([img_path, torch.tensor(caps)])

        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize([299, 299],
                     antialias=True),  # using inception_v3 to encode image
        ])

    def __len__(self):
        return len(self.preprocessed_dataset)

    def __getitem__(self, idx):
        img_path, captions = self.preprocessed_dataset[idx]
        image = imageio.v2.imread(img_path)
        image = self.transform(image)

        return image, captions


def prepare_dataset(captions: List[str],
                    dataset_dir: str,
                    glove_dir: str,
                    word_count_threshold: int = 10,
                    embedding_dim: int = 200) -> None:

    embedding_matrix_path = osp.join(dataset_dir, 'embedding_matrix.pkl')
    id2word_path = osp.join(dataset_dir, 'id2word.pkl')
    word2id_path = osp.join(dataset_dir, 'word2id.pkl')
    max_length_path = osp.join(dataset_dir, 'max_length.pkl')
    vocab_size_path = osp.join(dataset_dir, 'vocab_size.pkl')

    if osp.exists(embedding_matrix_path):
        with open(embedding_matrix_path, "rb") as file:
            embedding_matrix = load(file)

        with open(id2word_path, "rb") as file:
            id2word = load(file)

        with open(word2id_path, "rb") as file:
            word2id = load(file)

        with open(max_length_path, "rb") as file:
            max_length = load(file)

        with open(vocab_size_path, "rb") as file:
            vocab_size = load(file)

    else:
        word_counts = {}  # a dict : { word : number of appearances}
        max_length = 0
        for caption in captions:
            words = caption.split()
            max_length = len(words) if (max_length
                                        < len(words)) else max_length
            for w in words:
                try:
                    word_counts[w] += 1
                except:
                    word_counts[w] = 1

        vocab = ['<pad>'] + [
            w for w in word_counts if word_counts[w] >= word_count_threshold
        ]
        vocab_size = len(vocab)

        id2word, word2id = {}, {}
        for id, word in enumerate(vocab):
            word2id[word] = id
            id2word[id] = word
            id += 1

        file = open(osp.join(glove_dir, 'glove.6B.200d.txt'), encoding="utf-8")
        embeddings_index = {}
        for line in file:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            coefs = torch.from_numpy(coefs)
            embeddings_index[word] = coefs

        embedding_matrix = torch.zeros((len(vocab), embedding_dim))
        for word, i in word2id.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # Words not found in the embedding index will be all zeros
                embedding_matrix[i] = embedding_vector

        # Open a file for writing with binary mode
        with open(embedding_matrix_path, "wb") as file:
            dump(embedding_matrix, file)

        with open(id2word_path, "wb") as file:
            dump(id2word, file)

        with open(word2id_path, "wb") as file:
            dump(word2id, file)

        with open(max_length_path, "wb") as file:
            dump(max_length, file)

        with open(max_length_path, "wb") as file:
            dump(max_length, file)

        with open(vocab_size_path, "wb") as file:
            dump(vocab_size, file)

    print('Embedding matrix:', embedding_matrix.shape)
    print('Max length of caption:', max_length)
    print('Vocab size:', vocab_size)

    return max_length, id2word, word2id


if __name__ == "__main__":
    import rootutils
    rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
    from src.data.dataset import FlickrDataset8k

    dataset = FlickrDataset8k()
    preprocessing_dataset = PreprocessingDataset(dataset,
                                                 dataset_dir='data/flickr8k')
    print('|' * 60)
    print('Length Dataset:', len(dataset))
    print('Length Preprocessing Dataset:', len(preprocessing_dataset))

    image, input, target = preprocessing_dataset[0]
    print('Image shape:', image.shape)
    print('Input:', input)
    print('Input shape:', input.shape)
    print('Target:', target)
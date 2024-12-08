import torch
from torch import Tensor
from src.models.components import ImageCaptionNet


def greedy_search(model: ImageCaptionNet, images: Tensor):
    """_summary_

    Args:
        model (ImageCaptionNet): _description_
        images (Tensor): _description_

    Returns:
        _type_: _description_
    """

    captions = []
    for image in images:
        caption = 'startseq'
        image = image.unsqueeze(0)
        for i in range(model.max_length):
            sequence = [
                model.word2id[w] for w in caption.split() if w in model.word2id
            ]
            sequence = torch.nn.functional.pad(
                torch.tensor(sequence), (model.max_length - len(sequence), 0),
                value=0)

            sequence = sequence.unsqueeze(0).to(image.device)
            pred = model(image, sequence)
            pred = torch.argmax(pred, dim=1)
            word = model.id2word[pred.cpu().item()]
            caption += ' ' + word
            if word == 'endseq':
                break
        caption = caption.split()
        caption = caption[1:-1]
        caption = ' '.join(caption)
        captions.append(caption)
    return captions


def batch_greedy_search(model: ImageCaptionNet, images: Tensor):
    """_summary_

    Args:
        model (ImageCaptionNet): _description_
        images (Tensor): _description_

    Returns:
        _type_: _description_
    """
    sequences = torch.tensor([[model.word2id['startseq']]] *
                             images.shape[0]).to(images.device)
    for i in range(model.max_length - 1):
        seqs_pad = torch.nn.functional.pad(sequences,
                                           (model.max_length - i - 1, 0),
                                           value=0)

        seqs_pad = seqs_pad.to(images.device)
        pred = model(images, seqs_pad)
        pred = torch.argmax(pred, dim=1, keepdim=True)
        sequences = torch.cat((sequences, pred), dim=1)

    captions = []
    for sequence in sequences:
        caption = []
        for id in sequence:
            w = model.id2word[id.cpu().item()]
            if w == 'endseq': break
            caption.append(w)
        captions.append(' '.join(caption[1:]))

    return captions
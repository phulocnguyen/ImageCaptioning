import copy
from heapq import heappush, heappop

import torch
from torch import Tensor
from src.models.components import ImageCaptionNet


class BeamSearchNode(object):

    def __init__(self, prev_node, caption, logp, length):
        # self.h = h
        self.prev_node = prev_node
        self.caption = caption
        self.logp = logp
        self.length = length

    def eval(self):
        return self.logp / float(self.length - 1 + 1e-6)


def beam_search_decoding(model: ImageCaptionNet,
                         images: Tensor,
                         beam_width=10,
                         n_best: int = 5,
                         max_dec_steps: int = 1000):
    """_summary_

    Args:
        model (ImageCaptionNet): _description_
        images (Tensor): _description_
        beam_width (int, optional): _description_. Defaults to 10.
        n_best (int, optional): _description_. Defaults to 5.
        max_dec_steps (int, optional): _description_. Defaults to 1000.

    Returns:
        _type_: _description_
    """
    assert beam_width >= n_best

    n_best_list = []

    # Decoding goes sentence by sentence.
    # So this process is very slow compared to batch decoding process.
    for image in images:
        image = image.unsqueeze(0)

        # Number of sentence to generate
        end_nodes = []

        # starting node
        node = BeamSearchNode(prev_node=None,
                              caption='startseq',
                              logp=0,
                              length=1)

        # whole beam search node graph
        nodes = []

        # Start the queue
        heappush(nodes, (-node.eval(), id(node), node))
        n_dec_steps = 0

        # Start beam search
        while n_dec_steps < max_dec_steps:

            # Fetch the best node
            score, _, best_node = heappop(nodes)

            sequence = [
                model.word2id[w] for w in best_node.caption.split()
                if w in model.word2id
            ]

            if sequence[-1] == model.word2id[
                    'endseq'] and best_node.prev_node is not None:
                end_nodes.append((score, id(best_node), best_node))
                # If we reached maximum # of sentences required
                if len(end_nodes) >= n_best:
                    break
                else:
                    continue

            sequence = torch.nn.functional.pad(
                torch.tensor(sequence), (model.max_length - len(sequence), 0),
                value=0)

            sequence = sequence.unsqueeze(0).to(image.device)

            pred = model(image, sequence)
            pred = torch.nn.functional.log_softmax(pred, dim=1)

            # Get top-k
            topk_log_prob, topk_indexes = torch.topk(pred, beam_width)

            # Then, register new top-k nodes
            for new_k in range(beam_width):
                next_word = model.id2word[topk_indexes[0]
                                          [new_k].cpu().item()]  # (1)
                logp = topk_log_prob[0][new_k].item()

                node = BeamSearchNode(prev_node=best_node,
                                      caption=best_node.caption + ' ' +
                                      next_word,
                                      logp=best_node.logp + logp,
                                      length=best_node.length + 1)
                heappush(nodes, (-node.eval(), id(node), node))
            n_dec_steps += beam_width

        # if there are no end_nodes, retrieve best nodes (they are probably truncated)
        if len(end_nodes) == 0:
            end_nodes = [heappop(nodes) for _ in range(beam_width)]

        # Construct sequences from end_nodes
        n_best_seq_list = []
        for score, _id, node in end_nodes:
            caption = node.caption.split()
            caption = caption[1:-1] if caption[-1] == 'endseq' else caption[1:]
            n_best_seq_list.append(' '.join(caption))

        captions = ' | '.join(n_best_seq_list)
        n_best_list.append(captions)

    return n_best_list


# def batch_beam_search_decoding(model: ImageCaptionNet,
#                                images: Tensor,
#                                beam_width: int = 10,
#                                n_best: int = 5,
#                                max_dec_steps: int = 1000):
#     """_summary_

#     Args:
#         model (ImageCaptionNet): _description_
#         images (Tensor): _description_
#         beam_width (int, optional): _description_. Defaults to 10.
#         n_best (int, optional): _description_. Defaults to 5.
#         max_dec_steps (int, optional): _description_. Defaults to 1000.

#     Returns:
#         _type_: _description_
#     """

#     assert beam_width >= n_best

#     n_best_list = []

#     # Number of sentence to generate
#     end_nodes_list = [[] for _ in range(images.shape[0])]

#     # whole beam search node graph
#     nodes = [[] for _ in range(images.shape[0])]

#     # Start the queue
#     for bid in range(images.shape[0]):
#         # starting node
#         node = BeamSearchNode(prev_node=None,
#                               caption='startseq',
#                               logp=0,
#                               length=1)
#         heappush(nodes[bid], (-node.eval(), id(node), node))

#     # Start beam search
#     fin_nodes = set()
#     history = [None for _ in range(images.shape[0])]
#     n_dec_steps_list = [0 for _ in range(images.shape[0])]
#     while len(fin_nodes) < images.shape[0]:
#         # Fetch the best node
#         decoder_input, decoder_hidden = [], []
#         for bid in range(images.shape[0]):
#             if bid not in fin_nodes and n_dec_steps_list[bid] > max_dec_steps:
#                 fin_nodes.add(bid)

#             if bid in fin_nodes:
#                 score, n = history[bid]  # dummy for data consistency
#             else:
#                 score, _, n = heappop(nodes[bid])
#                 if n.wid.item() == eos_token and n.prev_node is not None:
#                     end_nodes_list[bid].append((score, id(n), n))
#                     # If we reached maximum # of sentences required
#                     if len(end_nodes_list[bid]) >= n_best:
#                         fin_nodes.add(bid)
#                 history[bid] = (score, n)
#             decoder_input.append(n.wid)
#             decoder_hidden.append(n.h)

#         decoder_input = torch.cat(decoder_input).to(device)  # (bs)
#         decoder_hidden = torch.stack(decoder_hidden, 0).to(device)  # (bs, H)

#         # Decode for one step using decoder
#         decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden,
#                                                  enc_outs)  # (bs, V), (bs, H)

#         # Get top-k from this decoded result
#         topk_log_prob, topk_indexes = torch.topk(
#             decoder_output, beam_width)  # (bs, bw), (bs, bw)
#         # Then, register new top-k nodes
#         for bid in range(bs):
#             if bid in fin_nodes:
#                 continue
#             score, n = history[bid]
#             if n.wid.item() == eos_token and n.prev_node is not None:
#                 continue
#             for new_k in range(beam_width):
#                 decoded_t = topk_indexes[bid][new_k].view(1)  # (1)
#                 logp = topk_log_prob[bid][new_k].item(
#                 )  # float log probability val

#                 node = BeamSearchNode(h=decoder_hidden[bid],
#                                       prev_node=n,
#                                       wid=decoded_t,
#                                       logp=n.logp + logp,
#                                       length=n.length + 1)
#                 heappush(nodes[bid], (-node.eval(), id(node), node))
#             n_dec_steps_list[bid] += beam_width

#     # Construct sequences from end_nodes
#     # if there are no end_nodes, retrieve best nodes (they are probably truncated)
#     for bid in range(bs):
#         if len(end_nodes_list[bid]) == 0:
#             end_nodes_list[bid] = [
#                 heappop(nodes[bid]) for _ in range(beam_width)
#             ]

#         n_best_seq_list = []
#         for score, _id, n in sorted(end_nodes_list[bid], key=lambda x: x[0]):
#             sequence = [n.wid.item()]
#             while n.prev_node is not None:
#                 n = n.prev_node
#                 sequence.append(n.wid.item())
#             sequence = sequence[::-1]  # reverse

#             n_best_seq_list.append(sequence)

#         n_best_list.append(copy.copy(n_best_seq_list))

#     return n_best_list

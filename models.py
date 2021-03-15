"""Top-level model classes.

Author:
    Chris Chute (chute@stanford.edu)
"""

import layers
import torch
import torch.nn as nn


class BiDAF(nn.Module):
    """Baseline BiDAF model for SQuAD.

    Based on the paper:
    "Bidirectional Attention Flow for Machine Comprehension"
    by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
    (https://arxiv.org/abs/1611.01603).

    Follows a high-level structure commonly found in SQuAD models:
        - Embedding layer: Embed word indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, word_vectors, char_vectors, hidden_size, drop_prob=0.):
        super(BiDAF, self).__init__()
        self.emb = layers.Embedding(word_vectors=word_vectors,
                                    char_vectors = char_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)
        # new_hidden_size = 2 * hidden_size
        # self.enc = layers.RNNEncoder(input_size=new_hidden_size,
        #                              hidden_size=hidden_size,
        #                              num_layers=1,
        #                              drop_prob=drop_prob)
        self.enc = layers.EmbeddingEncoder(nconvs = 4, 
                                           input_size = hidden_size, 
                                           output_size = hidden_size,
                                           k = 7,
                                           drop_prob=drop_prob,
                                           num_blocks=1)
        # hidden_size = int(128 / 2)
        self.att = layers.BiDAFAttention(hidden_size=hidden_size,
                                         drop_prob=drop_prob)

        # self.mod = layers.RNNEncoder(input_size=8  * hidden_size, # changed from 8 to 6
        #                              hidden_size=hidden_size,
        #                              num_layers=2,
        #                              drop_prob=drop_prob)

        self.mod = layers.ModelEncoder(nconvs=2, 
                                       input_size = 4*hidden_size, 
                                       num_blocks=7, 
                                       k = 7, 
                                       output_size = hidden_size,
                                       drop_prob=drop_prob)

        # self.out = layers.BiDAFOutput(hidden_size=hidden_size,
        #                               drop_prob=drop_prob)
        self.out = layers.QANetOutput(input_size=hidden_size, 
                                      drop_prob=drop_prob)

    # def forward(self, cw_idxs, qw_idxs):
    #     c_mask = torch.zeros_like(cw_idxs) != cw_idxs
    #     q_mask = torch.zeros_like(qw_idxs) != qw_idxs
    #     c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

    #     c_emb = self.emb(cw_idxs)         # (batch_size, c_len, hidden_size)
    #     q_emb = self.emb(qw_idxs)         # (batch_size, q_len, hidden_size)

    #     c_enc = self.enc(c_emb, c_len)    # (batch_size, c_len, 2 * hidden_size)
    #     q_enc = self.enc(q_emb, q_len)    # (batch_size, q_len, 2 * hidden_size)

    #     att = self.att(c_enc, q_enc,
    #                    c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)

    #     mod = self.mod(att, c_len)        # (batch_size, c_len, 2 * hidden_size)

    #     out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

    #     return out

    def forward(self, cw_idxs, qw_idxs, cc_idxs, qc_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)
        # print(q_mask)
        # print(q_len)

        c_emb = self.emb(cw_idxs, cc_idxs)         # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs, qc_idxs)         # (batch_size, q_len, hidden_size)

        # c_enc = self.enc(c_emb, c_len)    # (batch_size, c_len, 2 * hidden_size)
        # q_enc = self.enc(q_emb, q_len)    # (batch_size, q_len, 2 * hidden_size)
        c_enc = self.enc(c_emb, c_mask)
        q_enc = self.enc(q_emb, q_mask)

        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)

        # mod = self.mod(att, c_len)        # (batch_size, c_len, 2 * hidden_size)
        ME1, ME2, ME3 = self.mod(att, c_mask)

        # out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)
        out = self.out( ME1, ME2, ME3, c_mask)

        return out

"""Assortment of layers for use in models.py.

Author:
    Chris Chute (chute@stanford.edu)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from util import masked_softmax


class Embedding1(nn.Module):
    """Embedding layer used by BiDAF, without the character-level component.

    Word-level embeddings are further refined using a 2-layer Highway Encoder
    (see `HighwayEncoder` class for details).

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations
    """
    def __init__(self, word_vectors, char_vectors, hidden_size, drop_prob):
        super(Embedding, self).__init__()
        self.drop_prob = drop_prob
        self.embed = nn.Embedding.from_pretrained(word_vectors)
        # self.char_embed = nn.Embedding.from_pretrained(char_vectors) # (batch_size, seq_len, char_limit, ch_embed_size) 
        self.char_embed = nn.Embedding(num_embeddings=char_vectors.size(0), embedding_dim=char_vectors.size(1))
        # self.proj = nn.Linear(word_vectors.size(1), hidden_size, bias=False)

        self.out_channels = 200
        # self.cnn1 = nn.Conv1d(in_channels =char_vectors.shape[1], out_channels = self.out_channels, kernel_size = 5, stride = 1)
        self.cnn1 = nn.Conv1d(in_channels =char_vectors.size(1), out_channels = self.out_channels, kernel_size = 5, stride = 1)
        self.proj = nn.Sequential(nn.Linear(word_vectors.size(1) + self.out_channels, hidden_size, bias=False), nn.ReLU())

        self.hwy = HighwayEncoder(2, hidden_size)
    # def forward(self, x):
    #     emb = self.embed(x)   # (batch_size, seq_len, embed_size)
    #     emb = F.dropout(emb, self.drop_prob, self.training)
    #     emb = self.proj(emb)  # (batch_size, seq_len, hidden_size)
    #     emb = self.hwy(emb)   # (batch_size, seq_len, hidden_size)

    #     return emb

    def forward(self, x, char):
        # print('x.shape')
        # print(x.shape)
        # print('char.shape')
        # print(char.shape)
        # x ([64, 341])
        # char ([64, 341, 16])
        # emb ([64, 341, 300])
        emb = self.embed(x)   # (batch_size, seq_len, embed_size)


        char_emb = self.char_embed(char) # (batch_size, seq_len, char_limit, ch_embed_size) ([64, 341, 16, 64])
        batch_size, seq_len, char_limit, ch_embed_size = char_emb.shape
        char_emb = char_emb.view(batch_size * seq_len, char_limit, ch_embed_size) # cnn takes 3 dimensional input
        char_emb = char_emb.permute(0, 2, 1) # make ch_embed_size before char_limit to fit cnn
        # print(char_emb.shape) # ([20864, 64, 16])
        char_emb = self.cnn1(char_emb)
        # print(char_emb)
        char_emb = F.relu(char_emb)
        char_emb = torch.max(char_emb, dim = -1)[0]
        char_emb = char_emb.view(batch_size, seq_len, self.out_channels)
        # print(char_emb.shape)
        # print(emb.shape)

        total_emb = torch.cat([emb, char_emb], dim = -1)
        total_emb = F.dropout(total_emb, self.drop_prob, self.training)
        total_emb = self.proj(total_emb)  # (batch_size, seq_len, hidden_size)

        total_emb = self.hwy(total_emb)   # (batch_size, seq_len, hidden_size)

        return total_emb

class Embedding(nn.Module):
    """Embedding layer used by BiDAF, without the character-level component.
    Word-level embeddings are further refined using a 2-layer Highway Encoder
    (see `HighwayEncoder` class for details).
    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations
    """
    def __init__(self, word_vectors, char_vectors, hidden_size, drop_prob):
        super(Embedding, self).__init__()
        self.drop_prob = drop_prob
        self.embed = nn.Embedding.from_pretrained(word_vectors)
        self.char_embed = nn.Embedding.from_pretrained(char_vectors) # (batch_size, seq_len, char_limit, ch_embed_size) 
        self.proj = nn.Linear(word_vectors.size(1), hidden_size, bias=False)
        self.out_channels = 100
        self.cnn1 = nn.Conv1d(in_channels =char_vectors.shape[1], out_channels = self.out_channels, kernel_size = 5, stride = 1)
        self.hwy = HighwayEncoder(2, hidden_size + self.out_channels)
        self.proj_out = nn.Sequential(nn.Linear(hidden_size + self.out_channels, hidden_size, bias=False), nn.ReLU())


    def forward(self, x, char):
        # x ([64, 341])
        # char ([64, 341, 16])
        # emb ([64, 341, 300])
        emb = self.embed(x)   # (batch_size, seq_len, embed_size)
        emb = F.dropout(emb, self.drop_prob, self.training)
        emb = self.proj(emb)  # (batch_size, seq_len, hidden_size)
        char_emb = self.char_embed(char) # (batch_size, seq_len, char_limit, ch_embed_size) ([64, 341, 16, 64])
        batch_size, seq_len, char_limit, ch_embed_size = char_emb.shape
        char_emb = char_emb.view(batch_size * seq_len, char_limit, ch_embed_size) # cnn takes 3 dimensional input
        char_emb = char_emb.permute(0, 2, 1) # make ch_embed_size before char_limit to fit cnn
        # print(char_emb.shape) # ([20864, 64, 16])
        char_emb = self.cnn1(char_emb)
        # print(char_emb)
        char_emb = F.relu(char_emb)
        char_emb = torch.max(char_emb, dim = -1)[0]
        char_emb = char_emb.view(batch_size, seq_len, self.out_channels)
        # print(char_emb.shape)
        # print(emb.shape)
        total_emb = torch.cat([emb, char_emb], dim = -1)
        total_emb = self.hwy(total_emb)   # (batch_size, seq_len, hidden_size)
        total_emb = self.proj_out(total_emb)
        return total_emb

class HighwayEncoder(nn.Module):
    """Encode an input sequence using a highway network.

    Based on the paper:
    "Highway Networks"
    by Rupesh Kumar Srivastava, Klaus Greff, Jürgen Schmidhuber
    (https://arxiv.org/abs/1505.00387).

    Args:
        num_layers (int): Number of layers in the highway encoder.
        hidden_size (int): Size of hidden activations.
    """
    def __init__(self, num_layers, hidden_size):
        super(HighwayEncoder, self).__init__()
        self.transforms = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                         for _ in range(num_layers)])
        self.gates = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                    for _ in range(num_layers)])

    def forward(self, x):
        for gate, transform in zip(self.gates, self.transforms):
            # Shapes of g, t, and x are all (batch_size, seq_len, hidden_size)
            g = torch.sigmoid(gate(x))
            t = F.relu(transform(x))
            x = g * t + (1 - g) * x

        return x


# ----------------QANet-------------------

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, input_size, output_size, k):
        super(DepthwiseSeparableConv, self).__init__()
        # At groups= in_channels, each input channel is convolved with its own set of filters
        self.depthwise_conv = nn.Conv1d(in_channels=input_size, out_channels=input_size, kernel_size=k, groups=input_size, padding = k//2)
        self.pointwise_conv = nn.Conv1d(in_channels=input_size, out_channels=output_size, kernel_size=1)
    def forward(self, x):
        conv_output = self.depthwise_conv(x) # apps, apps, in_ch
        conv_output = self.pointwise_conv(conv_output) # apps, apps, out_ch
        return F.relu(conv_output)

class SelfAttention(nn.Module):
    def __init__(self, n_heads, n_embd, drop_prob):
        super(SelfAttention, self).__init__()
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(drop_prob)
        self.resid_drop = nn.Dropout(drop_prob)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        # self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
        #                              .view(1, 1, config.block_size, config.block_size))
        self.n_head = n_heads
    def forward(self, x):
        # print(x.size())
        B, T, C = x.size() # (B, l, d)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # self.key(x) (B, l, d), make (B, l, h, d/h) transpose (B, h, l, d/h) l is block size
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.mask[:,:,:T,:T] == 0, -1e10)
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y




class EmbeddingEncoder(nn.Module):
    def __init__(self,
                 nconvs,
                 input_size,
                 output_size,
                 k,
                 drop_prob,
                 num_blocks):
        super(EmbeddingEncoder, self).__init__()
        self.num_blocks = num_blocks
        self.conv_num = nconvs
        self.drop_prob = drop_prob
        self.pos_enc = PositionalEncoding(input_size)
        # self.cnn1 = nn.Conv1d(in_channels =input_size, out_channels = output_size, kernel_size = 5, stride = 1, padding = 2)
        # self.conv_layers = nn.ModuleList([DepthwiseSeparableConv(input_size, output_size, k)] + \
        #     [DepthwiseSeparableConv(output_size, output_size, k) for i in range(nconvs - 1)]) # (bsz, seq_len, output_sz 128)
        self.conv_layers = nn.ModuleList([DepthwiseSeparableConv(output_size, output_size, k) for i in range(nconvs)])
        self.norm_layers = nn.ModuleList([nn.LayerNorm(output_size) for i in range(nconvs+2)])
        # self.norm_layers = nn.ModuleList([nn.LayerNorm(input_size)] + \
        #     [nn.LayerNorm(output_size) for i in range(nconvs+1)])
        self.attn = SelfAttention(n_heads = 4, n_embd = output_size, drop_prob = self.drop_prob) # (bsz, seq_len, output_sz)
        self.ffn = nn.Conv1d(in_channels=output_size, out_channels=output_size, kernel_size=1) # cnn instead of ffn


    # def layer_dropout(self, inputs, residual, dropout):
    #     i = np.random.uniform()
    #     if i > self.drop_prob:
    #         return F.dropout(inputs, dropout, self.training) + residual
    #     else:
    #         return residual
    def layer_dropout(self, inputs, residual, dropout):
        if self.training == True:
            pred = torch.empty(1).uniform_(0,1) < dropout
            if pred:
                return residual
            else:
                return F.dropout(inputs, dropout, training=self.training) + residual
        else:
            return inputs + residual

    def res_block(self, inputs, residual):
            return residual + inputs


    # def forward(self, x, start, total_layers):
    def forward(self, x):
        # print("rrrrr")
        total_layers = (self.conv_num+1)*self.num_blocks
        batch_size, seq_len, embed_size = x.shape
        out = self.pos_enc(x)
        out = x.permute(0, 2, 1) # make ch_embed_size before char_limit to fit cnn
        # print(out.shape)
        # out = self.cnn1(out)
        # print(out.shape)
        # print("lllllll")
        for i, conv_layer in enumerate(self.conv_layers):
            res = out
            out = self.norm_layers[i](out.transpose(1,2)).transpose(1,2)
            # out = self.norm_layers[i](out)
            out = conv_layer(out)
            # print(out.shape)
            # out = self.layer_dropout(out, res, self.drop_prob * start / total_layers)
            out = self.layer_dropout(out, res, self.drop_prob * (i + 1)/total_layers)
            # start += 1
        res = out.transpose(1,2)
        out = self.norm_layers[-2](out.transpose(1,2))
        out = self.attn(out)
        # out = self.layer_dropout(out, res, self.drop_prob * start / total_layers)
        out = self.res_block(out, res)
        # start += 1
        res = out
        out = self.norm_layers[-1](out)
        out = self.ffn(out.permute(0, 2, 1)).permute(0, 2, 1)
        # print('2')
        # print(out.shape)
        # out = self.layer_dropout(out, res, self.drop_prob * start / total_layers)
        out = self.res_block(out, res)
        # print('3')
        # print(out.shape)
        return out


class ModelEncoder(nn.Module):
    def __init__(self,
                 nconvs,
                 input_size,
                 output_size,
                 num_blocks,
                 k,
                 drop_prob):
        super(ModelEncoder, self).__init__()
        # self.cnn1 = nn.Conv1d(in_channels =input_size, out_channels = 128, kernel_size = 1, stride = 1, bias = False)
        self.proj = nn.Linear(input_size, output_size, bias=False)
        self.encoder_blocks = nn.ModuleList([EmbeddingEncoder(nconvs = nconvs, input_size = 128, k = 7, drop_prob = drop_prob, output_size = output_size, num_blocks=num_blocks) for i in range(num_blocks)])

    # def forward(self, x, start, total_layers):
    def forward(self, x):

        # out = self.cnn1(x.permute(0, 2, 1)).permute(0, 2, 1)
        ME1 = self.proj(x)
        for block in self.encoder_blocks:
            ME1 = block(ME1)
        ME2 = ME1
        for block in self.encoder_blocks:
            ME2 = block(ME2)
        ME3 = ME2
        for block in self.encoder_blocks:
            ME3 = block(ME3)
        return ME1, ME2, ME3

class QANetOutput(nn.Module):
    def __init__(self, input_size, drop_prob):
        super(QANetOutput, self).__init__()
        self.lin1 = nn.Linear(input_size * 2, 1)
        self.lin2 = nn.Linear(input_size * 2, 1)
    
    def forward(self, ME1, ME2, ME3, mask):
        output1 = torch.cat([ME1, ME2], dim=2)
        output1 = self.lin1(output1).squeeze()
        p1 = masked_softmax(output1, mask, log_softmax=True)
        output2 = torch.cat([ME1, ME3], dim=2)
        output2 = self.lin2(output2).squeeze()
        p2 = masked_softmax(output2, mask, log_softmax=True)
        return p1, p2

# ----------------QANet-------------------
class RNNEncoder(nn.Module):
    """General-purpose layer for encoding a sequence using a bidirectional RNN.

    Encoded output is the RNN's hidden state at each position, which
    has shape `(batch_size, seq_len, hidden_size * 2)`.

    Args:
        input_size (int): Size of a single timestep in the input.
        hidden_size (int): Size of the RNN hidden state.
        num_layers (int): Number of layers of RNN cells to use.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 drop_prob=0.):
        super(RNNEncoder, self).__init__()
        self.drop_prob = drop_prob
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True,
                           bidirectional=True,
                           dropout=drop_prob if num_layers > 1 else 0.)

    def forward(self, x, lengths):
        # Save original padded length for use by pad_packed_sequence
        orig_len = x.size(1)

        # Sort by length and pack sequence for RNN
        lengths, sort_idx = lengths.sort(0, descending=True) # ????????
        x = x[sort_idx]     # (batch_size, seq_len, input_size)
        x = pack_padded_sequence(x, lengths, batch_first=True)

        # Apply RNN
        x, _ = self.rnn(x)  # (batch_size, seq_len, 2 * hidden_size)

        # Unpack and reverse sort
        x, _ = pad_packed_sequence(x, batch_first=True, total_length=orig_len)
        _, unsort_idx = sort_idx.sort(0)
        x = x[unsort_idx]   # (batch_size, seq_len, 2 * hidden_size)

        # Apply dropout (RNN applies dropout after all but the last layer)
        x = F.dropout(x, self.drop_prob, self.training)

        return x


class BiDAFAttention(nn.Module):
    """Bidirectional attention originally used by BiDAF.

    Bidirectional attention computes attention in two directions:
    The context attends to the query and the query attends to the context.
    The output of this layer is the concatenation of [context, c2q_attention,
    context * c2q_attention, context * q2c_attention]. This concatenation allows
    the attention vector at each timestep, along with the embeddings from
    previous layers, to flow through the attention layer to the modeling layer.
    The output has shape (batch_size, context_len, 8 * hidden_size).

    Args:
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob=0.1):
        super(BiDAFAttention, self).__init__()
        self.drop_prob = drop_prob
        self.c_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.q_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.cq_weight = nn.Parameter(torch.zeros(1, 1, hidden_size))
        for weight in (self.c_weight, self.q_weight, self.cq_weight):
            nn.init.xavier_uniform_(weight)
        self.bias = nn.Parameter(torch.zeros(1))


    def forward(self, c, q, c_mask, q_mask):
        batch_size, c_len, _ = c.size()
        q_len = q.size(1)
        s = self.get_similarity_matrix(c, q)        # (batch_size, c_len, q_len)
        c_mask = c_mask.view(batch_size, c_len, 1)  # (batch_size, c_len, 1)
        q_mask = q_mask.view(batch_size, 1, q_len)  # (batch_size, 1, q_len)
        s1 = masked_softmax(s, q_mask, dim=2)       # (batch_size, c_len, q_len)
        s2 = masked_softmax(s, c_mask, dim=1)       # (batch_size, c_len, q_len)

        # (bs, c_len, q_len) x (bs, q_len, hid_size) => (bs, c_len, hid_size)
        a = torch.bmm(s1, q)
        # (bs, c_len, c_len) x (bs, c_len, hid_size) => (bs, c_len, hid_size)
        b = torch.bmm(torch.bmm(s1, s2.transpose(1, 2)), c)

        x = torch.cat([c, a, c * a, c * b], dim=2)  # (bs, c_len, 4 * hid_size)

        return x


    def get_similarity_matrix(self, c, q):
        """Get the "similarity matrix" between context and query (using the
        terminology of the BiDAF paper).

        A naive implementation as described in BiDAF would concatenate the
        three vectors then project the result with a single weight matrix. This
        method is a more memory-efficient implementation of the same operation.

        See Also:
            Equation 1 in https://arxiv.org/abs/1611.01603
        """
        c_len, q_len = c.size(1), q.size(1)
        c = F.dropout(c, self.drop_prob, self.training)  # (bs, c_len, hid_size)
        q = F.dropout(q, self.drop_prob, self.training)  # (bs, q_len, hid_size)

        # Shapes: (batch_size, c_len, q_len)
        s0 = torch.matmul(c, self.c_weight).expand([-1, -1, q_len])
        s1 = torch.matmul(q, self.q_weight).transpose(1, 2)\
                                           .expand([-1, c_len, -1])
        s2 = torch.matmul(c * self.cq_weight, q.transpose(1, 2))
        s = s0 + s1 + s2 + self.bias

        return s


class BiDAFOutput(nn.Module):
    """Output layer used by BiDAF for question answering.

    Computes a linear transformation of the attention and modeling
    outputs, then takes the softmax of the result to get the start pointer.
    A bidirectional LSTM is then applied the modeling output to produce `mod_2`.
    A second linear+softmax of the attention output and `mod_2` is used
    to get the end pointer.

    Args:
        hidden_size (int): Hidden size used in the BiDAF model.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob):
        super(BiDAFOutput, self).__init__()
        self.att_linear_1 = nn.Linear(8 * hidden_size, 1) # changed from 8 to 6
        self.mod_linear_1 = nn.Linear(2 * hidden_size, 1)

        self.rnn = RNNEncoder(input_size=2 * hidden_size,
                              hidden_size=hidden_size,
                              num_layers=1,
                              drop_prob=drop_prob)

        self.att_linear_2 = nn.Linear(8 * hidden_size, 1) # changed from 8 to 6
        self.mod_linear_2 = nn.Linear(2 * hidden_size, 1)

    def forward(self, att, mod, mask):
        # Shapes: (batch_size, seq_len, 1)
        logits_1 = self.att_linear_1(att) + self.mod_linear_1(mod)
        mod_2 = self.rnn(mod, mask.sum(-1))
        logits_2 = self.att_linear_2(att) + self.mod_linear_2(mod_2)

        # Shapes: (batch_size, seq_len)
        log_p1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(), mask, log_softmax=True)

        return log_p1, log_p2

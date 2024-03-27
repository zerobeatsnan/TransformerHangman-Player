import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.nn import L1Loss
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OneHotEncoder
import itertools
import random
from tqdm import trange
from torch.nn.utils.rnn import pad_sequence
import math

class InputEmbedding(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        """
        d_model: the higher dimension the char will be mapped to
        vocab_size: 27 in hangman case, since we need to think about "_"
        """
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):

        return self.embedding(x) * math.sqrt(self.d_model) # implement normalization according to d_model
    

class PositionEmbedding(nn.Module):

    def __init__(self, d_model: int, dropout: float, max_len = 1000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.dropout = nn.Dropout(dropout)

        # create a matrix without updating
        pe = torch.zeros(max_len, d_model)
        # (max_len, 1)
        position = torch.arange(0, max_len, dtype = torch.float).unsqueeze(1) 
        # change the position to logspace
        div_term = torch.exp(torch.arange(0,self.d_model, 2).float() * (-math.log(10000.0)
                                                                        /self.d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) # (1, max_len, d_model)

        self.register_buffer("pe", pe)

    def forward(self,x):
        
        x = x + (self.pe[:, x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


class LayerNormalization(nn.Module):

    def __init__(self, eps: float = 10**-6):

        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self,x):

        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim = True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardSec(nn.Module):

    def __init__(self, d_model, d_ff, dropout):

        super().__init__()
        self.fc_1 = nn.Linear(d_model, d_ff) # mat1 and b1
        self.dropout = nn.Dropout(dropout)
        self.fc_2 = nn.Linear(d_ff, d_model) # mat2 and b2
        self.relu = nn.ReLU()
        
    def forward(self, x):

        # (Batch, seq_len, d_model) as x
        output = self.fc_1(x)
        # output = self.relu(output)
        output = F.gelu(output)
        output = self.dropout(output)
        res = self.fc_2(output)
        return res
    

class MultiHeadAttentionSec(nn.Module):

    def __init__(self, d_model, h, dropout):
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0

        self.d_k = d_model // h
        # we define the query/key/value matrix as a whole
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):

        d_k = query.shape[-1]

        attention_scores = (query @ key.transpose(-2,-1) / math.sqrt(d_k))
        if mask is not None:

            attention_scores.masked_fill_(mask == 0, -1e9)

        attention_scores = attention_scores.softmax(dim = -1)

        if dropout is not None:

            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores



    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        key = self.w_k(k)   # (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        value = self.w_v(v) # (batch, seq_len, d_model) -> (batch, seq_len, d_model)

        #(Batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)

        x, self.attention_scores = MultiHeadAttentionSec.attention(query, key, value, mask, self.dropout)

        # (Batch, h, seq_len, d_k) -> (batch, seq_len, h, d_k) -> (batch, seq_len, d_model)
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        return self.w_o(x)
    
class SkipConnection(nn.Module):

    def __init__(self, dropout: float):

        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self,x,sublayer):

        return x + self.dropout(sublayer(self.norm(x)))
    
class EncoderSec(nn.Module):

    def __init__(self, self_attention, feed_forward_sec, dropout):
        super(EncoderSec,self).__init__()
        self.self_attention = self_attention
        self.feed_forward_sec = feed_forward_sec
        self.skip_connection = nn.ModuleList([SkipConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):

        x = self.skip_connection[0](x, lambda x: self.self_attention(x,x,x,src_mask))
    
        x = self.skip_connection[1](x, self.feed_forward_sec)

        return x
    
class Encoder(nn.Module):

    def __init__(self, layers):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self,x,mask):

        for layer in self.layers:

            x = layer(x, mask)

        return self.norm(x)



    
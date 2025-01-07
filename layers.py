from torch import nn
import torch
import numpy as np
from utils import *

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, pdrop=0.1, max_len=256):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=pdrop)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h, causal=False, pdrop=0.1):
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.causal = causal
        
        # Reduce number of operations by combining the multi-head calulations into one
        # For these three, think of this:
        # we have "h" heads, each needs a W_Q, W_K, W_V that outputs d_model/h
        # So, we can combine all of these into one matrix multiplication
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(pdrop)

    def scaled_dot_product_attention(self, queries, keys, values, mask=None):
        print(f"Multiplying Q: {queries.shape}, K: {keys.shape}, with values {values.shape}")
        factor = torch.sqrt(torch.tensor(queries.shape[-2])).to(queries.device)

        weights = torch.matmul(queries, keys.transpose(-2, -1)) / factor

        if mask is not None:
            weights = weights.masked_fill_(mask == 0, -1e9)

        weights = torch.softmax(weights, dim=-1)

        return torch.matmul(weights, values), weights

    # from d2l.ai, allows us to compute multiple heads in parallel
    def transpose_for_attention(self, x):
        x = x.reshape(x.shape[0], x.shape[1], self.h, self.d_model//self.h)
        x = x.permute(0, 2, 1, 3)
        return x.reshape(-1, x.shape[2], x.shape[3])
    
    def get_last_attention_weights(self):
        return self.last_attention_weights

    def forward(self, queries, keys, values, mask=None):
        print(f"Queries: {queries.shape}, Keys: {keys.shape}, Values: {values.shape}")
        Q = self.W_Q(queries)
        K = self.W_K(keys)
        V = self.W_V(values)

        Q = self.transpose_for_attention(Q)
        K = self.transpose_for_attention(K)
        V = self.transpose_for_attention(V)

        attention, weights = self.scaled_dot_product_attention(Q, K, V, mask)

        # keep last attention weights for visualization
        self.last_attention_weights = weights.reshape(-1, self.h, weights.shape[1], weights.shape[2])

        attention = attention.reshape(-1, self.h, attention.shape[1], attention.shape[2])
        attention = attention.permute(0, 2, 1, 3).reshape(attention.shape[0], attention.shape[2], -1)

        return self.W_O(attention)


class FeedForwardLayer(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.feed_forward(x)
        x = self.layer_norm(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, h, p_dropout):
        super().__init__()
        self.attention_layer = MultiHeadAttention(d_model, h)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p_dropout)

        self.feed_forward_layer = FeedForwardLayer(d_model, d_ff)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p_dropout)

    def forward(self, x):
        x1 = self.attention_layer(x, x, x)
        x2 = self.layer_norm1(x1) + x
        x2 = self.dropout1(x2)

        x1 = self.feed_forward_layer(x2)
        x2 = self.layer_norm2(x1) + x2
        x2 = self.dropout2(x2)

        return x2
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, h, p_dropout):
        super().__init__()
        self.causal_attention_layer = MultiHeadAttention(d_model, h)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p_dropout)

        self.cross_attention_layer = MultiHeadAttention(d_model, h)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p_dropout)

        self.feed_forward_layer = FeedForwardLayer(d_model, d_ff)
        self.layer_norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(p_dropout)
    
    def get_cross_attention_weights(self):
        return self.cross_attention_layer.get_last_attention_weights()
    
    
    def get_self_attention_weights(self):
        return self.causal_attention_layer.get_last_attention_weights()
    
    def forward(self, input, context):
        mask = generate_causal_mask(input.shape[1])

        x1 = self.causal_attention_layer(input, input, input, mask)
        x2 = self.layer_norm1(x1) + input
        x2 = self.dropout1(x2)

        x1 = self.cross_attention_layer(x2, context, context)
        x2 = self.layer_norm2(x1) + x2
        x2 = self.dropout2(x2)

        x1 = self.feed_forward_layer(x2)
        x2 = self.layer_norm3(x1) + x2
        x2 = self.dropout3(x2)

        return x2

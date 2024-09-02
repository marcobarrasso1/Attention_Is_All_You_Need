"""
Definition of the transduction model proposed in the attention is all you need paper:
https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
"""

import torch as th
import torch.nn as nn
from torch.nn import functional as F

# sinusoid positional encoding as described in the paper
class PositionalEncoding(nn.Module):
    
    def __init__(self, config):
        super(PositionalEncoding,self).__init__()
        pos = th.arange(0, config.len_seq, dtype = th.float).unsqueeze(1)

        frequency = th.pow(10000,-th.arange(0,config.n_embed,2,dtype = th.float)/config.n_embed)
        # the positional encoding matrix shape is (len_seq, n_embed).
        pe = th.zeros((config.len_seq, config.n_embed))
        pe[:,0::2] = th.sin(pos * frequency)
        pe[:,1::2] = th.cos(pos * frequency)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # we slice this matrix to match the input sequence length, 
        # is different from len_seq only during inference
        seq_len = x.size(1)
        # the resulting matrix is then unsqueezed to shape (1, seq_len, n_embed),
        # in this way the positional encodings will be broadcasted across the batch dimension
        pe_slice = self.pe[:seq_len, :].unsqueeze(0) 
        return pe_slice
    
    
class EmbeddingLayer(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.tok_embedding = nn.Embedding(config.vocab_size, config.n_embed)
        self.pos_embedding = PositionalEncoding(config) 
        self.dropout = nn.Dropout(config.dropout)
        self._ = th.sqrt(th.tensor(config.n_embed))
        
    def forward(self, x):
        tok_embed = self.tok_embedding(x) * self._
        pos_embed = self.pos_embedding(x)
        embed = self.dropout(tok_embed + pos_embed) # the positional encoding are added to the input embeddings 
        return embed


class MultiHeadAttention(nn.Module):
    
    def __init__(self, config, self_mask=None):
        super().__init__()
        self.head_size = config.n_embed // config.n_head
        self.mask = self_mask
        self.n_head = config.n_head
        
        self.key = nn.Linear(config.n_embed, config.n_embed)
        self.query = nn.Linear(config.n_embed, config.n_embed)
        self.value = nn.Linear(config.n_embed, config.n_embed)
        self.linear_out = nn.Linear(config.n_embed, config.n_embed) # output projection
        self.dropout = nn.Dropout(config.dropout)
        
        # causal mask to ensure that attention is only applied to the left in the input sequence
        if self_mask is not None:
            self.register_buffer("bias", (th.tril(th.ones(config.len_seq, config.len_seq)).view(1, 1, config.len_seq, config.len_seq).bool()))
        
    def forward(self, k, q, v, mask=None):
        B, Tk, C = k.size() # batch size, size of the keys, size of the embeddings
        Tq = q.size(1) # size of the queries
        
        # apply projections
        k = self.key(k)
        q = self.query(q)
        v = self.value(v)
        
        k = k.view(B, Tk, self.n_head, self.head_size).transpose(1, 2) # (B, n_heads, key_size, head_size)
        q = q.view(B, Tq, self.n_head, self.head_size).transpose(1, 2) # (B, n_heads, query_size, head_size)
        v = v.view(B, Tk, self.n_head, self.head_size).transpose(1, 2) # (B, n_heads, value_size, head_size)
        
        att = (q @ k.transpose(-2, -1) * (1.0 / th.sqrt(th.tensor(self.head_size)))) # (B, n_head, query_size, key_size)
        
        # padding mask
        mask = mask.unsqueeze(1) # (B, 1, query_size, key_size)
        
        # during mask attention apply also causal mask
        if self.mask is not None:
            mask = (self.bias[:, :, :Tq, :Tk]) & mask
        
        att = att.masked_fill(mask == 0, -1e9)
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        y = att @ v # (B, n_head, len_seq, head_size)
        y = y.transpose(1, 2).contiguous().view(B, Tq, C) # (B, len_seq, n_embed)
        
        y = self.linear_out(y)
        return y
    
class FeedForward(nn.Module):
    
    def __init__ (self, config):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(config.n_embed, config.n_embed*4),
            nn.ReLU(),
            nn.Linear(config.n_embed*4,config.n_embed),
            nn.Dropout(config.dropout),
        )
        
    def forward(self, x):
        return self.fc(x)
        
        
class EncoderBlock(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config, self_mask=None) # self-attention without causal mask
        self.dropout1 = nn.Dropout(config.dropout)
        self.ln1 = nn.LayerNorm(config.n_embed)
        self.ff = FeedForward(config)
        self.dropout2 = nn.Dropout(config.dropout)
        self.ln2 = nn.LayerNorm(config.n_embed)
        
    def forward(self, x, mask=None):
        att_output = self.attention(k=x, q=x, v=x, mask=mask) # keys, queries and values comes from the encoder
        x = x + self.dropout1(att_output)
        x = self.ln1(x)
        
        ff_output = self.ff(x)
        x = x + self.dropout2(ff_output)
        x = self.ln2(x)
        
        return x
        
        
class Encoder(nn.Module):
    
    def __init__(self,config):
        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock(config) for _ in range(config.n_layer)])
        
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)     
            
        return x
                
                
class DecoderBlock(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.masked_attention = MultiHeadAttention(config, self_mask=True) # masked self-attention
        self.dropout1 = nn.Dropout(config.dropout)
        self.ln1 = nn.LayerNorm(config.n_embed)
        self.cross_attention = MultiHeadAttention(config, self_mask=None) # cross-attention without mask
        self.dropout2 = nn.Dropout(config.dropout)
        self.ln2 = nn.LayerNorm(config.n_embed)
        self.ff = FeedForward(config)
        self.dropout3 = nn.Dropout(config.dropout)
        self.ln3 = nn.LayerNorm(config.n_embed)
        
    def forward(self, x, enc_out, mask=None, cross_mask=None):
        self_att_output = self.masked_attention(k=x, q=x, v=x, mask=mask) # self-attention, k q and v comes from the decoder
        x = x + self.dropout1(self_att_output)
        x = self.ln1(x)
        
        cross_att_output = self.cross_attention(k=enc_out, q=x, v=enc_out, mask=cross_mask) # cross-attention, q comes from the decoder, k and v from the encoder 
        x = x + self.dropout2(cross_att_output)
        x = self.ln2(x)
        
        ff_output = self.ff(x)
        x = x + self.dropout3(ff_output)
        x = self.ln3(x)
        
        return x
    
class Decoder(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layer)])
        
    def forward(self, x, enc_out, mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, enc_out, mask=mask, cross_mask=cross_mask)
            
        return x
        
        
class Transformer(nn.Module):
    
    def __init__(self, config):
        super().__init__() 
        self.embed = EmbeddingLayer(config) 
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.linear_out = nn.Linear(config.n_embed, config.vocab_size)
    
    # padding mask to hide the pad token in the attention
    def get_pad_mask(self, seq):
        pad_token_id = 50259
        return (seq != pad_token_id)
        
    def forward(self, enc_input, dec_input, target=None):
        
        enc_mask = self.get_pad_mask(enc_input)
        # the encoder mask will have zeros in the in the rows and columns corresponding to the padding token
        enc_mask = enc_mask.unsqueeze(-2) & enc_mask.unsqueeze(-1) # (B, encoder_seq_length, encoder_seq_length)
        
        # the decoder mask will have zeros in the in the rows and columns corresponding to the padding token
        dec_mask = self.get_pad_mask(dec_input)
        dec_mask = dec_mask.unsqueeze(-2) & dec_mask.unsqueeze(-1) # (B, decoder_seq_length, decoder_seq_length)
        
        # mask for cross-attention, on the rows we have the decoder tokens and on the columns the encoder tokens,
        # when there is a padding token the corresponding row/column will have zero value  
        cross_mask = enc_mask[:, 0, :].unsqueeze(-2) & dec_mask[:, 0, :].unsqueeze(-1) # (B, decoder_seq_length, encoder_seq_length)
        
        enc_input_embed = self.embed(enc_input)
        dec_input_embed = self.embed(dec_input)
        enc_output = self.encoder(enc_input_embed, mask=enc_mask)
        decoder_output = self.decoder(dec_input_embed, enc_output, mask=dec_mask, cross_mask=cross_mask)
        
        
        if target is not None:
            logits = self.linear_out(decoder_output)
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            target = target.view(B * T)
            loss = F.cross_entropy(logits, target, ignore_index=50259)
        else:
            # during inference only applying transformation to the last position
            logits = self.linear_out(decoder_output[:, [-1], :])
            loss = None
        
        return logits, loss

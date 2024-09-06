"""
Model parameters 
"""

class Config:
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_embed: int = 512  # size of the embeddings
    len_seq: int = 128  # max length of the input sequences. You can change this as you wish, The longest sentence in the used dataset has 193 tokens
    n_head: int = 8
    n_layer: int = 6
    dropout: float = 0.1
    batch_size: int = 384  # depending on the device
    max_iter: int = 22 # epochs

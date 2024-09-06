# Attention_Is_All_You_Need

In this repo i tried to reproduce the transformer from the paper <a href="https://arxiv.org/pdf/1706.03762" target="_blank">Attention_Is_All_You_Need<a> and use it to make some translation from English-to-Italian. The network uses an Encoder-Decoder architecture based solely on attention mechanisms. The code for the Transformer can be found in the [model.py](model.py) file.

For the encoding i used the Open Ai's [tiktoken]([path/to/file](https://github.com/openai/tiktoken?tab=readme-ov-file)) BPE tokenizer which has a vocabulary size of 50257 tokens unlike the paper where they used a vocabulary of 32K tokens. The encodings procedure can be found inside the [utils](utils) directory.



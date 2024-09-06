# Attention_Is_All_You_Need

In this repo i tried to reproduce the transformer from the paper <a href="https://arxiv.org/pdf/1706.03762" target="_blank">Attention_Is_All_You_Need<a> and use it to make some translation from English-to-Italian. The network uses an Encoder-Decoder architecture based solely on attention mechanisms. The code for the Transformer can be found in the [model.py](model.py) file.

## Dataset
I decided to use the [Europarl Parallel Corpus](https://www.statmt.org/europarl/) Italian-English which is composed by 1,909,115 parallel sentences which is much smaller corpus than the one used in the paper (17M sentences). You can also find the .txt files inside the [data](data) directory.


### Encoding
For the text encoding i used the Open Ai's [tiktoken]([path/to/file](https://github.com/openai/tiktoken?tab=readme-ov-file)) BPE tokenizer which has a vocabulary size of 50257 tokens unlike the paper where they used a vocabulary of 32K tokens. The encodings code can be found inside the [utils](utils) directory.

For the encoding procedure we use the following command that will create a .pth file inside the data directory called tokenized_parallel_sentences.pth that will contain a list of pairs

```
python tokenization.py
```

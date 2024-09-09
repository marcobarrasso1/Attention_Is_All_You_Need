# Attention_Is_All_You_Need

In this repo i tried to reproduce the Transformer from the paper <a href="https://arxiv.org/pdf/1706.03762" target="_blank">Attention_Is_All_You_Need<a> and use it to make some translation from English-to-Italian. The network uses an Encoder-Decoder architecture based solely on attention mechanisms. The code for the Transformer can be found in the [model.py](model.py) file.


## Encoding
For the text encoding i used the Open Ai's [tiktoken]([path/to/file](https://github.com/openai/tiktoken?tab=readme-ov-file)) BPE tokenizer which has a vocabulary size of 50257 tokens unlike the paper where they used a vocabulary of 32K tokens. The encodings code can be found inside the [utils](utils) directory.

For the encoding procedure we run the following command from the utils directory:

```
python tokenization.py
```

this will create a .pkl file called tokenized_parallel_sentences.pkl inside the data directory that will contain a list of pairs where in the first position of the pair there is the tokenized sentence in the source language (English in my case) and in the second one the tokenized sentence in the target language (Italian).


Example:

-Parallele sentences in the dataset:<br> 
('In the U.S., it is illegal to torture people in order to get information from them.', <br>
 'Negli Stati Uniti è illegale torturare le persone per poter ottenere informazioni da loro.') 
 
 -After tokenization: <br>
 ([818,262,471,13,50,1539,340,318,5293,284,11543,661,287,1502,284,651,1321,422,606,13,50258],
 [32863,4528,5133,72,791,8846,6184,101,4416,1000,7619,333,533,443,2774,505,583,1787,263,267,32407,567,4175,1031,295,72,12379,300,16522,13,50258])


 ## Training

The model can be trained with one gpu: 
```
python train.py
```
but the code also support distributed training. In my case i trained the model on 4 RTX 4090 gpus with the following command:
```
torchrun --standalone --nproc_per_node=4 train.py
```
the --standalone flag is used to indicate that the processes will be located on the same node.

### Model Configuration

I used the base model configuration described in the original paper:

![Model Config](images/model_config.png)

The model configuration is located in the [config.py](config.py) file where i add the batch size and the max epoch for the training.
I used a batch size of 384 since i used 4 gpus and i wanted to make the most of the available RAM and trained for 22 epochs. The training took almost 3 hours.


### Dataset
I decided to use the [Europarl Parallel Corpus](https://www.statmt.org/europarl/) Italian-English which is composed by 1,909,115 parallel sentences which is much smaller corpus than the one used in the paper (17M sentences). You can also find the .txt files inside the [data](data) directory.
I used a smaller dataset because for a really large dataset, like the one in the paper, the training would have taken too much time and resources.

### Results



import torch as th
import random
from torch.nn import functional as F

from config import Config
from encoding.tokenization import enc

# function to translate    
def pred(src_seq, model):
    sos_token_id = 50259
    eos_token_id = 50258
    translation = [sos_token_id] # at the beginning the decoder takes in input only the start of sequence token
    
    for i in range(Config.len_seq):
        target_seq = th.tensor([translation], device=device)
        output, _ = model(src_seq, target_seq)
        
        probs = F.softmax(output, dim=-1)
        word_pred = th.argmax(probs, dim=-1)

        translation.append(word_pred.item())

        if word_pred.item() == eos_token_id:
            break
    
    return translation[1:]


def clean_decode(seq):
    special_tokens = ["<s>", "</s>", "<pad>"]
    
    decoded = enc.decode(seq)
    
    for token in special_tokens:
        decoded = decoded.replace(token, '')
        
    return decoded
    
# function to make translation of sentences in the validation set
def test(n, model):

    print("Random Tests")
    print("*"*30)
    for i in range(n):
        random_batch = random.choice(list(data_loader_val))
        source_test, _, target_test = random_batch
        actual_batch_size = source_test.size(0)

        random_sample = random.sample(range(actual_batch_size), 1)[0]
        src_seq, true_seq = source_test[random_sample, :].unsqueeze(0), target_test[random_sample, :]
        src_seq = src_seq.to(device)
        
        out = pred(src_seq, model)
        
        print(f"Prediction: {clean_decode(out)}, True: {clean_decode(true_seq.tolist())}, Source: {clean_decode(src_seq[0].tolist())} \n")
    print("*"*30)


def translate(sentence, model):
    
    pad_token_id = 50259
    print(sentence)
    sentence = enc.encode(sentence + "</s>", allowed_special={'</s>'})
    sentence = sentence[:Config.len_seq] + [pad_token_id] * (Config.len_seq - len(sentence))
    sentence = th.tensor(sentence, dtype=th.long).unsqueeze(0)
    sentence = sentence.to(device)
    
    translation = pred(sentence, model)
    
    print(clean_decode(translation))

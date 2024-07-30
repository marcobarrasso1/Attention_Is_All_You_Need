import torch as th
from torch.nn import functional as F
import random
from collections import OrderedDict
import argparse

from model import Transformer
from config import Config
from encoding.tokenizer import enc
from train import data_loader_val

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
    
device = th.device("cuda" if th.cuda.is_available() else "cpu")

model = Transformer(Config)
model.to(device)

state_dict = th.load("weigths/model_weigths_epoch_70.pth", map_location=device)

new_state_dict = OrderedDict()
for k, v in state_dict.items():
    if k.startswith("module._orig_mod."):
        # Remove the prefix 'module._orig_mod.'
        name = k.replace("module._orig_mod.", "")
    elif k.startswith("module."):
        # Remove the prefix 'module.'
        name = k.replace("module.", "")
    else:
        name = k
    new_state_dict[name] = v

# load the weigths
model.load_state_dict(new_state_dict)

parser = argparse.ArgumentParser(description='Number of senteces to translate.')
parser.add_argument('number', type=int, help='.')

# Parse the arguments
args = parser.parse_args()

test(args.numer, model)

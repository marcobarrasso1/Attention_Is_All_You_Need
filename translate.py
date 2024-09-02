import torch as th
from collections import OrderedDict

from utils import clean_decode, translate
from model import Transformer
from config import Config
from encoding.tokenizer import enc
    

device = th.device("cuda") if th.cuda.is_available() else th.device("cpu")
model = Transformer(Config)
model.to(device)

state_dict = th.load('weights/model_weights_epoch_70.pth', map_location=device) # load the model parameters

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

# load the weights
model.load_state_dict(new_state_dict)

sentence = input("Enter a sentence: ")

model.eval()
with th.no_grad():
    translate(sentence, model)

import torch as th
from collections import OrderedDict
import argparse
import os

from model import Transformer
from config import Config
from utils.utils import test

    
#device = th.device("cuda" if th.cuda.is_available() else "cpu")
#device = th.device("mps" if th.backends.mps.is_available() else "cpu")
device = "cpu"
model = Transformer(Config)
model.to(device)

#file_path = os.path.expanduser('~/Documents/prova/weights/model_weights_epoch_9.pth')
state_dict = th.load('weights/model_weights_epoch_21.pth', map_location=device)
#state_dict = th.load(file_path, map_location=device)

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

data_loader_val = th.load('data/data_loader_val.pth')

test(args.number, model, data_loader_val, Config, device)

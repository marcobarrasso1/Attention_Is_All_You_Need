"""
This training script can be run both on a single gpu
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py
"""


import torch as th
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import sys
import random
import pickle
import os

from model import Transformer
from config import Config
from utils.data_loader import create_data_loaders

ddp = int(os.environ.get('RANK', -1)) != -1

if ddp:
    # with multiple gpus
    assert th.cuda.is_available()
    dist.init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    th.cuda.set_device(device)
    master_process = ddp_local_rank == 0
else:
    # with a single gpu
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    ddp_rank = 0
    ddp_world_size = 1
    master_process = True

print("Using device:", device)

th.manual_seed(1339)
if th.cuda.is_available():
    th.cuda.manual_seed(1339)


with open('data/tokenized_parallel_sentences.pkl', 'rb') as f:
    tokenized_parallel_sentences = pickle.load(f)

# taking sentence that has a desired maximum length
tokenized_parallel_sentences = [sent for sent in tokenized_parallel_sentences if len(sent[0]) < Config.len_seq and len(sent[1]) < Config.len_seq]

DATASET_SIZE = 1.908.480 # the real size of the dataset is 1,909,115. Rounded (for efficiency) to the nearest number divisible by the used batch size (384)

# sampling as many parallel sentences as defined in DATASET_SIZE variable
tokenized_parallel_sentences = random.sample(tokenized_parallel_sentences, DATASET_SIZE)

data_loader_train, data_loader_val = create_data_loaders(
    tokenized_parallel_sentences, 
    Config, 
    ddp=ddp, 
    ddp_world_size=ddp_world_size, 
    ddp_rank=ddp_rank
)

th.save(data_loader_val, 'data/data_loader_val.pth')

th.set_float32_matmul_precision('high') # lower the precision for increase the efficiency

model = Transformer(Config)
model.to(device)
model = th.compile(model)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

print(sum(p.numel() for p in model.parameters())/1e6, "M parameters") 

# define the optimezer and learning rate used in the paper
lr= 1
opt1 = th.optim.Adam(model.parameters(),lr=lr, betas=(0.9, 0.98), eps=1e-09)

def lr_rate(step_num, d_model, factor, warmup_steps):
    step_num =max(1,step_num)
    return factor * (
        d_model ** (-0.5) * min(step_num ** (-0.5), step_num * warmup_steps ** (-1.5))
    )

lr_scheduler = LambdaLR(
    optimizer=opt1,
    lr_lambda=lambda step_num: lr_rate(
        step_num, 512, factor=1, warmup_steps=4000
    ),
)
# create a file to save the output
if master_process:
    log_file_path = 'results/training_log.txt'
    log_file = open(log_file_path, 'w')
    sys.stdout = log_file

n_train = len(data_loader_train) * (ddp_world_size if ddp else 1)
n_val = len(data_loader_val) * (ddp_world_size if ddp else 1)

# at each iteration we evaluate the loss on the training and validation set and write the results onto the file
for iter in range(Config.max_iter):  
    
    model.eval()  
    val_loss = 0
    with th.no_grad():
        for batch in data_loader_val:
            enc_input, dec_input, target = batch
            enc_input, dec_input, target = enc_input.to(device), dec_input.to(device), target.to(device)
            
            res, loss = model(enc_input, dec_input, target)
            val_loss += loss.item()
            
    if ddp:
        val_loss_tensor = th.tensor(val_loss, device=device)
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM) # summing losses calculated by the different processes
        val_loss = val_loss_tensor.item()
    
    if master_process:  
        avg_val_loss = val_loss / n_val # avarage validation loss per batch
        print(f"Iteration {iter} | Val Loss: {(avg_val_loss):.4f}")
        sys.stdout.flush()
              
    model.train()
    train_loss = 0  
       
    for i, batch in enumerate(data_loader_train):
        enc_input, dec_input, target = batch
        enc_input, dec_input, target = enc_input.to(device), dec_input.to(device), target.to(device)
        
        opt1.zero_grad()
        res, loss = model(enc_input, dec_input, target)
        train_loss += loss.item() 
        
        loss.backward()
        opt1.step()
        lr_scheduler.step()
        
    if ddp:
        train_loss_tensor = th.tensor(train_loss, device=device)
        dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM) # summing losses calculated by the different processes
        train_loss = train_loss_tensor.item() 

        
    if master_process:
        avg_train_loss = train_loss / n_train # avarage training loss per batch 
        print(f"Iteration {iter} | Train Loss: {(avg_train_loss):.4f} | learning rate: {round(lr_scheduler.get_last_lr()[0],6)}")
        sys.stdout.flush()
    
        
    if iter % 3 == 0: # every 3 epochs save the model parameters
        if master_process:
            model_path = f"weights/model_weights_epoch_{iter}.pth"
            th.save(model.state_dict(), model_path)
            print(f"Model weights saved to {model_path}")
            sys.stdout.flush()
    

if master_process:
    sys.stdout = sys.__stdout__
    log_file.close()


if ddp:
    dist.destroy_process_group()

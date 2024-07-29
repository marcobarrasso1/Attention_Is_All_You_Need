"""
This training script can be run both on a single gpu
and also in a larger training run with distributed data parallel (ddp).
"""

import torch as th
from torch.nn import functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from sklearn.model_selection import train_test_split

import sys
import random
import pickle
import os

from model import Transformer
from config import Config
from encoding.tokenizer import enc

DATASET_SIZE = 351744 # the real size of the dataset is 352019. Rounded (for efficiency) to the nearest number divisible by the used batch size (384)

class TranslationDataset(Dataset):
    def __init__(self, tokenized_pairs, max_length):
        self.tokenized_pairs = tokenized_pairs
        self.pad_token_id = 50259
        self.sos_token_id = 50257 
        self.max_length = max_length

    def __len__(self):
        return len(self.tokenized_pairs)

    def __getitem__(self, idx):
        source_ids, target_ids= self.tokenized_pairs[idx]
        
        # padding the sequences until the max length, encoder input and target sequence do not have the start of sequence token 
        enc_input = source_ids[:self.max_length] + [self.pad_token_id] * (self.max_length - len(source_ids)) 
        target = target_ids[:self.max_length] + [self.pad_token_id] * (self.max_length - len(target_ids))
        dec_input = [self.sos_token_id] + target[:-1] # decoder input with the additional start of sequence token
        
        enc_input = th.tensor(enc_input, dtype=th.long)
        target = th.tensor(target, dtype=th.long)
        dec_input = th.tensor(dec_input, dtype=th.long)

        return enc_input, dec_input, target
    
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
# sampling as many parallel senteces as defined in DATASET_SIZE variable
tokenized_parallel_sentences = random.sample(tokenized_parallel_sentences, DATASET_SIZE)

train, val = train_test_split(tokenized_parallel_sentences, test_size=0.1, random_state=42)
dataset_train = TranslationDataset(train, Config.len_seq)
dataset_val = TranslationDataset(val, Config.len_seq)

if ddp:
    # we are splitting the original dataset into chunks equal to the number of gpus and each of this chunk is assigned to 
    # a different one. Each gpu will have its own data loader where the batch size is the global batch size divided by 
    # the number of gpus
    train_sampler = DistributedSampler(dataset_train, num_replicas=ddp_world_size, rank=ddp_rank)
    val_sampler = DistributedSampler(dataset_val, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=False)
    data_loader_train = DataLoader(dataset_train, batch_size=Config.batch_size//ddp_world_size, sampler=train_sampler)
    data_loader_val = DataLoader(dataset_val, batch_size=Config.batch_size//ddp_world_size, sampler=val_sampler)
else:
    data_loader_train = DataLoader(dataset_train, batch_size=Config.batch_size, shuffle=True)
    data_loader_val = DataLoader(dataset_val, batch_size=Config.batch_size, shuffle=False)
    
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

for iter in range(Config.max_iter):        
    model.train()
    train_loss = 0  
    
    if ddp:
        data_loader_train.sampler.set_epoch(iter)
        
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
        avg_train_loss = train_loss / n_train # avarage loss per batch 
        print(f"Iteration {iter} | Train Loss: {(avg_train_loss):.4f} | learning rate: {round(lr_scheduler.get_last_lr()[0],6)}")
        sys.stdout.flush()
    
    if iter % 5 == 0: # every 5 epoch calculate the loss on the validation set
        model.eval()  
        val_loss = 0
        with th.no_grad():
            for batch in data_loader_val:
                enc_input, dec_input, target = batch
                enc_input, dec_input, target = enc_input.to(device), dec_input.to(device), target.to(device)
                
                res, loss = model(enc_input, dec_input, target)
                val_loss += loss.item()
                #test(1, model)
                
        if ddp:
            val_loss_tensor = th.tensor(val_loss, device=device)
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
            val_loss = val_loss_tensor.item()
        
        if master_process:  
            avg_val_loss = val_loss / n_val
            print(f"Iteration {iter} | Val Loss: {(avg_val_loss):.4f}")
            sys.stdout.flush()
        
    if iter % 10 == 0: # every ten epochs save the model parameters
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

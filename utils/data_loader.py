import torch as th
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader, DistributedSampler
from sklearn.model_selection import train_test_split


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
    
    

def create_data_loaders(tokenized_parallel_sentences, Config, ddp=False, ddp_world_size=None, ddp_rank=None):
    train, val = train_test_split(tokenized_parallel_sentences, test_size=0.15, random_state=48)
    dataset_train = TranslationDataset(train, Config.len_seq)
    dataset_val = TranslationDataset(val, Config.len_seq)
    
    if ddp:
        train_sampler = DistributedSampler(dataset_train, num_replicas=ddp_world_size, rank=ddp_rank)
        val_sampler = DistributedSampler(dataset_val, num_replicas=ddp_world_size, rank=ddp_rank)
        data_loader_train = DataLoader(dataset_train, batch_size=Config.batch_size // ddp_world_size, sampler=train_sampler)
        data_loader_val = DataLoader(dataset_val, batch_size=Config.batch_size // ddp_world_size, sampler=val_sampler)
    else:
        data_loader_train = DataLoader(dataset_train, batch_size=Config.batch_size, shuffle=True)
        data_loader_val = DataLoader(dataset_val, batch_size=Config.batch_size, shuffle=False)
    
    return data_loader_train, data_loader_val

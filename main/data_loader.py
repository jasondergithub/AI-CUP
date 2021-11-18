'''
可以回傳mini-batch的DataLoader
製作mask tensor
'''

from torch.utils import data
import dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch

def create_mini_batch(samples):
    tokens_tensors = [s[0] for s in samples]
    segments_tensors = [s[1] for s in samples]

    tokens_tensors = pad_sequence(tokens_tensors, batch_first=True)                                      
    segments_tensors = pad_sequence(segments_tensors, batch_first=True)

    masks_tensors = torch.zeros(tokens_tensors.shape, dtype=torch.long)
    masks_tensors = masks_tensors.masked_fill(tokens_tensors != 0, 1)
                                                                    
    return tokens_tensors, segments_tensors, masks_tensors

BATCH_SIZE = 32
trainloader = DataLoader(dataset.trainset, batch_size=BATCH_SIZE, collate_fn=create_mini_batch)

data = next(iter(trainloader))
tokens_tensors, segments_tensors, masks_tensors = data

print(f"""
tokens_tensors.shape   = {tokens_tensors.shape} 
{tokens_tensors}
------------------------
segments_tensors.shape = {segments_tensors.shape}
{segments_tensors}
------------------------
masks_tensors.shape    = {masks_tensors.shape}
{masks_tensors}
""")    
import config
import data_loader
import engine
import torch
import pandas as pd

from bert_auto_encoder_base1 import Model
from dataset import encoderDataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

device = torch.device(config.DEVICE)
model = Model()
model.load_state_dict(torch.load(config.MODEL_PATH))
model.to(device)

trainset = encoderDataset('train', config.tokenizer)
tokens_tensors, segments_tensors = trainset.__getitem__(0)
# tokens_tensors = pad_sequence(tokens_tensors, batch_first=True)                                      
# segments_tensors = pad_sequence(segments_tensors, batch_first=True)
masks_tensors = torch.zeros(tokens_tensors.shape, dtype=torch.long)
masks_tensors = masks_tensors.masked_fill(tokens_tensors != 0, 1)

model.eval()
outputs = model(tokens_tensors=tokens_tensors,
                segments_tensors=segments_tensors,
                masks_tensors=masks_tensors)
targets = model.bert_output[0]
del model.bert_output[0]

loss = torch.nn.MSELoss()
output = loss(outputs, targets)
print(f'sample 0 loss = {loss}')
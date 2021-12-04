import config
import data_loader
import engine
import torch
import pandas as pd

from bert_auto_encoder_base1 import Model
from dataset import encoderDataset

device = torch.device(config.DEVICE)
model = Model()
model.load_state_dict(torch.load(config.MODEL_PATH))
model.to(device)

# trainset = encoderDataset('train', config.tokenizer)
# tokens_tensors, segments_tensors = trainset.__getitem__(0)

# masks_tensors = torch.zeros(tokens_tensors.shape, dtype=torch.long)
# masks_tensors = masks_tensors.masked_fill(tokens_tensors != 0, 1)
# tokens_tensors = tokens_tensors.unsqueeze(0)
# segments_tensors = segments_tensors.unsqueeze(0)
# masks_tensors = masks_tensors.unsqueeze(0)

# tokens_tensors = tokens_tensors.to(device)
# masks_tensors = masks_tensors.to(device)
# segments_tensors = segments_tensors.to(device)

# model.eval()
# outputs = model(tokens_tensors=tokens_tensors,
#                 segments_tensors=segments_tensors,
#                 masks_tensors=masks_tensors)
# targets = model.bert_output[0]
# del model.bert_output[0]

# loss = torch.nn.MSELoss()
# output = loss(outputs, targets)
# print(f'sample 0 loss = {output}')

# print('======================================================================================')
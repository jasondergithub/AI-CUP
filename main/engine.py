import torch
import torch.nn as nn
from tqdm import tqdm

def loss_fn(outputs, targets):
    return nn.MSELoss()(outputs, targets)

def train_fn(data_loader, model, optimizer, device, scheduler):
    model.train()

    loss = 0

    for  batch_index, data in tqdm(enumerate(data_loader), total=len(data_loader)):
        if next(model.parameters()).is_cuda:
             data = [t.to("cuda:0") for t in data if t is not None]
        
        tokens_tensors, segments_tensors, masks_tensors = data[:3]

        optimizer.zero_grad()
        outputs = model(input_ids=tokens_tensors, attention_mask=masks_tensors, token_type_ids=segments_tensors)
        targets = model.bert_output

        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()

    return loss

def eval_fn(data_loader, model, device):
    model.eval()

    outpus_loss = []
    with torch.no_grad():
        pass
from numpy import dtype
import torch
import torch.nn as nn
from tqdm import tqdm

def loss_fn(outputs, targets):
    return nn.BCEWithLogitsLoss()(outputs, targets)

def train_fn(data_loader, model, optimizer, device, scheduler):
    model.train()

    running_loss = 0
    for batch_index, data in tqdm(enumerate(data_loader), total=len(data_loader)):
        tokens_tensor = data['tokens_tensor']
        segments_tensor = data['segments_tensor']
        masks_tensor = data['masks_tensor']
        targets = data['target']

        tokens_tensor = tokens_tensor.to(device, dtype=torch.long)
        segments_tensor = segments_tensor.to(device, dtype=torch.long)
        masks_tensor = masks_tensor.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)

        optimizer.zerp_grad()
        outputs = model(input_ids=tokens_tensor, token_type_ids=segments_tensor, attention_mask=masks_tensor)
        logits = outputs.logits

        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()

    return running_loss    

def eval_fn(data_loader, model, device):
    model.eval()
    fin_targets = []
    fin_outputs = []

    with torch.no_grad():
        for batch_index, data in tqdm(enumerate(data_loader), total=len(data_loader)): 
            tokens_tensor = data['tokens_tensor']
            segments_tensor = data['segments_tensor']
            masks_tensor = data['masks_tensor']
            targets = data['target']

            tokens_tensor = tokens_tensor.to(device, dtype=torch.long)
            segments_tensor = segments_tensor.to(device, dtype=torch.long)
            masks_tensor = masks_tensor.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.float)   

            outputs = model(input_ids=tokens_tensor, token_type_ids=segments_tensor, attention_mask=masks_tensor)
            logits = outputs.logits   
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(logits).cpu().detach().numpy().tolist())
    
    return fin_outputs, fin_targets
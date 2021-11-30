import config
import data_loader
import engine
import pandas as pd
import torch
import torch.nn as nn

from bert_auto_encoder_base1 import Model
from dataset import encoderDataset
from torch.utils.data import DataLoader
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

def run_train():
    trainset = encoderDataset('train', config.tokenizer)
    train_data_loader = DataLoader(trainset, batch_size=config.BATCH_SIZE, collate_fn=data_loader.create_mini_batch)

    device = torch.device(config.DEVICE)
    model = Model()
    model.to(device)
    
    # I am not familiar with this part yet
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    length_df = len(pd.read_csv('../data/TrainLabel.csv'))
    num_train_steps = int(length_df / config.BATCH_SIZE * config.EPOCHS)
    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )
    #########

    for epoch in range(config.EPOCHS):
        loss = engine.train_fn(train_data_loader, model, optimizer, device, scheduler)
        print(f'Epoch:{epoch+1}, Loss:{loss.item():.4f}')

    #save model
    torch.save(model.state_dict(), config.MODEL_PATH)
    
if __name__ == "__main__":
    run_train()
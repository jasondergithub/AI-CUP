import config
import data_loader
import engine
import pandas as pd
import pickle
import numpy as np
import torch
import torch.nn as nn

from stage1_dataset import ArticleClassificationDataset
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn import metrics

def run_train(tableNumber):
    trainset = ArticleClassificationDataset('train', tableNumber)
    train_data_loader = DataLoader(trainset, batch_size=config.BATCH_SIZE, num_workers=4)

    device = torch.device(config.DEVICE)
    model =  BertForSequenceClassification(config.bert_config)
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


    with open('../../table/table'+ str(tableNumber) +'.txt', 'rb') as fp:
        table = pickle.load(fp)
    length_df = len(table)
    num_train_steps = int(length_df / config.BATCH_SIZE * config.EPOCHS)
    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )
    #########
    
    #best_accuracy = 0

    for epoch in range(config.EPOCHS):
        loss = engine.train_fn(train_data_loader, model, optimizer, device, scheduler)
        print(f'Epoch:{epoch+1}, Loss:{loss.item():.4f}')
        outputs, targets = engine.eval_fn(train_data_loader, model, device)
        outputs = np.array(outputs) >= 0.5
        accuracy = metrics.accuracy_score(targets, outputs)
        print(f"After training {epoch+1} epoch(s), Accuracy Score = {accuracy}")
    #save model
    torch.save(model.state_dict(), config.MODEL_PATH)
    
if __name__ == "__main__":
    for i in range(10):
        run_train(i+1)
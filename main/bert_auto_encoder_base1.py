'''
模型架構:
    |----變數 bertList
    |      |-----儲存BertModel每次mini batch的輸出，用在之後跟整個模型輸出的比較
    |
    |----BertModel : (batch, seq_len, 768)
    |
    |----Encoder
    |      |----linear(768, 128)
    |      |----Relu
    |      |----linear(128, 64)
    |      |----Relu
    |      |----linear(64, 32) : (batch, seq_len, 32)
    |
    |----Decoder
    |      |----linear(32, 64)
    |      |----Relu
    |      |----linear(64, 128)
    |      |----Relu
    |      |----linear(128, 768) : 模型的輸出                           
'''

from transformers import BertModel

import config
import torch.nn as nn

PRETRAINED_MODEL_NAME = "bert-base-chinese" 

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.bert_output = []
        self.bert = BertModel(config.bert_config)
        self.encoder = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 32)
        )

        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 768)
        )

    def forward(self, tokens_tensors, segments_tensors, masks_tensors): # data: (batch, seq_len)
        outputs = self.bert(input_ids = tokens_tensors, token_type_ids = segments_tensors, 
            attention_mask = masks_tensors)
        
        self.bert_output.append(outputs.last_hidden_state) # save bert output as labels
        encoded = self.encoder(outputs.last_hidden_state)
        decoded = self.decoder(encoded)

        return decoded

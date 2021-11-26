'''
讀入兩篇txt檔，將裡面的文章轉換成Bert接受的格式
[CLS]文章1[SEP]文章2[SEP]

讀取 TrainLabel.csv，根據每一列讀到的號碼去存取有關連的兩篇文章
'''
import pandas as pd
import torch
from torch.utils.data import Dataset

class encoderDataset(Dataset):
    def __init__(self, mode, tokenizer) :
        assert mode in ['train', 'test']
        self.mode = mode
        self.df = pd.read_csv('../data/TrainLabel.csv')
        self.len = len(self.df)
        self.tokenizer = tokenizer
    
    def __getitem__(self, index):
        if self.mode == 'train':
            num1 = self.df.iloc[index, 0]
            num2 = self.df.iloc[index, 1]
            with open('../processed_files/' + str(num1) + '.txt', 'r', encoding='UTF-8') as text1:
                file1 = text1.read()
            with open('../processed_files/' + str(num2) + '.txt', 'r', encoding='UTF-8') as text2:
                file2 = text2.read() 
            
            wordpieces = ['[CLS]']
            tokens1 = self.tokenizer.tokenize(file1)
            wordpieces += tokens1 + ['[SEP]']
            article1_len = len(wordpieces)
            tokens2 = self.tokenizer.tokenize(file2)
            wordpieces += tokens2 + ['[SEP]']
            article2_len = len(wordpieces) - article1_len

            ids = self.tokenizer.convert_tokens_to_ids(wordpieces)
            tokens_tensor = torch.tensor(ids)

            segments_tensor = torch.tensor([0] * article1_len + [1] * article2_len, dtype=torch.long)
            
            return (tokens_tensor, segments_tensor)

    def __len__(self):
        return self.len
'''
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
trainset = encoderDataset('train', tokenizer)

sample_idx = 0
num1 = trainset.df.iloc[sample_idx, 0]
num2 = trainset.df.iloc[sample_idx, 1]
with open('../processed_files/' + str(num1) + '.txt', 'r', encoding='UTF-8') as text1:
    file1 = text1.read()
with open('../processed_files/' + str(num2) + '.txt', 'r', encoding='UTF-8') as text2:
    file2 = text2.read()


tokens_tensor, segments_tensor = trainset[sample_idx]
tokens = tokenizer.convert_ids_to_tokens(tokens_tensor.tolist())
combined_text = "".join(tokens)

# 渲染前後差異，毫無反應就是個 print。可以直接看輸出結果
print(f"""[原始文本]
句子 1：{file1}
句子 2：{file2}

--------------------

[Dataset 回傳的 tensors]
tokens_tensor  ：{tokens_tensor}

segments_tensor：{segments_tensor}

--------------------

[還原 tokens_tensors]
{combined_text}
""")
'''
from transformers import BertConfig, BertTokenizer

DEVICE = "cuda"
EPOCHS = 10
BATCH_SIZE = 32
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
MODEL_PATH = "../saved_model"

bert_config = BertConfig.from_pretrained('bert-base-chinese')
bert_config.num_hidden_layers = 6
bert_config.num_attention_heads = 6
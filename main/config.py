from transformers import BertConfig, BertTokenizer
from transformers.models import bert

DEVICE = "cuda"
EPOCHS = 3
BATCH_SIZE = 16
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
MODEL_PATH = "../saved_model"

bert_config = BertConfig.from_pretrained('bert-base-chinese')
bert_config.max_position_embeddings = 1000
bert_config.num_hidden_layers = 6
bert_config.num_attention_heads = 4
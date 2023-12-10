# Entitty-Extraction AKA Q&A
Tweet sentiment extraction

Update config in order to train on Tweets for text extraction also known as (AKA) Question and answer.


## command to train the Bert based model after updating config file
```python train_TE.py```

```

'''


'''
import transformers
import tokenizers
import os

DEVICE = "cuda:1"
num_classes = 2 # for QandA
MAX_LEN =128
FOLD = 5
LR = 5e-5
TRAIN_BATCH_SIZE = 64
VALID_BATCH_SIZE = 64
EPOCHS = 50
checkpoint = 'exp1'
aug = True
BERT_PATH = "bert-base-uncased"
MODEL_PATH = "./model.pth"
TRAINING_FILE = ./inputs/train.csv"
SPLIT = 0.75
return_dict = False
#TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)
TOKENIZER = tokenizers.BertWordPieceTokenizer(f"./inputs/vocab.txt",lowercase=True)


```

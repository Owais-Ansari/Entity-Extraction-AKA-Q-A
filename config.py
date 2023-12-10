'''
Created on 22-Sep-2023

@author: owaishs

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
TRAINING_FILE = "/home/owaishs/temp2/Scripts/git-workspace/NLP/QandA/inputs/train.csv"
SPLIT = 0.75
return_dict = False
#TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)
TOKENIZER = tokenizers.BertWordPieceTokenizer(f"./inputs/vocab.txt",lowercase=True)

# ##～





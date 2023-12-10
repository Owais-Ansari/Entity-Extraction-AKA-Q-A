import config
from dataloader import TweetDataset_train,TweetDataset_val
import os
import engine
import torch
import utils


import shutil
import pandas as pd
import torch.nn as nn
import numpy as np
from torch.optim import lr_scheduler

from utils.model import BERTBaseUncasedTweet
from sklearn import model_selection
from sklearn import metrics
import transformers
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from pickle import TRUE

# os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

def optimizer_params(model):
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]
    return optimizer_parameters

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename = 'checkpoint.pth.tar'):
    #filepath = os.path.join(checkpoint, str(state['epoch'])+'_'+filename)
    filepath = os.path.join(checkpoint, str(filename))
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


def run(fold):
    best_jaccard = 0
    dfx = pd.read_csv(config.TRAINING_FILE)
    dfx = dfx[dfx.notna()]
    dfx = dfx[dfx['selected_text'].notna()]
    #dfx = dfx.notna()
#
    # df_train = dfx[dfx.kfold != fold].reset_index(drop=True)
    # df_valid = dfx[dfx.kfold == fold].reset_index(drop=True)
    #
    # df["kfold"] = -1

    dfx = dfx.sample(frac=1).reset_index(drop=True)
    df_train = dfx[:int(config.SPLIT * len(dfx))]
    df_valid = dfx[int(config.SPLIT * len(dfx)):]
      

    kf = model_selection.StratifiedKFold(n_splits=5)
    print(df_train.shape)
    print(df_valid.shape)
    train_dataset = TweetDataset_train(
        tweet=df_train.text.values,
        sentiment=df_train.sentiment.values,
        selected_text=df_train.selected_text.values,
        aug=config.aug)

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
        num_workers=6,
        drop_last=True,
    )

    valid_dataset = TweetDataset_val(
        tweet=df_valid.text.values,
        sentiment=df_valid.sentiment.values,
        selected_text=df_valid.selected_text.values
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.VALID_BATCH_SIZE,
        num_workers=2
    )

    device = torch.device(config.DEVICE)
    model = BERTBaseUncasedTweet(num_classes=config.num_classes)
    model.to(device)

    num_train_steps = int(len(df_train) / config.TRAIN_BATCH_SIZE * config.EPOCHS)

    optimizer = AdamW(optimizer_params(model), lr=config.LR)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_train_steps)


    scaler = torch.cuda.amp.GradScaler()
    checkpoint_dir = os.path.join('checkpoints', config.checkpoint)
    os.makedirs(checkpoint_dir, exist_ok=True)
    for epoch in range(config.EPOCHS):
        print('Epoch number: ', epoch)
        jaccard_train = engine.train_fn(train_data_loader, model, optimizer, device, scaler, scheduler=scheduler)
        jaccard_val = engine.eval_fn(valid_data_loader, model, device)
        
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict':model.state_dict(), #to  avoid adding module in the keys name (model.state_dict() replace by model.module.state_dict())
            #'acc': test_acc,

            'best_jaccard': best_jaccard,
            'optimizer' : optimizer.state_dict(),
        }, jaccard_val>best_jaccard, checkpoint=checkpoint_dir)
        if jaccard_val>best_jaccard:
            print(f"Jaccard Score for train and best val  = {jaccard_train,jaccard_val}")  
        best_jaccard = max(jaccard_val, best_jaccard)
if __name__ == "__main__":
    FOLD = config.FOLD
    run(fold=FOLD)
    
    
    
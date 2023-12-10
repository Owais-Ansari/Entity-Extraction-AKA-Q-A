from utils.eval import jaccard
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import re
import string
import config
from utils.misc import AverageMeter
from pandas.conftest import other_closed



scaler = torch.cuda.amp.GradScaler()

def loss_fn(start_logits, end_logits, start_positions, end_positions):
    loss_fct = nn.CrossEntropyLoss()
    start_loss = loss_fct(start_logits, start_positions)
    end_loss = loss_fct(end_logits, end_positions)
    total_loss = (start_loss + end_loss)
    return total_loss


def train_fn(data_loader, model, optimizer, device, scaler, scheduler=None ):
    
    
    model.train()
    losses = AverageMeter()
    jaccards = AverageMeter()
    
    fin_output_start = []
    fin_output_end = []
    fin_padding_lens = []
    fin_tweet_tokens = []
    fin_orig_sentiment = []
    fin_orig_tweet = []
    fin_orig_tweet = []
    fin_orig_selected = []
    
    tk0 = tqdm(data_loader, total=len(data_loader))
    
    for bi, d in enumerate(tk0):

        ids = d["ids"]
        token_type_ids = d["token_type_ids"]
        mask = d["mask"]
        targets_start = d["targets_start"]
        targets_end = d["targets_end"]
        sentiment = d["sentiment"]
        orig_selected = d["original_selected_text"]
        orig_tweet = d["original_tweet"]
        targets_start = d["targets_start"]
        targets_end = d["targets_end"]
        padding_len = d['padding_len']
        orig_sentiment = d["original_sentiment"]
        tweet_tokens = d["tweet_tokens"]
        offsets_start = d["offsets_start"].numpy()
        offsets_end = d["offsets_end"].numpy()

        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets_start = targets_start.to(device, dtype=torch.float)
        targets_end = targets_end.to(device, dtype=torch.float)


        with torch.set_grad_enabled(True):
            with torch.cuda.amp.autocast():
                outputs_start, outputs_end = model(ids=ids,mask=mask,token_type_ids=token_type_ids)
                loss = loss_fn(outputs_start, outputs_end, targets_start, targets_end)
    
        #optimizer.zero_grad()
        #backpropagation
        scaler.scale(loss).backward()
        #updating optimizer
        scaler.step(optimizer)
        # Updates the scale (grad-scale) for next iteration
        scaler.update()
        
        
        
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        outputs_start = torch.softmax(outputs_start, dim=1).cpu().detach().numpy()
        outputs_end = torch.softmax(outputs_end, dim=1).cpu().detach().numpy()
        
        losses.update(loss.item(), ids.size(0))
        tk0.set_postfix(loss=losses.avg)
        
        fin_output_start.append(outputs_start)
        fin_output_end.append(outputs_end)
        fin_padding_lens.extend(padding_len.cpu().detach().numpy().tolist())
        
        fin_tweet_tokens.extend(tweet_tokens)
        fin_orig_sentiment.extend(orig_sentiment)
        fin_orig_tweet.extend(orig_tweet)
        fin_orig_selected.extend(orig_selected)
        
    fin_output_start = np.vstack(fin_output_start)  
    fin_output_end = np.vstack(fin_output_end)
    threshold = 0.2
    jaccards = []
    
    for j in range(len(fin_tweet_tokens)):
        target_string = fin_orig_selected[j]
        tweet_tokens = fin_tweet_tokens[j]
        padding_len = fin_padding_lens[j]
        original_tweet = fin_orig_tweet[j]
        original_sentiment = fin_orig_sentiment[j]
        
        if padding_len>0:
            mask_start = fin_output_start[j,:][:-padding_len] >= threshold
            mask_end  = fin_output_end[j,:][:-padding_len] >= threshold
        else:
            mask_start = fin_output_start[j,:] >= threshold
            mask_end = fin_output_end[j,:] >= threshold
            
            
        mask = [0]*len(mask_start) 
        idx_start = np.nonzero(mask_start)[0]     
        idx_end = np.nonzero(mask_end)[0]
        
        if len(idx_start)> 0:
            idx_start = idx_start[0]
            if len(idx_end)>0:
                idx_end = idx_end[0]
            else:
                idx_end = idx_start
        else:
            idx_start = 0
            idx_end = 0
            
        for mj in range(idx_start, idx_end+1):
            mask[mj] = 1
            
            
        output_tokens = [x for p,x in enumerate(tweet_tokens.split()) if mask[p]==1]
        
        output_tokens = [x for x in output_tokens if not x in ['[CLS]','[SEP]']]
        
        final_output = ''
        for ot in output_tokens:
            if ot.startswith('##'):
                final_output = final_output + ot[2]
            elif len(ot)==1 and ot in string.punctuation:
                final_output = final_output + ot
                
            else:
                final_output = final_output + " " +ot
        final_output = final_output.strip()
        
        # if sentiment == "neutral" or len(original_tweet.split())<4:
        if sentiment == [1,0,0] or len(str(original_tweet).split())<4:
            final_output = original_tweet
        jac = jaccard(str(target_string).strip(), str(final_output).strip())
        jaccards.append(jac)
                
                
    mean_jac =  np.mean(jaccards)
            
            
        
    
      
        # jaccard_scores = []
        # for px, tweet in enumerate(orig_tweet):
        #     selected_tweet = orig_selected[px]
        #     tweet_sentiment = sentiment[px]
        #     jaccard_score = calculate_jaccard_score(
        #         original_tweet=str(tweet),
        #         target_string=str(selected_tweet),
        #         sentiment_val=tweet_sentiment,
        #         idx_start=np.argmax(outputs_start[px, :]),
        #         idx_end=np.argmax(outputs_end[px, :]),
        #         offsets_start=offsets_start[px, :],
        #         offsets_end=offsets_end[px, :]
        #     )
        #     jaccard_scores.append(jaccard_score)
        #
        # jaccards.update(np.mean(jaccard_scores), ids.size(0))
        # losses.update(loss.item(), ids.size(0))
        
    print(mean_jac)
    return mean_jac #jaccards.avg, 

@torch.no_grad()
def eval_fn(data_loader, model, device):
    model.eval()
    losses = AverageMeter()
    jaccards = AverageMeter()
    
    fin_output_start = []
    fin_output_end = []
    fin_padding_lens = []
    fin_tweet_tokens = []
    fin_orig_sentiment = []
    fin_orig_tweet = []
    fin_orig_tweet = []
    fin_orig_selected = []
    
    with torch.no_grad():
        tk0 = tqdm(data_loader, total=len(data_loader))
        for bi, d in enumerate(tk0):
            ids = d["ids"]
            token_type_ids = d["token_type_ids"]
            mask = d["mask"]
            sentiment = d["sentiment"]
            orig_selected = d["original_selected_text"]
            orig_tweet = d["original_tweet"]
            targets_start = d["targets_start"]
            padding_len = d['padding_len']
            targets_end = d["targets_end"]
            tweet_tokens = d["tweet_tokens"]
            orig_sentiment = d["original_sentiment"]
            offsets_start = d["offsets_start"].numpy()
            offsets_end = d["offsets_end"].numpy()

            ids = ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            targets_start = targets_start.to(device, dtype=torch.float)
            targets_end = targets_end.to(device, dtype=torch.float)
            
            with torch.set_grad_enabled(False):
                with torch.cuda.amp.autocast():
                    outputs_start, outputs_end = model(
                                    ids=ids,
                                    mask=mask,
                                    token_type_ids=token_type_ids,
                                                        )
                    loss = loss_fn(outputs_start, outputs_end, targets_start, targets_end)
                    
            outputs_start = torch.softmax(outputs_start, dim=1).cpu().detach().numpy()
            outputs_end = torch.softmax(outputs_end, dim=1).cpu().detach().numpy()

        
        losses.update(loss.item(), ids.size(0))
        tk0.set_postfix(loss=losses.avg)
        
        fin_output_start.append(outputs_start)
        fin_output_end.append(outputs_end)
        fin_padding_lens.extend(padding_len.cpu().detach().numpy().tolist())
        
        fin_tweet_tokens.extend(tweet_tokens)
        fin_orig_sentiment.extend(orig_sentiment)
        fin_orig_tweet.extend(orig_tweet)
        fin_orig_selected.extend(orig_selected)
        
    fin_output_start = np.vstack(fin_output_start)  
    fin_output_end = np.vstack(fin_output_end)
    threshold = 0.2
    jaccards = []
    
    for j in range(len(fin_tweet_tokens)):
        target_string = fin_orig_selected[j]
        tweet_tokens = fin_tweet_tokens[j]
        padding_len = fin_padding_lens[j]
        original_tweet = fin_orig_tweet[j]
        original_sentiment = fin_orig_sentiment[j]
        
        if padding_len>0:
            mask_start = fin_output_start[j,:][:-padding_len] >= threshold
            mask_end  = fin_output_end[j,:][:-padding_len] >= threshold
        else:
            mask_start = fin_output_start[j,:] >= threshold
            mask_end = fin_output_end[j,:] >= threshold
            
            
        mask = [0]*len(mask_start) 
        idx_start = np.nonzero(mask_start)[0]     
        idx_end = np.nonzero(mask_end)[0]
        
        if len(idx_start)> 0:
            idx_start = idx_start[0]
            if len(idx_end)>0:
                idx_end = idx_end[0]
            else:
                idx_end = idx_start
        else:
            idx_start = 0
            idx_end = 0
            
        for mj in range(idx_start, idx_end+1):
            mask[mj] = 1
            
            
        output_tokens = [x for p,x in enumerate(tweet_tokens.split()) if mask[p]==1]
        
        output_tokens = [x for x in output_tokens if not x in ['[CLS]','[SEP]']]
        
        final_output = ''
        for ot in output_tokens:
            if ot.startswith('##'):
                final_output = final_output + ot[2]
            elif len(ot)==1 and ot in string.punctuation:
                final_output = final_output + ot
                
            else:
                final_output = final_output + " " +ot
        final_output = final_output.strip()
        
        if sentiment == [1, 0, 0] or len(str(original_tweet).split())<4:
            final_output = original_tweet
        jac = jaccard(target_string.strip(), final_output.strip())
        #jaccards.update(np.mean(jaccard_scores), ids.size(0))
        jaccards.append(jac)
                
                
    mean_jac =  np.mean(jaccards)
            

    
    print(f"Jaccard = {mean_jac}")
    return mean_jac

def calculate_jaccard_score(
    original_tweet, 
    target_string, 
    sentiment_val, 
    idx_start, 
    idx_end, 
    offsets_start, 
    offsets_end,
    verbose=False):

    offsets = list(zip(offsets_start, offsets_end))
    
    if idx_end < idx_start:
        idx_end = idx_start
        
    
    filtered_output  = ""
    original_tweet_sp = " ".join(original_tweet.split())
    for ix in range(idx_start, idx_end + 1):
        
        if  (offsets[ix][0] == 0 and offsets[ix][1] == 0):
            continue
        filtered_output += original_tweet_sp[offsets[ix][0]: offsets[ix][1]]
        if (ix+1) < len(offsets) and offsets[ix][1] < offsets[ix+1][0]:
            filtered_output += " "

    filtered_output = filtered_output.replace(" .", ".")
    filtered_output = filtered_output.replace(" ?", "?")
    filtered_output = filtered_output.replace(" !", "!")
    filtered_output = filtered_output.replace(" ,", ",")
    filtered_output = filtered_output.replace(" ' ", "'")
    filtered_output = filtered_output.replace(" n't", "n't")
    filtered_output = filtered_output.replace(" 'm", "'m")
    filtered_output = filtered_output.replace(" do not", " don't")
    filtered_output = filtered_output.replace(" 's", "'s")
    filtered_output = filtered_output.replace(" 've", "'ve")
    filtered_output = filtered_output.replace(" 're", "'re")

    if sentiment_val == "neutral":
        filtered_output = original_tweet

    if sentiment_val != "neutral" and verbose == True:
        if filtered_output.strip().lower() != target_string.strip().lower():
            print("********************************")
            print(f"Output= {filtered_output.strip()}")
            print(f"Target= {target_string.strip()}")
            print(f"Tweet= {original_tweet.strip()}")
            print("********************************")

    jac = jaccard(target_string.strip(), filtered_output.strip())
    return jac

import config
import torch
import numpy as np
import pandas as pd
from utils import text_ip
from utils import nlp_aug


class TweetDataset_train:
    def __init__(self,tweet,sentiment, selected_text, aug=False):
        self.tweet = tweet
        self.sentiment = sentiment
        self.selected_text = selected_text
        self.max_len = config.MAX_LEN
        self.tokenizer = config.TOKENIZER
        self.aug = aug
        
    def __len__(self):
        return len(self.tweet)
    
    def __getitem__(self, item):
        tweet = self.tweet[item]
        selected_text = self.selected_text[item]
        
        tweet = text_ip.convert_lowercase(tweet)
        selected_text = text_ip.convert_lowercase(selected_text)
        #
        #tweet = text_ip.remove_html_tags(tweet)
        #elected_text = text_ip.remove_html_tags(selected_text)
        #
        tweet = text_ip.remove_special_char(tweet)
        selected_text = text_ip.remove_special_char(selected_text)
        #
        #
        tweet = text_ip.spelling_correct(tweet)
        selected_text = text_ip.spelling_correct(selected_text)
        #

        # tweet = text_ip.remove_stopwords(tweet)
        # selected_text = text_ip.remove_stopwords(selected_text)
        #tweet = text_ip.stemming(tweet)
        if self.aug:
            rand_num = np.random.randint(10)
            if rand_num < 6: 
                tweet = text_ip.stemming(tweet)
        #        tweet = nlp_aug.synonym_aug(tweet)[0]
                #selected_text = nlp_aug.synonym_aug(selected_text)[0]
                
        #        tweet = nlp_aug.swap_words(tweet)[0]
                #selected_text = nlp_aug.swap_words(selected_text)[0]
            #elif rand_num >=3 & rand_num <=5:
            #    review = nlp_aug.summarize(review)[0] #nlp_aug.crop_sent
            #elif rand_num >5 & rand_num <=8:
                #pass
                #tweet = nlp_aug.crop_sent(tweet)[0]
                #tweet = nlp_aug.replaces_by_context_word(tweet)[0]
                #selected_text = nlp_aug.replaces_by_context_word(selected_text)[0]
        #    else:
            #review = nlp_aug.back_translation(review)
                #tweet = nlp_aug.insert_word(tweet)[0]
            #selected_text = nlp_aug.insert_word(selected_text)[0]
            #tweet = nlp_aug.delete_random_char(tweet)[0]
            #selected_text = nlp_aug.delete_random_char(selected_text)[0]
        
        tweet = " ".join(str(tweet).split())
        selected_text = " ".join(str(selected_text).split())
        
        len_sel_text = len(selected_text)
        idx0 =-1
        idx1 =-1
        if len(selected_text)==0:
            print('FALSE')
            
        for ind in (i for i,e in enumerate(tweet) if str(e)==selected_text[0]):
            if tweet[ind:ind+len_sel_text] == selected_text:
                idx0 = ind
                idx1 = ind+len_sel_text-1
                break
            
            
        char_targets = [0]*len(tweet)
        #[0,0,0,0,0,0,0,0,0,0,0,0,0]
        if idx0 != -1 and idx1 != -1:
            for j in range(idx0,idx1+1):
                if tweet[j] != " ":
                    char_targets[j] = 1
                    
        #[0,0,1,1,1,0,1,1,1,0,0,0,0]
        
        
        #from huggingface tokenizer
        tok_tweet = self.tokenizer.encode(tweet)
        tok_tweet_tokens = tok_tweet.tokens
        tok_tweet_ids = tok_tweet.ids
        tok_tweet_offsets = tok_tweet.offsets[1:-1]  # removing [cls] and [sep]
        
        #from bert tokenizer
        # tok_tweet = self.tokenizer.encode_plus(tweet,
        #                             None,
        #                             add_special_tokens=True,
        #                             max_length=self.max_len,
        #                             pad_to_max_length=True,
        #                                                 )
        #
        # tok_tweet_tokens = tok_tweet.tokens
        # tok_tweet_ids = tok_tweet.word_ids
        # tok_tweet_offsets = tok_tweet.offsets[1:-1]  # removing [cls] and [sep]
        

        
        targets = [0]*(len(tok_tweet_tokens)-2) # as offsets are removed
        # [0,0,0,0,0,0,0]
        for j, (offset1, offset2) in enumerate(tok_tweet_offsets):
            # there is a word having even partial match
            if sum(char_targets[offset1:offset2])>0: #checking if there is a word
                targets[j]=1
            
        # [0,0,0,1,1,1,0]
        
        targets = [0] + targets + [0] # adding null tokens for [cls] and [sep]
        targets_start = [0]*len(targets)
        targets_end   = [0]*len(targets)
        
        non_zero = np.nonzero(targets)[0] # first non-zero indices
        if len(non_zero)>0:
            targets_start[non_zero[0]]=1
            targets_end[non_zero[-1]]=1
            
        mask = [1] * len(tok_tweet_ids)
        token_type_ids = [0] * len(tok_tweet_ids)
        
        #bert add padding on the right
        padding_len = self.max_len-len(tok_tweet_ids)
        ids = tok_tweet_ids + [0] * padding_len
        mask = mask + [0] * padding_len
        token_type_ids = token_type_ids + [0] * padding_len
        targets = targets + [0] * padding_len
        targets_start = targets_start + [0] * padding_len
        targets_end = targets_end + [0] * padding_len
        tweet_offsets = tok_tweet_offsets + ([(0, 0)] * padding_len)
        
        
        sentiment = [1,0,0]
        if self.sentiment[item] == 'positive':
            sentiment = [0,0,1] 
        elif self.sentiment[item] == 'negative':
            sentiment = [0,1,0] 
            
        return {
                'ids':torch.tensor(ids,dtype=torch.long),
                'mask':torch.tensor(mask,dtype=torch.long),
                'token_type_ids':torch.tensor(token_type_ids,dtype=torch.long),
                'targets':torch.tensor(targets,dtype=torch.long),
                'targets_start':torch.tensor(targets_start,dtype=torch.long),
                'targets_end':torch.tensor(targets_end,dtype=torch.long),
                'padding_len':torch.tensor(padding_len,dtype=torch.long),
                'tweet_tokens' : " ".join(tok_tweet_tokens),
                'original_tweet': self.tweet[item],
                'sentiment':torch.tensor(sentiment,dtype=torch.long),
                'original_sentiment': self.sentiment[item],
                'original_selected_text': self.selected_text[item],
                'offsets_start': torch.tensor([x for x, _ in tweet_offsets], dtype=torch.long),
                'offsets_end': torch.tensor([x for _, x in tweet_offsets], dtype=torch.long)
                }
        
        
        
class TweetDataset_val:
    def __init__(self,tweet,sentiment, selected_text):
        self.tweet = tweet
        self.sentiment = sentiment
        self.selected_text = selected_text
        self.max_len = config.MAX_LEN
        self.tokenizer = config.TOKENIZER
        
    def __len__(self):
        return len(self.tweet)
    
    def __getitem__(self, item):
        tweet = " ".join(str(self.tweet[item]).split())
        selected_text = " ".join(str(self.selected_text[item]).split())
        
        len_sel_text = len(selected_text)
        idx0 =-1
        idx1 =-1
        
        for ind in (i for i,e in enumerate(tweet) if e==selected_text[0]):
            if tweet[ind:ind+len_sel_text] == selected_text:
                idx0 = ind
                idx1 = ind+len_sel_text-1
                break
            
        char_targets = [0]*len(tweet)
        #[0,0,0,0,0,0,0,0,0,0,0,0,0]
        if idx0 != -1 and idx1 != -1:
            for j in range(idx0,idx1+1):
                if tweet[j] != " ":
                    char_targets[j] = 1
                    
        #[0,0,1,1,1,0,1,1,1,0,0,0,0]
        
        
        #from huggingface tokenizer
        tok_tweet = self.tokenizer.encode(tweet)
        tok_tweet_tokens = tok_tweet.tokens
        tok_tweet_ids = tok_tweet.ids
        tok_tweet_offsets = tok_tweet.offsets[1:-1]  # removing [cls] and [sep]
        
        #from bert tokenizer
        # tok_tweet = self.tokenizer.encode_plus(tweet,
        #                             None,
        #                             add_special_tokens=True,
        #                             max_length=self.max_len,
        #                             pad_to_max_length=True,
        #                                                 )
        #
        # tok_tweet_tokens = tok_tweet.tokens
        # tok_tweet_ids = tok_tweet.word_ids
        # tok_tweet_offsets = tok_tweet.offsets[1:-1]  # removing [cls] and [sep]
        
    
        targets = [0]*(len(tok_tweet_tokens)-2) # as offsets are removed
        # [0,0,0,0,0,0,0]
        for j, (offset1, offset2) in enumerate(tok_tweet_offsets):
            # there is a word having even partial match
            if sum(char_targets[offset1:offset2])>0: #checking if there is a word
                targets[j]=1
            
        # [0,0,0,1,1,1,0]
        
        targets = [0] + targets + [0] # adding null tokens for [cls] and [sep]
        targets_start = [0]*len(targets)
        targets_end   = [0]*len(targets)
        
        non_zero = np.nonzero(targets)[0] # first non-zero indices
        if len(non_zero)>0:
            targets_start[non_zero[0]]=1
            targets_end[non_zero[-1]]=1
            
        mask = [1] * len(tok_tweet_ids)
        token_type_ids = [0] * len(tok_tweet_ids)
        
        #bert add padding on the right
        padding_len = self.max_len-len(tok_tweet_ids)
        ids = tok_tweet_ids + [0] * padding_len
        mask = mask + [0] * padding_len
        token_type_ids = token_type_ids + [0] * padding_len
        targets = targets + [0] * padding_len
        targets_start = targets_start + [0] * padding_len
        targets_end = targets_end + [0] * padding_len
        tweet_offsets = tok_tweet_offsets + ([(0, 0)] * padding_len)
        
        
        sentiment = [1,0,0]
        if self.sentiment[item] == 'positive':
            sentiment = [0,0,1] 
        elif self.sentiment[item] == 'negative':
            sentiment = [0,1,0] 
            
        return {
                'ids':torch.tensor(ids,dtype=torch.long),
                'mask':torch.tensor(mask,dtype=torch.long),
                'token_type_ids':torch.tensor(token_type_ids,dtype=torch.long),
                'targets':torch.tensor(targets,dtype=torch.long),
                'targets_start':torch.tensor(targets_start,dtype=torch.long),
                'targets_end':torch.tensor(targets_end,dtype=torch.long),
                'padding_len':torch.tensor(padding_len,dtype=torch.long),
                'tweet_tokens' : " ".join(tok_tweet_tokens),
                'original_tweet': self.tweet[item],
                'sentiment':torch.tensor(sentiment,dtype=torch.long),
                'original_sentiment': self.sentiment[item],
                'original_selected_text': self.selected_text[item],
                'offsets_start': torch.tensor([x for x, _ in tweet_offsets], dtype=torch.long),
                'offsets_end': torch.tensor([x for _, x in tweet_offsets], dtype=torch.long)
                }
        



        
# if __name__ == '__main__':
#     df = pd.read_csv(config.TRAINING_FILE).dropna().reset_index(drop=True)
#
#     dataset = TweetDataset(tweet = df.text.values,
#                            sentiment = df.sentiment.values,
#                            selected_text = df.selected_text.values
#                            )
#
#     print(dataset[0])
#

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
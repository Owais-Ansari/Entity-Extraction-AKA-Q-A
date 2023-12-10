import os
import numpy as np
import pandas as pd

import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as nafc
from nlpaug.util import Action


#==========================================================================
#path = '/home/owaishs/temp2/Scripts/git-workspace/NLP/Bert/IMDB Dataset.csv'
#df = pd.read_csv(path)
#text = df['review'][0]

def delete_random_char(text):
    '''Delete character randomly'''
    aug = nac.RandomCharAug(action="delete")
    augmented_text = aug.augment(text)
    return augmented_text

def replaces_by_context_word(text):
    '''  
    Insert word by contextual word embeddings (BERT, DistilBERT, RoBERTA or XLNet)
    ''' 
    aug = naw.ContextualWordEmbsAug(
        model_path='bert-base-uncased', action="insert")
    augmented_text = aug.augment(text)
    return augmented_text

def antonym_aug(text):
    aug = naw.AntonymAug()
    new_text = []
    for word in text.split():
        new_text.append(word)
    total_words = len(new_text)
    indx = np.random.randint(total_words)
    _text = new_text[indx]
    augmented_text = aug.augment(_text)
    return augmented_text

def swap_words(text):
    aug = naw.RandomWordAug(action="swap")
    augmented_text = aug.augment(text)
    return augmented_text

def crop_sent(text):
    aug = naw.RandomWordAug(action='crop')
    augmented_text = aug.augment(text)
    return augmented_text

def back_translation(text):
    back_translation_aug = naw.BackTranslationAug(
        from_model_name='facebook/wmt19-en-de', 
        to_model_name='facebook/wmt19-de-en'
    )
    aug_text = back_translation_aug.augment(text)
    return aug_text
        
def synonym_aug(text):
    aug = naw.SynonymAug(aug_src='wordnet')
    augmented_text = aug.augment(text)
    return augmented_text

def summarize(text):
    aug = nas.AbstSummAug(model_path='t5-base')
    augmented_text = aug.augment(text)
    return augmented_text

def insert_word(text):
    aug = nac.RandomCharAug(action="insert")
    augmented_text = aug.augment(text)
    return augmented_text




#reveiw = text_ip.stemming(review)
# review = synonym_aug(review)
# review = swap_words(review)
# review = antonym_aug(review)
# review = crop_sent(review)
#
# review = summarize(review)
#
# review = replaces_by_context_word(review)
# review = delete_random_char(review)








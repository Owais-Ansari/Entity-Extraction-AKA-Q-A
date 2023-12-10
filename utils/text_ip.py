#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 19:06:25 2022
@author: owaishs
"""

import pandas as pd
import re
import string
from textblob import TextBlob as textblob
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
import emoji
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

#path = '/home/owaishs/temp2/Scripts/git-workspace/NLP/Bert/IMDB Dataset.csv'

#df = pd.read_csv(path)

#df = df.dropna()

def convert_lowercase(text):
    '''
    Parameters
    ----------
    column : panda DataFrame object
        Contains entries of text in each row

    Returns
    -------
    column : panda DataFrame
        Converted sentences into lowercase

    '''
    #column = column.str.lower()
    text = str(text).lower()
    return text

def remove_html_tags(text):
    '''
    Parameters
    ----------
    column : panda DataFrame object
        Contains entries of text in each row.

    Returns
    -------
    column : panda DataFrame
       html removed sentences
       
     whenever we scrapes a website for review or comments, html tags such as header, body, anchor,
     etc will be present. These tags do not add any values to the data removing these tags should be better.

    '''
    re_html = re.compile('<.*?>')
    text = re_html.sub(r'',text) #[re_html.sub(r'',text) for text in column] 
    return text

def remove_url(text):
    '''
    Parameters
    ----------
    column : panda DataFrame object
        Contains entries of text in each row.

    Returns
    -------
    column : panda DataFrame
        url removed sentences
        
    URL do not add any values to the review except it maps to a reference.

    '''
    re_url = re.compile('https?://\S+|www\.\S+')
    text = re_url.sub(r'',text)#[re_url.sub(r'',text) for text in column] 
    return text


def  remove_punc(text):
    '''
    Parameters
    ----------
    column : panda DataFrame object
       Contains entries of text in each row.
    Returns
    -------
    columns : panda DataFrame
    Punctions removed sentence
    
    It is similar to lower casing, in certain cases, hello and hello! should be treated same way.
    Word can't should not be converted into cant.

    '''
    exclude = string.punctuation
    #re_punc = text.translate(str.maketrans('','',exclude))
    text = text.translate(str.maketrans('','',exclude))#[text.translate(str.maketrans('','',exclude)) for text in column] 
    return text
    



def spelling_correct(text):
    '''
    Parameters
    ----------
    column :panda DataFrame object
        Contains entries of text in each row.

    Returns
    -------
    column : panda DataFrame
        Spelling corrected sentences.

    '''
    text = str(textblob(text)) #[str(textblob(text)) for text in column] 
    return text



def chat_words_conv(column):
    '''
    Parameters
    ----------
    column : panda DataFrame object
        Contains entries of text in each row.
    Returns
    -------
    new_column : list
        abbrivation converted into respective sentence
    '''
    chat_words = {
            'FYI':'for your reference',
            'LOL': 'laugh out load',
            'AFK': 'away from keyboard',
            }
    
    new_column = []
    for text in column:
        new_text = []
        for word in text.split():
            if word.upper() in chat_words:
                new_text.append(chat_words[word.upper()])
            else:
                new_text.append(word)
        new_column.append(' '.join(new_text))
    return new_column
    

def remove_stopwords(text):
    '''

    Parameters
    ----------
    column : panda DataFrame object
        Contains entries of text in each row.

    Returns
    -------
    new_column : list
    eg: the, an, so,
    

    '''
    stopwords_enlish = stopwords.words('english')
    new_text = []
    for word in text.split():
        if word in stopwords_enlish:
            continue
        else:
            new_text.append(word)
    new_text = ' '.join(new_text)
    return new_text


def remove_special_char(text):
    text = text.replace("\'",'')
    text = text.replace('/','')
    text = text.replace(',',' ')
    text = text.replace('.',' ')
    return text

def emoji_text(column):
    '''
    Parameters
    ----------
    column :panda DataFrame object
        Contains entries of text in each row.

    Returns
    -------
    column : panda DataFrame
        emoji converted into computer understandable text
        eg: He is suffering from fever :face_with_thermometer: 

    '''
    column = [emoji.demojize(text) for text in column] 
    return column


#
#Breaking data into smaller chunks : tokenization
#    It is used to separate the sentences, words, characters
# nltk and spacy


def stemming(text):
    '''
    Parameters
    ----------
    column :panda DataFrame object
        Contains entries of text in each row.

    Returns
    -------
    column : panda DataFrame
        eg: 'walked, walks, walk, walking' --> 'walk,walk,walk,walk'
    '''
    ps  = PorterStemmer()
    
    #new_column = []
    #for text in column:
    new_text = [ps.stem(word) for word in text.split()]
    #    new_column.append(' '.join(new_text))
    return ' '.join(new_text)#new_column


def sent_token(column):
    '''
    Parameters
    ----------
    column :panda DataFrame object
        Contains entries of text in each row.

    Returns
    -------
    column : panda DataFrame
        eg: sent_tokenize(text)
    '''
    column = [sent_tokenize(text) for text in column] 
    return column

# def word_token(column):
#     '''
#     Parameters
#     ----------
#     column :panda DataFrame object
#         Contains entries of text in each row.
#     Returns
#     -------
#     column : panda DataFrame word_tokenize(sents)
#         eg: sent_tokenize(text)
        
#     '''
#     #column = [word_tokenize(sents) for sents in (sent_tokenize(text) for text in column)] 
#     new_column = []
#     for text in column:
#         new_text = []
#         for sents in (sent_tokenize(text)):
#             new_text.append(word_tokenize(sents))
            
#         new_column.append(' '.join(new_text))
        
#     return column


# df['review'] = remove_url(df['review'])
# df['review'] = remove_html_tags(df['review'])
# #df['review'] = chat_words_conv(df['review'])
# df['review'] = convert_lowercase(df['review'])
# #should be done after tokenization
# #df['review'] = stemming(df['review'])
#
# df['review'] = remove_punc(df['review'])
# #should be done after tokenization
# df['review'] = remove_stopwords(df['review'])
# #should be done after tokenization
# df['review'] = spelling_correct(df['review'])
# df['review'] = emoji_text(df['review'])
# df['review'] = sent_token(df['review'])
# df['review'] = word_token(df['review'])









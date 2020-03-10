import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from _collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection, naive_bayes, svm

np.random.seed(500)

corpus = pd.read_csv(r"C:\Users\jlu39\Desktop\prevDev.csv", usecols=[2,9,13], encoding='latin-1')

#remove any possible blank rows and convert into lower case
corpus['text'].dropna(inplace=True)
corpus['text'] = [entry.lower() for entry in corpus['text']]
#each entry broken into set of words
corpus['text'] = [word_tokenize(entry) for entry in corpus['text']]

tag_map = defaultdict[lambda : wn.NOUN]
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV

for index,entry in enumerate(corpus['text']):
    Final_words = []
    word_lemmatized = WordNetLemmatizer()
    for word, tag in pos_tag(entry):
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_lemmatized.lemmatize(word, tag_map[tag[0]])
            Final_words.append(word_Final)
    corpus.loc[index, 'text_final'] = str(Final_words)


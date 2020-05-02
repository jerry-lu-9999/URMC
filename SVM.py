import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import json
import re
import math
from pprint import pprint as pp
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

start_time = time.time()
#WHAT DOES THIS DO EVEN
np.random.seed(500)

corpus = pd.read_csv(r"C:/Users/jlu39/Desktop/2017 test.csv",encoding='latin-1',header = 0, usecols = [2,18,22])

#remove any possible blank rows and convert into lower case
#each entry broken into set of words
# corpus['note'] = [entry.lower() for entry in corpus['note']]
# corpus['note'] = [word_tokenize(entry) for entry in corpus['note']]

master_Dict = {}
nest_dict = {}
list_for_note = []

#Todo: remove AVS, pick out the empty enc_id

pattern = re.compile('       GOLISANO|                         After Visit Summary')

for index in range(len(corpus.index)-1):
    """ a design choice here: Either NaN for the entry that doesn't have a manual Y/N, or think it as a part of other notes 
    the same patient"""
    if pd.notnull(corpus.iloc[index, corpus.columns.get_loc('Fever?')]):
        fever = corpus.iloc[index, corpus.columns.get_loc('Fever?')]
    note = corpus.iloc[index, corpus.columns.get_loc('note')]
    enc_id = corpus.iloc[index, corpus.columns.get_loc('pat_enc_csn_id')]
    next_enc_id = corpus.iloc[index+1, corpus.columns.get_loc('pat_enc_csn_id')]

    if not pattern.match(note):
        list_for_note.append(note)

    if (not enc_id == next_enc_id) or (index == len(corpus.index)-2):
        nest_dict['Fever?'] = fever
        nest_dict['note'] = "".join(str(e) for e in list_for_note)
        master_Dict[enc_id] = nest_dict
        list_for_note = []
        nest_dict = {}

df = pd.DataFrame.from_dict(master_Dict, orient='index')
df['enc_id'] = df.index

tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV

#Giving each word a tag and filtering out useless data such as :a, an
for index, entry in enumerate(df['note']):
    Final_words = []
    word_lemmatized = WordNetLemmatizer()
    entry = word_tokenize(entry.lower())
    for word, tag in pos_tag(entry):
        if word not in stopwords.words('english'):
            word_Final = word_lemmatized.lemmatize(word, tag_map[tag[0]])
            Final_words.append(word_Final)
    enc_id = df.iloc[index, df.columns.get_loc('enc_id')]
    df.loc[enc_id, 'note_final'] = " ".join(str(e) for e in Final_words)

print("--- lemmatization finished: %s seconds ---" % round(time.time() - start_time, 2))
#print(df.head(6))

#transform dataset into numerical value that the model can understand
Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(df['note_final'],df['Fever?'],test_size=0.3)
Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)
#print(Train_Y)

#word vectorization: turning text into numerical feature vectors :1. Term Frequency-Inverse Document
#max_feature means the maximum unique words/features we can have
Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(df['note_final'])
Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)

Naive = naive_bayes.MultinomialNB()
Naive.fit(Train_X_Tfidf, Train_Y)
predictions_NB = Naive.predict(Test_X_Tfidf)
print("Naive Bayes Accuracy Score -> ", accuracy_score(predictions_NB, Test_Y) * 100)

#SVM
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf, Train_Y)
print(Train_Y)
predictions_SVM = SVM.predict(Test_X_Tfidf)
print(predictions_SVM)

print("SVM Accuracy Score ->" , accuracy_score(predictions_SVM, Test_Y) * 100)
print("---%s seconds ---" % round(time.time() - start_time, 2))
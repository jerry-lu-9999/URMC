import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from timer import Timer
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn import model_selection, naive_bayes, svm
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from _collections import defaultdict

np.random.seed(500)


corpus = pd.read_csv(r"/Users/jiahaolu/Desktop/2017 Master Notes Data.12.9.19.csv", 
                        header = 0,
                        usecols = ['emrn', 'Fever?', 'pat_enc_csn_id', 'note'])
print(corpus.head(5))
print(type(corpus['Fever?']))
for index, entry in enumerate(corpus['emrn']):
    if corpus['emrn'][index] == corpus['emrn']['index']:
        print("yess it matches")
    else:
        print(" no it doesn't")

#remove any possible blank rows and convert into lower case
corpus['note'].dropna(inplace=True)
corpus['note'] = [entry.lower() for entry in corpus['note']]
#each entry broken into set of words
corpus['note'] = [word_tokenize(entry) for entry in corpus['note']]

print(corpus.head(5))

#pos tag for words, default words is a NOUN
tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV

for index,entry in enumerate(corpus['note']):
    Final_words = []
    word_lemmatized = WordNetLemmatizer()
    for word, tag in pos_tag(entry):
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_lemmatized.lemmatize(word, tag_map[tag[0]])
            Final_words.append(word_Final)
    corpus.loc[index, 'note_final'] = str(Final_words)

Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(corpus['note_final'], 
                                                                    corpus['Fever?'], 
                                                                    test_size = 0.3)
print(Train_Y)
# Encoder = LabelEncoder()
# #fit label encoder and return encoded labels
# #Y in our case is fever or not(T/F), X in our case is the doctor's notes
# Train_Y = Encoder.fit_transform(Train_Y)
# Test_Y = Encoder.fit_transform(Test_Y)

# #Using TF-IDF    Term frequency-Inverse Document Frequency    Turning text data to vectors
# #Tfidf_vect = TfidfVectorizer(max_features=5000)
# #Tfidf_vect.fit(corpus['note_final'])
# Hash_vect = HashingVectorizer()
# Hash_vect.fit(corpus['note_final'])

# # Train_X_Tfidf = Tfidf_vect.transform(Train_X)
# # Test_X_Tfidf = Tfidf_vect.transform(Test_X)

# Train_X_Tfidf = Hash_vect.transform(Train_X)
# Test_X_Tfidf = Hash_vect.transform(Test_X)
# print(Test_X.shape)
# # print(Tfidf_vect.vocabulary_)
# # print(Hash_vect.vocabulary_)

# #SVM
# SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
# SVM.fit(Train_X_Tfidf, Train_Y)

# predictions_SVM = SVM.predict(Test_X_Tfidf)

# print("SVM Accuracy Score ->" , accuracy_score(predictions_SVM, Test_Y) * 100)

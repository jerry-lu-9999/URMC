import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import re
import math
from sympy import symbols, Eq, solve
from pprint import pprint as pp
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from _collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, plot_confusion_matrix, confusion_matrix, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection, naive_bayes, svm, datasets
from sklearn.model_selection import GridSearchCV, learning_curve

start_time = time.time()
# WHAT DOES THIS DO EVEN
np.random.seed(500)

corpus2017 = pd.read_csv(r"C:/Users/jlu39/Desktop/2017 Master Notes Data.12.9.19.csv", encoding='latin-1', header=0,
                     usecols=[2, 18, 22])
corpus2016 = pd.read_csv(r"C:/Users/jlu39/Desktop/2016 Notes Data.csv", encoding='latin-1', header=0,
                     usecols=[2, 9, 13])
corpus_list = [corpus2016, corpus2017]
master_dict_list = []

seen = False
pattern = re.compile('       GOLISANO|                         After Visit Summary')

for corpus in corpus_list:
    master_dict = {}
    nest_dict = {}
    list_for_note = []
    list_for_empty_diag = []
    for index in range(len(corpus.index) - 1):
        note = corpus.iloc[index, corpus.columns.get_loc('note')]
        enc_id = corpus.iloc[index, corpus.columns.get_loc('pat_enc_csn_id')]
        next_enc_id = corpus.iloc[index + 1, corpus.columns.get_loc('pat_enc_csn_id')]

        if pd.notnull(corpus.iloc[index, corpus.columns.get_loc('Fever?')]):
            fever = corpus.iloc[index, corpus.columns.get_loc('Fever?')]
        else:
            if not list_for_note and seen is False:
                list_for_empty_diag.append(enc_id)

        if not pattern.match(note):
            list_for_note.append(note)
        else:
            seen = True

        if (not enc_id == next_enc_id) or (index == len(corpus.index) - 2):
            if enc_id in master_dict.keys():
                nest_dict['Fever?'] = fever
                nest_dict['note'] = master_dict[enc_id]['note'] + "".join(str(e) for e in list_for_note)
                master_dict[enc_id] = nest_dict
                if enc_id in list_for_empty_diag:
                    list_for_empty_diag.remove(enc_id)
            else:
                nest_dict['Fever?'] = fever
                nest_dict['note'] = "".join(str(e) for e in list_for_note)
                master_dict[enc_id] = nest_dict
            list_for_note = []
            nest_dict = {}
            seen = False
    master_dict_list.append(master_dict)

print("the size of 2016 is ", len(master_dict_list[0]))

df_list = []
for master_dict in  master_dict_list:
    df = pd.DataFrame.from_dict(master_dict, orient='index')
    df['enc_id'] = df.index
    df_list.append(df)

tag_map = defaultdict(lambda: wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV

# Giving each word a tag and filtering out useless data such as :a, an
for df in df_list:
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

print(len(df_list))
# transform dataset into numerical value that the model can understand
df2017 = df_list[1]
df2016 = df_list[0]

Train_X= df2017['note_final']
Train_Y = df2017['Fever?']
Test_X = df2016['note_final']
Test_Y = df2016['Fever?']
Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)

# word vectorization: turning text into numerical feature vectors
# Term Frequency-Inverse Document Frequency
# number of times a word appear in a given sentence -- log to the base e of number of the total documents divided by the documents in which the word appears.
# max_feature means the maximum unique words/features we can have
Tfidf_vect = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
Tfidf_vect.fit(df2017['note_final'])
Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)


def reverse_confusion_matrix(index, accuracy, precision, recall, total_size):
    TN, TP, FP, FN = symbols('TN TP FP FN')
    print(accuracy[index])
    print(precision[index])
    print(recall[index])
    print(total_size[index])

    #   Eq(lhs, rhs)
    eq1 = Eq(TP - precision[index] * TP, precision[index] * FP)
    eq2 = Eq(TP - recall[index] * TP, recall[index] * FN)
    eq3 = Eq(FN + FP, (1 - accuracy[index]) * total_size[index])
    eq4 = Eq(TN + TP, accuracy[index] * total_size[index])
    sol = solve((eq1, eq2, eq3, eq4), (TN, TP, FP, FN))

    print(sol)


"""
Second plot shows the time required to train each sizes of the training dataset
Third plot shows how much time required to train each training sizes
"""


def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None, n_jobs=None,
                        train_sizes=np.linspace(0.1, 1.0, 5)):
    if axes is None:
        _, axes = plt.subplot(1, 3, figsize=(20, 5))
    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training Samples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, accuracy_test_scores, fit_times, _ = learning_curve(estimator, X, y, cv=cv,
                                                                                   n_jobs=n_jobs,
                                                                                   scoring='accuracy',
                                                                                   train_sizes=train_sizes,
                                                                                   return_times=True)
    _, _, precision_test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                                                 scoring='precision', train_sizes=train_sizes)
    _, _, recall_test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                                              scoring='recall', train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    accuracy_test_scores_mean = np.mean(accuracy_test_scores, axis=1)  # cross validation score
    accuracy_test_scores_std = np.std(accuracy_test_scores, axis=1)
    precision_test_scores_mean = np.mean(precision_test_scores, axis=1)
    precision_test_scores_std = np.std(precision_test_scores, axis=1)
    recall_test_scores_mean = np.mean(recall_test_scores, axis=1)
    recall_test_scores_std = np.std(recall_test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, accuracy_test_scores_mean - accuracy_test_scores_std,
                         accuracy_test_scores_mean + accuracy_test_scores_std, alpha=0.1,
                         color="b")
    axes[0].fill_between(train_sizes, precision_test_scores_mean - precision_test_scores_std,
                         precision_test_scores_mean + precision_test_scores_std, alpha=0.1,
                         color='y')
    axes[0].fill_between(train_sizes, recall_test_scores_mean - recall_test_scores_std,
                         recall_test_scores_mean + recall_test_scores_std, alpha=0.1,
                         color='g')
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, accuracy_test_scores_mean, 'o-', color="b",
                 label="Accuracy score")
    axes[0].plot(train_sizes, precision_test_scores_mean, 'o-', color='y',
                 label='precision score')
    axes[0].plot(train_sizes, recall_test_scores_mean, 'o-', color='g',
                 label='recall score')
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, accuracy_test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, accuracy_test_scores_mean - accuracy_test_scores_std,
                         accuracy_test_scores_mean + accuracy_test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Accuracy Score")
    axes[2].set_title("Performance of the model")

    return accuracy_test_scores_mean, precision_test_scores_mean, recall_test_scores_mean, train_sizes, plt


fig, axes = plt.subplots(3, 1, figsize=(10, 15), constrained_layout=True)
title = r"Learning curve SVM"
estimator = svm.SVC(C=10, kernel="sigmoid", gamma='scale')
acc, pre, rec, train_sizes_abs, plt = plot_learning_curve(estimator, title, Train_X_Tfidf, Train_Y, axes,
                                                          ylim=(0.5, 1.01), cv=5, n_jobs=-1)
plt.show(block=False)

for index in range(len(train_sizes_abs)):
    reverse_confusion_matrix(index, acc, pre, rec, train_sizes_abs)

param_grid = {'C': [0.1, 1, 10, 50],
              'kernel': ['rbf', 'poly', 'sigmoid'],
              'degree': [2, 3, 4],
              'gamma': ['scale']}

# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV
# search = GridSearchCV(svm.SVC(), param_grid=param_grid, n_jobs=-1, verbose=2)
SVM = svm.SVC(C=10.0, kernel='sigmoid', degree=2, gamma='scale')
SVM.fit(Train_X_Tfidf, Train_Y)
predictions_SVM = SVM.predict(Test_X_Tfidf)

"""
Kernel functions give us similarity scores from higher dimensions
The higher gamma is, the more likely overfitting occurs
                     predicted                      
		             negative	positive
True       negative	   TN       FP
           positive    FN       TP
C: the penalty parameter, higher it is, the finer the boundaries between classification is

Recall/Sensitivity = TP / (TP + FN)
Precision = TP / (TP + FP)
Accuracy = (TP + TN) / (ALL FOUR)

TN = 984, FP = 40, FN = 24, TP = 181
SVM Accuracy Score -> 94.79251423921887
---5631.23 seconds ---
"""

# https://scikit-learn.org/stable/modules/model_evaluation.html#confusion-matrix
tn, fp, fn, tp = confusion_matrix(predictions_SVM, Test_Y).ravel()
print("TN = %d, FP = %d, FN = %d, TP = %d" % (tn, fp, fn, tp))

print("SVM Accuracy Score ->", accuracy_score(predictions_SVM, Test_Y) * 100)
print("---%s seconds ---" % round(time.time() - start_time, 2))

plt.show()

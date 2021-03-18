import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from _collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection, svm
from sklearn.model_selection import GridSearchCV


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


np.random.seed(500)
corpus = pd.read_excel("C:/Users/jlu39/Desktop/2017 Master Notes Data. ED, Admission Notes. 8.20.xlsx",
                       sheet_name="ED, H&P Notes Only",
                       header=0,
                       usecols="D,T,X")

master_dict = {}
nest_dict = {}
list_for_note = []
list_for_missing_fever = []
fever = None

for index in range(len(corpus.index) - 1):
    if pd.notnull(corpus.iloc[index, corpus.columns.get_loc('pat_enc_csn_id')]):
        enc_id = corpus.iloc[index, corpus.columns.get_loc('pat_enc_csn_id')]
    else:
        continue
    note = corpus.iloc[index, corpus.columns.get_loc('note')]
    next_enc_id = corpus.iloc[index + 1, corpus.columns.get_loc('pat_enc_csn_id')]

    if pd.notnull(corpus.iloc[index, corpus.columns.get_loc('Fever?')]):
        fever = corpus.iloc[index, corpus.columns.get_loc('Fever?')]

    # ignore check for AVS cuz it is clean data already
    list_for_note.append(note)

    if (not enc_id == next_enc_id) or (index == len(corpus.index) - 2):
        if fever is None:
            list_for_missing_fever.append(enc_id)

        nest_dict['Fever?'] = fever
        if enc_id in master_dict.keys():
            nest_dict['note'] = master_dict[enc_id]['note'] + "".join(str(e) for e in list_for_note)
            master_dict[enc_id] = nest_dict
        else:
            nest_dict['note'] = "".join(str(e) for e in list_for_note)
            master_dict[enc_id] = nest_dict
        list_for_note = []
        nest_dict = {}

df = pd.DataFrame.from_dict(master_dict, orient='index')
df['enc_id'] = df.index

tag_map = defaultdict(lambda: wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV

# Giving each word a tag and filtering out useless data such as :a, an
for index, entry in enumerate(df['note']):
    Final_words = []
    word_lemmatized = WordNetLemmatizer()
    try:
        entry = word_tokenize(str(entry).lower())
    except AttributeError:
        continue
    for word, tag in pos_tag(entry):
        if word not in stopwords.words('english'):
            word_Final = word_lemmatized.lemmatize(word, tag_map[tag[0]])
            Final_words.append(word_Final)
    enc_id = df.iloc[index, df.columns.get_loc('enc_id')]
    df.loc[enc_id, 'note_final'] = " ".join(str(e) for e in Final_words)

list_of_ids = df['enc_id'].tolist()

Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(df['note_final'], df['Fever?'], test_size=0.3)
Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y.astype(str))
Test_Y = Encoder.fit_transform(Test_Y.astype(str))

Tfidf_vect = TfidfVectorizer(max_features=5000, )
Tfidf_vect.fit(df['note_final'])
Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)

# fig, axes = plt.subplots(3, 1, figsize=(10, 15), constrained_layout=True)
# title = r"Learning curve SVM"
# estimator = svm.SVC(C=10.0, kernel="sigmoid", gamma='scale')
# acc, pre, rec, train_sizes_abs, plt = plot_learning_curve(estimator, title, Train_X_Tfidf, Train_Y, axes,
#                                                           ylim=(0.5, 1.01), cv=5, n_jobs=-1)
# plt.show(block=False)

# learning curve
# param_grid = {'C': [0.1, 1, 10, 50],
#               'kernel': ['rbf', 'poly', 'sigmoid'],
#               'degree': [2, 3, 4],
#               'gamma': ['scale']}
# search = GridSearchCV(svm.SVC(), param_grid = param_grid, n_jobs=-1, verbose=2)
SVM = svm.SVC(C=10.0, kernel='sigmoid', degree=2, gamma='scale')
# search.fit(Train_X_Tfidf, Train_Y)
SVM.fit(Train_X_Tfidf, Train_Y)
predictions_SVM = SVM.predict(Test_X_Tfidf)
# print(search.best_estimator_)
# print(predictions_SVM)
# print(Test_Y)
# print(confusion_matrix(predictions_SVM, Test_Y))
tn, fp, fn, tp = confusion_matrix(predictions_SVM, Test_Y).ravel()
print("TN = %d, FP = %d, FN = %d, TP = %d" % (tn, fp, fn, tp))

list_fp = []
list_fn = []
TP = 0; TN = 0; FP = 0; FN = 0

for i in range(len(predictions_SVM)):
    if predictions_SVM[i] == 1 and Test_Y[i] == 1:
        TP += 1
    elif predictions_SVM[i] == 0 and Test_Y[i] == 0:
        TN += 1
    elif predictions_SVM[i] == 1 and Test_Y[i] == 0:
        FP += 1
        list_fp.append(list_of_ids[i])
    elif predictions_SVM[i] == 0 and Test_Y[i] == 1:
        FN += 1
        list_fn.append(list_of_ids[i])
print("SVM Accuracy Score ->", accuracy_score(predictions_SVM, Test_Y) * 100)
print("Manual TN = %d, FP = %d, FN = %d, TP = %d" % (TN, FP, FN, TP))
print(list_fp)
print(list_fn)

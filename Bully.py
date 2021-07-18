# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 12:46:11 2020

@author: shovon5795
"""

import pandas as pd
import re


dataset = pd.read_csv(r"C:\Users\shovon5795\Desktop\Research\Saeed Proshun\Bully\Dataset.csv")


import string

#Preprocessing Step
#Punctuation Removal
other_punct = [',', '.', '"', ':', ')', '(', '!', '?', '|', ';', "'", '$', '&',
    '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£',
    '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',
    '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', '“', '★', '”',
    '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾',
    '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', '▒', '：', '¼', '⊕', '▼',
    '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲',
    'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', '∙', '）', '↓', '、', '│', '（', '»',
    '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø',
    '¹', '≤', '‡', '√', '«', '»', '´', 'º', '¾', '¡', '§', '£', '₤', 'Ã' 'Å', '•', 'Ã³']

def noise_clean(tweet):
    tweet = re.sub("@[A-Za-z0-9]+","",tweet) #Remove @ sign
    tweet = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", tweet) #Remove http links
    #tweet = re.sub("^\d+\s|\s\d+\s|\s\d+$", " ", tweet)
    tweet = re.sub(r"$\d+\W+|\b\d+\b|\W+\d+$", "", tweet)
    tweet = " ".join(tweet.split())
    return tweet
dataset["News"] = dataset["News"].map(lambda x: noise_clean(x))



def remove_reg_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text
dataset["News"] = dataset["News"].apply(remove_reg_punctuations)

def remove_other_punct(text):
    for punct in other_punct:
        text = text.replace(punct, '')
    return text
dataset["News"] = dataset["News"].apply(remove_other_punct)

#Lowercase Convertion
dataset["News"] = dataset["News"].str.lower()

#Tokenization
from nltk.tokenize import word_tokenize
dataset["News"] = dataset["News"].apply(word_tokenize)

#dataset['News'] = dataset['News'].apply(lambda x: ' '.join([w for w in x if len(w)>2]))

#Stemming
#from nltk.stem import PorterStemmer
#stemmer = PorterStemmer()

#Snowball Stemmer
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")

dataset["News"] = dataset["News"].apply(lambda x: [stemmer.stem(y) for y in x])

#dataset = dataset.drop(columns = ['Stemmed'])

#removal of stopwords
from nltk.corpus import stopwords
stopwords = stopwords.words('english')

dataset["News"] = dataset["News"].apply(lambda x: [word for word in x if not word in stopwords])


#dataset = dataset.drop(columns = ['Remove_Stop'])

#dataset["News"] = dataset["News"].apply(lambda x: [" ".join(y) for y in x])

#Coverting to String again from token
from nltk.tokenize.moses import MosesDetokenizer
detokenizer = MosesDetokenizer()

dataset['News']=dataset['News'].apply(lambda x: detokenizer.detokenize(x, return_str=True))



#Write clean data to csv file
clean_text = dataset.to_csv(r"C:\Users\shovon5795\Desktop\Research\Saeed Proshun\Bully\cleaned_news.csv", index = False)

X = dataset.iloc[:,1]
y = dataset.iloc[:,0].values

#Test-Train Set Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.2)

#Feature Extraction Step (BoW and TFID[word, n-gram, char])
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
count_vect = CountVectorizer(max_features = 5000)
X_train = count_vect.fit_transform(X_train)
X_test = count_vect.transform(X_test)
X1 = count_vect.fit_transform(X) 

tfidw = TfidfVectorizer(analyzer = 'word', max_features = 5000)
X_train = tfidw.fit_transform(X_train)
X_test = tfidw.transform(X_test)

tfidng = TfidfVectorizer(analyzer = 'word', ngram_range = (2,2), max_features = 5000)
X_train = tfidng.fit_transform(X_train)
X_test = tfidng.transform(X_test)

tfidc = TfidfVectorizer(analyzer = 'char', ngram_range = (2,3), max_features = 5000)
X_train = tfidc.fit_transform(X_train)
X_test = tfidc.transform(X_test)

#Prediction based on different classifier

#Multinomial Naive Bayes
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)

#Logistic Regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = 'liblinear', multi_class = 'ovr')
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

#Decision Tree
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)

#SVM
from sklearn.svm import LinearSVC
svc = LinearSVC()
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)

'''
#K-NN
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(X_train, y_train)
y_pred = neigh.predict(X_test)
'''

#Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()
gbc.fit(X_train, y_train)
y_pred = gbc.predict(X_test)

#Adaboost
from sklearn.ensemble import AdaBoostClassifier
abc = AdaBoostClassifier(learning_rate = 0.25)
abc.fit(X_train, y_train)
y_pred = abc.predict(X_test)

#Bagging
from sklearn.ensemble import BaggingClassifier
bg = BaggingClassifier(base_estimator = LinearSVC())
bg.fit(X_train, y_train)
y_pred = bg.predict(X_test)


#Two-Level Voting Classifier
from sklearn.ensemble import VotingClassifier
estimator3 = []
estimator3.append(('MNB',  MultinomialNB()))
estimator3.append(('DT', DecisionTreeClassifier()))
estimator3.append(('LR', LogisticRegression(solver = 'liblinear', multi_class = 'ovr')))
estimator3.append(('LSVC', LinearSVC()))
estimator3.append(('GDB', GradientBoostingClassifier()))
estimator3.append(('ABC', AdaBoostClassifier(learning_rate = 0.25)))
estimator3.append(('BAG', BaggingClassifier(base_estimator = LinearSVC())))

#Level 1
estimator = []
estimator.append(('MNB',  MultinomialNB()))
estimator.append(('DT', DecisionTreeClassifier()))
estimator.append(('LR', LogisticRegression(solver = 'liblinear', multi_class = 'ovr')))
estimator.append(('LSVC', LinearSVC()))

#Level 2
estimator2 =[]
estimator2.append(('GDB', GradientBoostingClassifier()))
estimator2.append(('ABC', AdaBoostClassifier(learning_rate = 0.25)))
estimator2.append(('BAG', BaggingClassifier(base_estimator = LinearSVC())))

estimatorl1 = []
estimatorl1.append(('VC1', VotingClassifier(estimators = estimator, voting ='hard')))
estimatorl1.append(('VC2', VotingClassifier(estimators = estimator2, voting ='hard')))


vhl3 = VotingClassifier(estimators = estimatorl1, voting ='hard') 
vhl3 = VotingClassifier(estimators = estimator3, voting ='hard') 
vhl3.fit(X_train, y_train) 
y_pred = vhl3.predict(X_test) 

#Performance Measurement
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
auc = roc_auc_score(y_pred, y_test)
acc = accuracy_score(y_pred, y_test)
cr = classification_report(y_pred, y_test)

from sklearn.model_selection import KFold, StratifiedKFold, ShuffleSplit, StratifiedShuffleSplit, cross_val_score
kf = KFold(n_splits = 10)
skf = StratifiedKFold(n_splits = 10)
shs = ShuffleSplit(n_splits = 10)
sshs = StratifiedShuffleSplit(n_splits = 10)

X1 = tfidng.fit_transform(X)

acc1 = cross_val_score(vhl3, X1, y, scoring = 'accuracy', cv=kf, n_jobs=1)
acc2 = cross_val_score(vhl3, X1, y, scoring = 'accuracy', cv=skf, n_jobs=1)
acc3 = cross_val_score(vhl3, X1, y, scoring = 'accuracy', cv=shs, n_jobs=1)
acc4 = cross_val_score(vhl3, X1, y, scoring = 'accuracy', cv=sshs, n_jobs=1)

prec1 = cross_val_score(vhl3, X1, y, scoring = 'precision', cv=kf, n_jobs=1)
prec2 = cross_val_score(vhl3, X1, y, scoring = 'precision', cv=skf, n_jobs=1)
prec3 = cross_val_score(vhl3, X1, y, scoring = 'precision', cv=shs, n_jobs=1)
prec4 = cross_val_score(vhl3, X1, y, scoring = 'precision', cv=sshs, n_jobs=1)

rec1 = cross_val_score(vhl3, X1, y, scoring = 'recall', cv=kf, n_jobs=1)
rec2 = cross_val_score(vhl3, X1, y, scoring = 'recall', cv=skf, n_jobs=1)
rec3 = cross_val_score(vhl3, X1, y, scoring = 'recall', cv=shs, n_jobs=1)
rec4 = cross_val_score(vhl3, X1, y, scoring = 'recall', cv=sshs, n_jobs=1)

f1 = cross_val_score(vhl3, X1, y, scoring = 'f1', cv=kf, n_jobs=1)
f2 = cross_val_score(vhl3, X1, y, scoring = 'f1', cv=skf, n_jobs=1)
f3 = cross_val_score(vhl3, X1, y, scoring = 'f1', cv=shs, n_jobs=1)
f4 = cross_val_score(vhl3, X1, y, scoring = 'f1', cv=sshs, n_jobs=1)

auc1 = cross_val_score(vhl3, X1, y, scoring = 'roc_auc_score', cv=kf, n_jobs=1)
auc2 = cross_val_score(vhl3, X1, y, scoring = 'roc_auc_score', cv=skf, n_jobs=1)
auc3 = cross_val_score(vhl3, X1, y, scoring = 'roc_auc_score', cv=shs, n_jobs=1)
auc4 = cross_val_score(vhl3, X1, y, scoring = 'roc_auc_score', cv=sshs, n_jobs=1)














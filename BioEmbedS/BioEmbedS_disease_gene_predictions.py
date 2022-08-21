#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import fasttext
import fasttext.util
import csv
import numpy as np
import random
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
    
model = fasttext.load_model("BioWordVec_PubMed_MIMICIII_d200.bin")


# In[4]:


def transform_X_values(pair_list):
    X_pairs = []
    for pair in pair_list:
        #this needs to be modified as per biobert
        np1 = model.get_word_vector(pair[0])
        np2 = model.get_word_vector(pair[1])
        X_pairs.append(np.concatenate([np1,np2]))
    return np.array(X_pairs)


# In[5]:


#Data preparation

true_associations = []
false_associations = []
lst1 = set()
lst2 = set()
with open('./EUADR_target_disease.csv',encoding= 'unicode_escape')as f:
    data = csv.reader(f)
    next(data)
    for row in data:
        typ = row[0].split("\t")[0]
        if typ == 'PA' or typ == 'SA':
            true_associations.append((row[0].split("\t")[3].strip('\"'),row[0].split("\t")[7].strip('\"')))
            lst1.add(row[0].split("\t")[3].strip('\"'))
            lst2.add(row[0].split("\t")[7].strip('\"'))
        elif typ == 'FA':
            false_associations.append((row[0].split("\t")[3].strip('\"'),row[0].split("\t")[7].strip('\"')))
            lst1.add(row[0].split("\t")[3].strip('\"'))
            lst2.add(row[0].split("\t")[7].strip('\"'))
            
X_true = transform_X_values(true_associations)
X_false = transform_X_values(false_associations)
X = np.concatenate([X_true, X_false])
y_true = np.ones((X_true.shape[0],), dtype=int)
y_false = np.zeros((X_false.shape[0],), dtype=int)
y= np.concatenate([y_true, y_false])

print(len(true_associations))
print(len(false_associations))
#print(X.shape)
#print(y.shape)
print(len(lst1))
print(len(lst2))
print(len(y_true))
print(len(y_false))
print(len(y_true)/(len(y_true)+len(y_false)))


# In[6]:


#Nested 10-fold cross-validation

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import average_precision_score

prec_scr = []
rec_scr = []
f1_scr = []
split_strat = StratifiedShuffleSplit(n_splits=10, test_size=0.3, random_state=42)
i = 0
f, axes = plt.subplots(figsize=(7,7))

y_real = []
y_proba = []
for train_index,test_index in split_strat.split(X,y):
    print("results for iteration: "+str(i))
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    rbf_classifier = SVC(kernel='poly',C=1.0,degree=3)
    rbf_classifier.fit(X_train, y_train)
    y_pred_test = rbf_classifier.predict(X_test)
    y_pred_proba = rbf_classifier.decision_function(X_test)
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    lab = 'Fold %d AUC=%.4f' % (i+1, auc(recall, precision))
    axes.step(recall, precision, label=lab)
    axes.plot(recall_score(y_test, y_pred_test),precision_score(y_test, y_pred_test),marker='x',mew=2,color='black')
    #axes.step(recall, precision)
    y_real.append(y_test)
    y_proba.append(y_pred_proba)
    prec_scr.append(precision_score(y_test, y_pred_test))
    rec_scr.append(recall_score(y_test, y_pred_test))
    f1_scr.append(f1_score(y_test, y_pred_test))
    print(precision_score(y_test, y_pred_test))
    print(recall_score(y_test, y_pred_test))
    print(f1_score(y_test, y_pred_test))
    print("\n")
    i += 1

y_real = np.concatenate(y_real)
y_proba = np.concatenate(y_proba)
no_skill = len(y[y==1]) / len(y)
plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='Random')
precision, recall, _ = precision_recall_curve(y_real, y_proba)
lab = 'Overall AUC=%.4f' % (auc(recall, precision))
axes.step(recall, precision, label=lab, lw=2, color='black')
axes.set_xlabel('Recall')
axes.set_ylabel('Precision')
axes.legend(loc='lower right', fontsize='small')
axes.set_title("Precision Recall curve for EUADR Dataset")


# In[7]:


print(np.mean(prec_scr))
print(np.mean(rec_scr))
print(np.mean(f1_scr))


# In[8]:


print(np.std(prec_scr))
print(np.std(rec_scr))
print(np.std(f1_scr))


# In[ ]:





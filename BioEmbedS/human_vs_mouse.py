#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import csv
import numpy as np
import pandas as pd
import random
import fasttext
import math
import operator
import itertools
import pickle
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from collections import Counter
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import MinMaxScaler
from joblib import Parallel, delayed
from statistics import mean
model = fasttext.load_model("BioWordVec_PubMed_MIMICIII_d200.bin")

#with open('./BioEmbedS_dummy/dataset/hgv1_hormone_genes.json') as json_file:
with open('./dataset/hgv1_hormone_genes.json') as json_file:
    hormone_genes = json.load(json_file)
    
alias_embeddings = dict()
for hormone in hormone_genes.keys():
    if "/" in hormone:
        parts = hormone.split("/")
        w1 = model.get_word_vector(parts[0])
        w2 = model.get_word_vector(parts[1])
        alias_embeddings[hormone] = np.add(w1,w2)
        
def transform_X_values(hormone, gene_list):
    embeddings = []
    pairs = []
    if "/" in hormone:
        np1 = alias_embeddings[hormone]
    else:
        np1 = model.get_word_vector(hormone)
    for gene in gene_list:
        np2 = model.get_word_vector(gene)
        embeddings.append(np.concatenate([np1,np2]))
        pairs.append((hormone,gene))
    return np.array(embeddings), pairs


# In[ ]:


#with open('./dataset/hgv1_hormone_gene_bins.json') as json_file:
with open('./dataset/hgv1_hormone_gene_bins.json') as json_file:
    hormone_gene_bins = json.load(json_file)
    
#with open('./BioEmbedS_dummy/dataset/hgv1_negative_hormone_gene_bins.json') as json_file:
with open('./dataset/hgv1_hormone_gene_bins.json') as json_file:
    neg_hormone_gene_bins = json.load(json_file)
    

hormone_results = dict()
for hormone in hormone_genes.keys():
    hormone_results[hormone] = [0,0]
    
for i in range(5):
    X_train = np.load('./dataset/bioembeds_X_train_val_fold_'+str(i)+'.npy')
    y_train = np.load('./dataset/bioembeds_y_train_val_fold_'+str(i)+'.npy')
    param = {}
    param['C'] = 1.0
    param['degree'] = 3
    param['kernel'] = 'poly'
    param['probability'] = True
    classifier = SVC()
    classifier.set_params(**param)
    classifier.fit(X_train,y_train)
    test_data = hormone_gene_bins[str(i)]
    neg_test_data = neg_hormone_gene_bins[str(i)]
    #with open('.BioEmbedS_dummy/dataset/train_val_marking_fold_'+str(i)+'.json') as json_file:
    with open('./dataset/train_val_marking_fold_'+str(i)+'.json') as json_file:
        train_marked = json.load(json_file)
        
    for hormone in hormone_genes.keys():
        if hormone in train_marked.keys() and train_marked[hormone] == 1:
            X_test_pos,pairs = transform_X_values(hormone,test_data[hormone])
            X_test_neg,pairs = transform_X_values(hormone,neg_test_data[hormone])
            X_test = np.concatenate([X_test_pos,X_test_neg])
            y_test_pos = np.ones((X_test_pos.shape[0],), dtype=int)
            y_test_neg = np.zeros((X_test_pos.shape[0],), dtype=int)
            y_test = np.concatenate([y_test_pos, y_test_neg])
            y_pred_test = classifier.predict(X_test)
            #print(confusion_matrix(y_test, y_pred_test))
            tp = confusion_matrix(y_test, y_pred_test)[0][0]
            fp = confusion_matrix(y_test, y_pred_test)[0][1]
            fn = confusion_matrix(y_test, y_pred_test)[1][0]
            tn = confusion_matrix(y_test, y_pred_test)[1][1]
            hormone_results[hormone][0] += tp + tn
            hormone_results[hormone][1] += tp + tn + fn + fp
            
            
for hormone in hormone_results.keys():
    print("for hormone: "+hormone+" accuracy: "+str(hormone_results[hormone][0]/hormone_results[hormone][1]))


# In[2]:


import csv

x_pts = []
y_pts = []
x1_pts = []
y1_pts = []
with open('./dataset/human_vs_mouse_stats_v1.csv','rt')as f:
    data = csv.reader(f)
    next(data)
    for row in data:
        #print(row)
        a = int(row[1])
        b = int(row[3])
        a_n_b = int(row[5])
        a1 = int(row[2])
        b1 = int(row[4])
        a1_n_b1 = int(row[6])
        
        #print(a)
        #print(b)
        #print(a_n_b)
        #print("\n")
        x_pts.append(a_n_b/(a + b- a_n_b))
        x1_pts.append(a1_n_b1/(a1 + b1- a1_n_b1))
        y_pts.append(float(row[9]))
        y1_pts.append(float(row[10]))
        
print(x_pts)
print(y_pts)
print("\n")
print(x1_pts)
print(y1_pts)


# In[4]:


import matplotlib.pyplot as plt
plt.scatter(x_pts, y1_pts, alpha=1)
#plt.title('Scatter plot for mouse predictions')
plt.xlabel('Similarity between positive mouse genes and human genes')
plt.ylabel('Accuracy on HGv1.mouse dataset')
plt.savefig('scatter_accuracy_human_vs_mouse.png')
plt.show()


# In[3]:


import matplotlib.pyplot as plt
plt.scatter(x1_pts, y1_pts, alpha=1)
#plt.title('Scatter plot for mouse predictions')
plt.xlabel('Similarity between positive mouse genes and positive human genes')
plt.ylabel('Recall on Mouse dataset')
plt.savefig('human_vs_mouse_recall_scatter.png')
plt.show()


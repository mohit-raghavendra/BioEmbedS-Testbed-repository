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


# In[5]:


with open('./dataset/hgv1_source_hormone_gene_bins.json') as json_file:
    src_hormone_gene_bins = json.load(json_file)
    
with open('./dataset/hgv1_target_hormone_gene_bins.json') as json_file:
    tgt_hormone_gene_bins = json.load(json_file)
    
print(len(src_hormone_gene_bins.keys()))
print(len(tgt_hormone_gene_bins.keys()))

for bin in src_hormone_gene_bins.keys():
    print("bin: "+str(bin)+" count: "+str(len(src_hormone_gene_bins[bin])))
    
for bin in tgt_hormone_gene_bins.keys():
    print("bin: "+str(bin)+" count: "+str(len(tgt_hormone_gene_bins[bin])))


# In[6]:


# This is a list of genes that are associated with multiple hormones
dup_genes = []
with open('./dataset/genes_associated_with_multiple_hormones.txt','r') as f:
    for line in f:
        dup_genes.append(line[:-1])
        
#a dict containing the hormone and the list of source and target genes associated with it in HGv1 database
with open('./dataset/hgv1_hormone_src_tgt_genes.json') as json_file:
    hormone_src_tgt_genes = json.load(json_file)

# get embedding for hormones having aliases by adding the original and alias word embedding.
# Hormones having aliases have a "/" in between the two alias names
alias_embeddings = dict()
for hormone in hormone_src_tgt_genes.keys():
    if "/" in hormone:
        parts = hormone.split("/")
        w1 = model.get_word_vector(parts[0])
        w2 = model.get_word_vector(parts[1])
        alias_embeddings[hormone] = np.add(w1,w2)

src_hormone_genes = dict()
tgt_hormone_genes = dict()

for hormone in hormone_src_tgt_genes.keys():
    src_hormone_genes[hormone] = []
    tgt_hormone_genes[hormone] = []
    for gene in hormone_src_tgt_genes[hormone]['source']:
        src_hormone_genes[hormone].append(gene)
    for gene in hormone_src_tgt_genes[hormone]['target']:
        tgt_hormone_genes[hormone].append(gene)
        
# function that takes and input as a dict containing hormones and its associated list of genes and gets the 
# corresponding word vectors for each of these tokens and returns the set. Only hormomes used in training are considered
def transform_X_values(data_dict,train_marked):
    embeddings = []
    pairs = []
    src_count = 0
    count = 0
    for hormone in data_dict.keys():
        if hormone in train_marked.keys() and train_marked[hormone] == 1:
            count += 1
        else:
            if "/" in hormone:
                np1 = alias_embeddings[hormone]
            else:
                np1 = model.get_word_vector(hormone)
            for gene in data_dict[hormone]:
                np2 = model.get_word_vector(gene)
                embeddings.append(np.concatenate([np1,np2]))
                pairs.append((hormone,gene))
                
    #print(count)
    return np.array(embeddings), pairs

def get_oversampled_train_data(src_train_data, tgt_train_data, dup_genes):
    hor_map = dict()
    train_marked = dict()
    X_train_smote = []
    y_train_smote = []
    src_eligible_genes = dict()
    src_duplicate_genes = dict()
    tgt_eligible_genes = dict()
    tgt_duplicate_genes = dict()
    cnt = 1
    # get the list of genes for each hormone after removing genes which are associated with multiple hormones
    # (src_eligible genes and tgt_eligible_genes).
    for hormone in src_train_data.keys():
        src_eligible_genes[hormone] = []
        src_duplicate_genes[hormone] = []
        for gene in src_train_data[hormone]:
            if gene in dup_genes:
                src_duplicate_genes[hormone].append(gene)
            else:
                src_eligible_genes[hormone].append(gene)

        tgt_eligible_genes[hormone] = []
        tgt_duplicate_genes[hormone] = []
        for gene in tgt_train_data[hormone]:
            if gene in dup_genes:
                tgt_duplicate_genes[hormone].append(gene)
            else:
                tgt_eligible_genes[hormone].append(gene)
        
        # Consider the source and target genes for every hormone as a different class for applying SMOTE
        # Consider a hormone only if it has atleast 3 eligible source and 3 eligible target genes (constraint to apply SMOTE) 
        # and mark this gene.
        if len(src_eligible_genes[hormone]) >= 3 and len(tgt_eligible_genes[hormone]) >= 3:
            train_marked[hormone] = 1
            hor_map[cnt] = hormone+'#source'
            for gene in src_eligible_genes[hormone]:
                X_train_smote.append(model.get_word_vector(gene))
                y_train_smote.append(cnt)
            cnt += 1
            hor_map[cnt] = hormone+'#target'
            for gene in tgt_eligible_genes[hormone]:
                X_train_smote.append(model.get_word_vector(gene))
                y_train_smote.append(cnt)
            cnt += 1
        else:
            train_marked[hormone] = 0
    
    # transform dataset using smote-tomek
    smote_strategy = SMOTETomek(smote=SMOTE(k_neighbors=2))
    X_dataset_oversampled, y_dataset_oversampled = smote_strategy.fit_resample(np.array(X_train_smote), np.array(y_train_smote))
    counter = Counter(y_dataset_oversampled)
    print(counter)
    oversampled_genes_per_hormone = dict()
    X_train = []
    
    # get the oversampled embeddings for the source and the target sets
    for hormone, embedding in zip(y_dataset_oversampled, X_dataset_oversampled):
        if "/" in hor_map[hormone]:
            w1 = alias_embeddings[hor_map[hormone].split("#")[0]]
        else:
            w1 = model.get_word_vector(hor_map[hormone].split("#")[0])

        if "source" in hor_map[hormone]:
            embedding = np.append(embedding,1)
            X_train.append(np.concatenate([w1,embedding]))
        if "target" in hor_map[hormone]:
            embedding = np.append(embedding,0)
            X_train.append(np.concatenate([w1,embedding]))

        if hor_map[hormone] in oversampled_genes_per_hormone:
            oversampled_genes_per_hormone[hor_map[hormone]].append(embedding)   
        else:
            oversampled_genes_per_hormone[hor_map[hormone]] = [embedding]
    
    # add back the genes associated with multiple hormones which were removed earlier.
    for hormone in oversampled_genes_per_hormone.keys():
        if "/" in hormone:
            w1 = alias_embeddings[hormone.split("#")[0]]
        else:
            w1 = model.get_word_vector(hormone.split("#")[0])
        for gene in src_duplicate_genes[hormone.split("#")[0]]:
            w2 = model.get_word_vector(gene)
            w2 = np.append(w2,1)
            X_train.append(np.concatenate([w1,w2]))
        for gene in tgt_duplicate_genes[hormone.split("#")[0]]:
            w2 = model.get_word_vector(gene)
            w2 = np.append(w2,0)
            X_train.append(np.concatenate([w1,w2]))

    X_train = np.array(X_train)
    y_train = X_train[:,-1]
    y_train = y_train.astype(np.int32)
    X_train = np.delete(X_train,-1,axis=1)

    print("Train shape")
    print(X_train.shape)
    print(y_train.shape)
    
    return X_train, y_train, train_marked


# In[7]:


# The genes associated with every hormone are divided into 5 bins. The below files contain the source and the target genes associated with a hormone for each of the 5 bins in a dictionary form. 
with open('./dataset/hgv1_source_hormone_gene_bins.json') as json_file:
    src_hormone_gene_bins= json.load(json_file)
    
with open('./dataset/hgv1_target_hormone_gene_bins.json') as json_file:
    tgt_hormone_gene_bins = json.load(json_file)
    
f1, axes1 = plt.subplots(figsize=(5,5))
f2, axes2 = plt.subplots(figsize=(5,5))
all_bins = [0,1,2,3,4]
test_bin = 0
#print("Fold: "+str(test_bin))
all_bins.remove(test_bin)
#print("train+val available bins:")
#print(all_bins)

src_train_val_data = dict()
for hormone in src_hormone_gene_bins[str(test_bin)].keys():
    src_train_val_data[hormone] = []
    for bin_no in all_bins:
        for gene in src_hormone_gene_bins[str(bin_no)][hormone]:
            src_train_val_data[hormone].append(gene)

tgt_train_val_data = dict()
for hormone in tgt_hormone_gene_bins[str(test_bin)].keys():
    tgt_train_val_data[hormone] = []
    for bin_no in all_bins:
        for gene in tgt_hormone_gene_bins[str(bin_no)][hormone]:
            tgt_train_val_data[hormone].append(gene)

X_train_all, y_train_all, _train_marked = get_oversampled_train_data(src_train_val_data, tgt_train_val_data, dup_genes)

classifier = SVC(kernel='poly',C=1.0,degree=3)
classifier.fit(X_train_all,y_train_all)

# get the test data considering the test bin
X_test_src, src_pairs = transform_X_values(src_hormone_genes,_train_marked)
X_test_tgt, tgt_pairs = transform_X_values(tgt_hormone_genes,_train_marked)
X_test = np.concatenate([X_test_src,X_test_tgt])
all_pairs = src_pairs + tgt_pairs
y_test_src = np.ones((X_test_src.shape[0],), dtype=int)
y_test_tgt = np.zeros((X_test_tgt.shape[0],), dtype=int)
print(len(y_test_src))
print(len(y_test_tgt))

y_test = np.concatenate([y_test_src, y_test_tgt])
#print(y_test)
print("Test shape")
print(X_test.shape)
print(y_test.shape)

y_pred_test = classifier.predict(X_test)
y_dec_score_test = classifier.decision_function(X_test)

# get results on the src test set
y_pred_test_src = classifier.predict(X_test_src)
y_dec_score_test_src = classifier.decision_function(X_test_src)

# get results on the tgt test set
y_pred_test_tgt = classifier.predict(X_test_tgt)
y_dec_score_test_tgt = classifier.decision_function(X_test_tgt)


print("Testing results: source")
print(cohen_kappa_score(y_test,y_pred_test))
print(confusion_matrix(y_test, y_pred_test))
print(classification_report(y_test, y_pred_test))

fpr, tpr, thresholds = roc_curve(y_test, y_dec_score_test)
roc_lab = 'AUC=%.2f' % (roc_auc_score(y_test, y_dec_score_test))
axes1.step(fpr, tpr, label=roc_lab)

precision, recall, _ = precision_recall_curve(y_test, y_dec_score_test)
pr_lab = 'AUC=%.2f' % (auc(recall, precision))
axes2.step(recall, precision, label=pr_lab)


axes1.plot([0, 1], [0, 1], 'k--')
axes1.set_xlabel('False Positive Rate')
axes1.set_ylabel('True Positive Rate')
axes1.set_title("Receiver Operating Characteristics Curve")
axes1.legend(loc='lower right', fontsize='small')


no_skill = len(y_test[y_test==1]) / len(y_test)
axes2.plot([0, 1], [no_skill, no_skill], linestyle='--', label='Random AUC=%.2f'%(no_skill),color='black')    
axes2.set_xlabel('Recall')
axes2.set_ylabel('Precision')
axes2.legend(loc='upper right', fontsize='small')
axes2.set_title("Precision Recall curve")

f2.savefig('./output/out_of_training_pr_aur.pdf')

results = []
src_cnt = 0
tgt_cnt = 0
correct = 0
for pair,y,y_score in zip(all_pairs,y_pred_test,y_dec_score_test):
    if pair[1] in hormone_src_tgt_genes[pair[0]]['source']:
        y1 = 1
        src_cnt += 1
    elif pair[1] in hormone_src_tgt_genes[pair[0]]['target']:
        y1 = 0
        tgt_cnt += 1
    
    if y1 == y:
        correct += 1
    results.append([pair[0], pair[1], y1, y, y_score])

print(src_cnt)
print(tgt_cnt)
print(correct/(src_cnt+tgt_cnt))
with open('./bioembeds_ts_predictions.csv', mode='w') as pred_file:
    writer = csv.writer(pred_file, delimiter=',')
    header = ['Hormone','Gene','Original Type','Prediction','SVM score']
    writer.writerow(header)
    for row in results:
        writer.writerow(row)



#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import json
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve


# In[3]:


with open('./dataset/hgv1_hormone_genes.json') as json_file:
    hormone_genes = json.load(json_file)
    
with open('./dataset/fasttext_word_embeddings.json') as json_file:
    word_embeddings = json.load(json_file)
    
with open('./dataset/hgv1_hormone_gene_bins.json') as json_file:
    hormone_gene_bins = json.load(json_file)
    
with open('./dataset/hgv1_negative_hormone_gene_bins.json') as json_file:
    neg_hormone_gene_bins = json.load(json_file)
    

color_dict = {"-2":"c","-1":"m","0":"b","1":"g","2":"r","3":"c","4":"y"}

alias_embeddings = dict()
for hormone in hormone_genes.keys():
    if "/" in hormone:
        parts = hormone.split("/")
        w1 = np.array(word_embeddings[parts[0]])
        w2 = np.array(word_embeddings[parts[1]])
        alias_embeddings[hormone] = np.add(w1,w2)
        
def get_pairs(data_dict,train_marked):
    pairs = []
    for hormone in data_dict.keys():
        if train_marked[hormone] == 1:
            for gene in data_dict[hormone]:
                pairs.append((hormone,gene))
    return pairs

def cosine_similarity(a,b):
    return np.inner(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# In[4]:


f1, axes1 = plt.subplots(figsize=(4,4))
f2, axes2 = plt.subplots(figsize=(4,4))
for i in range(5):
    y_test = np.load('./output/y_fold_'+str(i)+'.npy')
    y_pred = np.load('./output/y_pred_fold_'+str(i)+'.npy')
    y_prob = np.load('./output/y_dec_score_fold_'+str(i)+'.npy')
    
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_lab = 'Fold %d AUC=%.2f' % (i+1, roc_auc_score(y_test, y_prob))
    axes1.step(fpr, tpr, label=roc_lab, linewidth=2.5,color=color_dict[str(i)])
    
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_lab = 'Fold %d AUC=%.4f' % (i+1, auc(recall, precision))
    axes2.step(recall, precision, label=pr_lab)
    
    pos_dict = hormone_gene_bins[str(i)]
    neg_dict = neg_hormone_gene_bins[str(i)]
    
    with open('./output/train_val_marking_fold_'+str(i)+'_v1.json') as json_file:
        train_marked = json.load(json_file)
    
    pos_pairs = get_pairs(pos_dict,train_marked)
    neg_pairs = get_pairs(neg_dict,train_marked)
    
    y_test_cos = []
    y_pred_cos = []
    y_dec_score_cos = []
    
    for pair in pos_pairs:
        y_test_cos.append(1)
        if "/" in pair[0]:
            np1 = alias_embeddings[pair[0]]
        else:
            np1 = np.array(word_embeddings[pair[0]])
        np2 = np.array(word_embeddings[pair[1]])
        cos_sim = cosine_similarity(np1,np2)
        if cos_sim >= 0.5:
            y_pred_cos.append(1)
        else:
            y_pred_cos.append(0)
        y_dec_score_cos.append(cos_sim)
    
    for pair in neg_pairs:
        y_test_cos.append(0)
        if "/" in pair[0]:
            np1 = alias_embeddings[pair[0]]
        else:
            np1 = np.array(word_embeddings[pair[0]])
        np2 = np.array(word_embeddings[pair[1]])
        cos_sim = cosine_similarity(np1,np2)
        if cos_sim >= 0.5:
            y_pred_cos.append(1)
        else:
            y_pred_cos.append(0)
        y_dec_score_cos.append(cos_sim)
        
    print(confusion_matrix(y_test_cos, y_pred_cos))
    print(classification_report(y_test_cos, y_pred_cos))
    
    precision, recall, _ = precision_recall_curve(y_test_cos, y_dec_score_cos)
    pr_lab = 'Fold %d AUC=%.4f' % (i+1, auc(recall, precision))
    axes2.step(recall, precision, '--',label=pr_lab)

    fpr, tpr, thresholds = roc_curve(y_test_cos, y_dec_score_cos)
    roc_lab = 'Fold %d AUC=%.2f' % (i+1, roc_auc_score(y_test_cos, y_dec_score_cos))
    axes1.step(fpr, tpr, '--',label=roc_lab,color=color_dict[str(i)])
        
    
axes1.plot([0, 1], [0, 1], 'k--')
axes1.set_xlabel('False Positive Rate')
axes1.set_ylabel('True Positive Rate')
axes1.set_title("Receiver Operating Characteristics Curve")
axes1.legend(loc='lower right', fontsize=7.5)


no_skill = len(y_test[y_test==1]) / len(y_test)
axes2.plot([0, 1], [no_skill, no_skill], linestyle='--', label='Random')    
axes2.set_xlabel('Recall')
axes2.set_ylabel('Precision')
axes2.legend(loc='upper right', fontsize='small')
axes2.set_title("Precision Recall curve")

f1.savefig('./output/BioEmbedS_vs_cos_roc.pdf')


# In[ ]:





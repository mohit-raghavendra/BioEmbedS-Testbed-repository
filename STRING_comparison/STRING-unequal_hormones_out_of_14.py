#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import fasttext
import csv
import numpy as np
import random
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import math
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from matplotlib import pyplot
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc

(/add, the, path, the, fasttext, model.)
model = fasttext.load_model("../BioWordVec_PubMed_MIMICIII_d200.bin")


# In[2]:


with open('./BioEmbedS/dataset/hgv1_hormone_genes.json') as json_file:
    hormone_genes = json.load(json_file)
#print(hormone_genes.keys())

with open('./BioEmbedS/dataset/hgv1_hormone_gene_bins.json') as json_file:
    hormone_gene_bins = json.load(json_file)
    
with open('./BioEmbedS/dataset/hgv1_negative_hormone_gene_bins.json') as json_file:
    neg_hormone_gene_bins = json.load(json_file)


# In[3]:


name_mapping = dict()
for line in open('9606.protein.info.v11.0.txt','r'):
    p1 = line.split(" ")[0].split("\t")[0]
    p2 = line.split(" ")[0].split("\t")[1]
    if p1 != 'protein_external_id':
        name_mapping[p1] = p2
        
hormone_genename_dict = {'calcitonin':'CALCA','cholecystokinin':'CCK','gastrin':'GAST','ghrelin':'GHRL','glucagon':'GCG','growth hormone-releasing hormone':'GHRH','insulin':'INS','leptin':'LEP','oxytocin':'OXT','parathyroid hormone/parathyrin':'PTH','prolactin':'PRL','somatostatin':'SST','thyrotropin-releasing hormone':'TRH','antidiuretic hormone/vasopressin':'AVP'}
inv_hormone_genename_dict = {'CALCA':'calcitonin','CCK':'cholecystokinin','GAST':'gastrin','GHRL':'ghrelin','GCG':'glucagon','GHRH':'growth hormone-releasing hormone','INS':'insulin','LEP':'leptin','OXT':'oxytocin','PTH':'parathyroid hormone/parathyrin','PRL':'prolactin','SST':'somatostatin','TRH':'thyrotropin-releasing hormone','AVP':'antidiuretic hormone/vasopressin'}
hor_gene_lst = ['CALCA','CCK','GAST','GHRL','GCG','GHRH','INS','LEP','PTH','PRL','SST','TRH','AVP']

color_dict = {"-2":"c","-1":"m","0":"b","1":"g","2":"r","3":"c","4":"y"}


# In[4]:


def select_pept_hormones(all_hormone_genes,hormone_genename,train_marked):
    selected_hormone_genes = dict()
    for hor in all_hormone_genes.keys():
        if hor in hormone_genename.keys() and train_marked[hor] == 1:
            selected_hormone_genes[hor] = all_hormone_genes[hor]
    return selected_hormone_genes

def get_string_results(pept_hormone_genes, neg_pept_hormone_genes, string_preds, hormone_genename,t):
    #print("threshold= "+str(t))
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for hor in pept_hormone_genes.keys():
        string_pos_preds = []
        string_neg_preds = []
        for p in string_preds[hormone_genename[hor]]:
            if int(p[1]) >= t:
                string_pos_preds.append(p[0].lower())
            else:
                string_neg_preds.append(p[0].lower())
        
        for gene in pept_hormone_genes[hor]:
            if gene in string_pos_preds:
                tp += 1
            else:
                fn += 1
            
        for gene in neg_pept_hormone_genes[hor]:
            if gene in string_neg_preds:
                tn += 1
            else:
                fp += 1
            
           
    #print(tp)
    #print(fp)
    #print(fn)
    if tp!= 0:
        precision = tp/(tp+fp)
    else:
        precision = 0
    
    recall = tp/(tp+fn)
    fpr = fp/(fp+tn)
    #print(precision)
    #print(recall)
    #f1_score = (2*precision*recall)/(precision+recall)
        
    return precision, recall, fpr


# In[5]:


#fig, axes = plt.subplots(2, 5,figsize=(15,10))
#fig=plt.figure(figsize=(25,4))
#rows = 1
#columns = 5
minn_lit= 0
maxx_lit = 970
f1, axes1 = plt.subplots(figsize=(4,4))
f2, axes2 = plt.subplots(figsize=(4,4))
for i in range(5):
    #print("fold= "+str(i))
    
    y_test = np.load('./BioEmbedS/output/y_fold_'+str(i)+'.npy')
    y_pred = np.load('./BioEmbedS/output/y_pred_fold_'+str(i)+'.npy')
    y_prob = np.load('./BioEmbedS/output/y_dec_score_fold_'+str(i)+'.npy')
    
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_lab = 'Fold %d AUC=%.2f' % (i+1, roc_auc_score(y_test, y_prob))
    axes2.step(fpr, tpr, label=roc_lab, color=color_dict[str(i)], linewidth=2)
    
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_lab = 'Fold %d AUC=%.2f' % (i+1, auc(recall, precision))
    axes1.step(recall, precision,label=pr_lab, color=color_dict[str(i)], linewidth=2)
    
    test_data = hormone_gene_bins[str(i)]
    neg_test_data = neg_hormone_gene_bins[str(i)]
    
    #using newly generated train markings
    with open('./BioEmbedS/output/train_val_marking_fold_'+str(i)+'_v1.json') as json_file:
        train_marked = json.load(json_file)
    
    #print(train_marked)
    hormone_genename = dict()
    inv_hormone_genename = dict()
    for hormone in hormone_genename_dict.keys():
        if hormone in train_marked.keys():
            if train_marked[hormone] == 1:
                hormone_genename[hormone] = hormone_genename_dict[hormone]
                inv_hormone_genename[hormone_genename_dict[hormone]] = hormone
    
    #print(hormone_genename)
    #print(inv_hormone_genename)
    
    string_preds = dict()
    string_preds_lit = dict()
    string_hormone_genes = dict()
    for line in open('9606.protein.links.detailed.v11.0.txt','r'):
        splits = line.split(" ")
        if splits[0] != 'protein1' and name_mapping[splits[0]] in hor_gene_lst:
            if name_mapping[splits[0]] not in string_preds.keys():
                string_preds[name_mapping[splits[0]]] = []
                string_preds[name_mapping[splits[0]]].append((name_mapping[splits[1]],splits[9].split("\n")[0]))
                string_preds_lit[name_mapping[splits[0]]] = []
                string_preds_lit[name_mapping[splits[0]]].append((name_mapping[splits[1]],splits[8]))
                string_hormone_genes[name_mapping[splits[0]]] = []
                string_hormone_genes[name_mapping[splits[0]]].append(name_mapping[splits[1]])
            else:
                string_preds[name_mapping[splits[0]]].append((name_mapping[splits[1]],splits[9].split("\n")[0]))
                string_preds_lit[name_mapping[splits[0]]].append((name_mapping[splits[1]],splits[8]))
                string_hormone_genes[name_mapping[splits[0]]].append(name_mapping[splits[1]])
    
    
    pept_hormone_genes = select_pept_hormones(test_data,hormone_genename,train_marked)
    neg_pept_hormone_genes = select_pept_hormones(neg_test_data,hormone_genename,train_marked)
    #print(pept_hormone_genes.keys())
    
    rem_pairs = dict()
    string_genes = dict()
    for hor in string_preds_lit.keys():
        if hor in inv_hormone_genename.keys() and inv_hormone_genename[hor] in pept_hormone_genes.keys():
            rem_pairs[hor] = []
            string_genes[hor] = []
            for pair in string_preds_lit[hor]:
                string_genes[hor].append(pair[0])
                if pair[0].lower() not in pept_hormone_genes[inv_hormone_genename[hor]] and pair[0].lower() not in neg_pept_hormone_genes[inv_hormone_genename[hor]]:
                    rem_pairs[hor].append(pair)
    
    for hor in rem_pairs.keys():
        for pair in rem_pairs[hor]:
            if pair in string_preds_lit[hor]:
                string_preds_lit[hor].remove(pair)
    
    
    for hor in pept_hormone_genes.keys():
        for gene in pept_hormone_genes[hor]:
            if gene.upper() not in string_genes[hormone_genename[hor]]:
                string_preds_lit[hormone_genename[hor]].append((gene.upper(),str(0)))
                                                               
    for hor in neg_pept_hormone_genes.keys():
        for gene in neg_pept_hormone_genes[hor]:
            if gene.upper() not in string_genes[hormone_genename[hor]]:
                string_preds_lit[hormone_genename[hor]].append((gene.upper(),str(0)))
    
    """
    for hor in pept_hormone_genes.keys():
        print(hor)
        print(len(string_preds_lit[hormone_genename[hor]]))
        print(len(pept_hormone_genes[hor]))
        print(len(neg_pept_hormone_genes[hor]))                                                                          
    """
    
    thresholds = np.arange(minn_lit,maxx_lit,25).tolist()
    thresholds.append(970)
    s_prec = []
    s_rec = []
    s_fpr = []
    for t in thresholds:
        #print("t= "+str(t))
        precision, recall, fpr= get_string_results(pept_hormone_genes, neg_pept_hormone_genes, string_preds_lit, hormone_genename,t)
        #print(precision)
        #print(recall)
        s_prec.append(precision)
        s_rec.append(recall)
        s_fpr.append(fpr)
    #print("tpr")
    #print(s_rec)
    #print("fpr")
    #print(s_fpr)
    
    #s_rec.insert(0,1)
    #s_fpr.insert(0,1)
    s_roc_lab = 'Fold %d AUC=%.2f' % (i+1, auc(s_fpr,s_rec))
    axes2.plot(s_fpr, s_rec,'--', label=s_roc_lab, color=color_dict[str(i)])
    
    s_pr_lab = 'Fold %d AUC=%.2f' % (i+1, auc(s_rec,s_prec))
    axes1.step(s_rec, s_prec,'--', label=s_pr_lab, color=color_dict[str(i)])
    
no_skill = len(y_test[y_test==1]) / len(y_test)
axes1.plot([0, 1], [no_skill, no_skill], linestyle='--', label='Random')    
axes1.set_xlabel('Recall')
axes1.set_ylabel('Precision')
axes1.legend(loc='lower right', fontsize=10)
axes1.set_title("Precision Recall curve")

axes2.plot([0, 1], [0, 1], 'k--')
axes2.set_xlabel('False Positive Rate')
axes2.set_ylabel('True Positive Rate')
axes2.set_title("Receiver Operating Characteristics Curve")
axes2.legend(loc='lower right', fontsize=7.5)

#f2.savefig('unequal_bioembeds_vs_string_roc.pdf')
#f1.savefig('bioembeds_vs_string_pr.pdf')







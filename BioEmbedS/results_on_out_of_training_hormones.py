#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv
import json
import random
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from matplotlib import pyplot
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve

with open('./dataset/hgv1_hormone_genes.json') as json_file:
    hormone_genes = json.load(json_file)
    
with open('./dataset/gene_types.txt') as json_file:
    gene_types = json.load(json_file)
        
hormone_list = ['ghrelin', 'human chorionic gonadotropin', 'gastrin', 'thyroxine t4', 'growth hormone-releasing hormone', 'dihydrotestosterone', 'glucagon like peptide 1', 'androgen', 'progestin', 'luteinizing hormone', 'dopamine', 'dehydroepiandrosterone', 'corticotropin-releasing hormone', 'thyroid-stimulating hormone/thyrotropin', 'kisspeptin/metastin', 'estrone/oestrone', 'triiodothyronin t3', 'aromatase', 'oxytocin']
neg_hormone_genes = dict()
for hormone in hormone_list:
    neg_hormone_genes[hormone] = []

with open('./dataset/hgv1_negative_hormone_gene_bins.json') as json_file:
    neg_hormone_gene_bins = json.load(json_file)
    for i in range(5):
        for hormone in neg_hormone_gene_bins[str(i)].keys():
            if hormone in hormone_list:
                for gene in neg_hormone_gene_bins[str(i)][hormone]:
                    neg_hormone_genes[hormone].append(gene) 
                    
doc_genes = []
with open('./dataset/hgv1_gene_list.txt','r') as f:
    for line in f:
        doc_genes.append(line[:-1])

del_genes = []
for gene in doc_genes:
    if gene.upper() in gene_types.keys():
        if gene_types[gene.upper()] != 'gene with protein product':
            del_genes.append(gene)
    else:
        del_genes.append(gene)

#print(del_genes)

for gene in del_genes:
    doc_genes.remove(gene)
#print(len(doc_genes))

for hormone in neg_hormone_genes.keys():
    print(hormone+": "+str(len(neg_hormone_genes[hormone])))

for hormone in neg_hormone_genes.keys():
    neg_hormone_genes[hormone] = []
    if len(neg_hormone_genes[hormone]) == 0:
        print(hormone)
        rem_set = set(doc_genes).difference(set(hormone_genes[hormone]))
        random.seed(42)
        neg_set = random.sample(rem_set, len(hormone_genes[hormone]))
        for gene in list(neg_set):
            neg_hormone_genes[hormone].append(gene)
            
#for hormone in neg_hormone_genes.keys():
#    print(hormone)
#    print(len(neg_hormone_genes[hormone]))
#    print(len(hormone_genes[hormone]))


# In[3]:


hormone_genes_preds = dict()
for hormone in neg_hormone_genes.keys():
    hormone_genes_preds[hormone] = dict()
    for gene in hormone_genes[hormone]:
        hormone_genes_preds[hormone][gene] = [1,'x','x']
    for gene in neg_hormone_genes[hormone]:
        if gene == 'urs00005cc20c_9606':
            print(hormone+"found")
        hormone_genes_preds[hormone][gene] = [0,'x','x']

#print(hormone_genes_preds)
hormone_genes_preds.pop('progestin')
hormone_genes_preds.pop('aromatase')
#print(len(hormone_genes_preds.keys()))
f1, axes1 = plt.subplots(figsize=(5,5))
f2, axes2 = plt.subplots(figsize=(5,5))
with open('./dataset/protein_coding_genes_out_of_training_set_predictions.csv','rt')as f:
    data = csv.reader(f)
    next(data)
    for row in data:
        if row[0] in hormone_genes_preds.keys():
            if row[1].lower() in hormone_genes_preds[row[0]].keys():
                if row[4] == '1':
                    hormone_genes_preds[row[0]][row[1].lower()][1] = 1
                    hormone_genes_preds[row[0]][row[1].lower()][2] = float(row[5])
                if row[4] == '0':
                    hormone_genes_preds[row[0]][row[1].lower()][1] = 0
                    hormone_genes_preds[row[0]][row[1].lower()][2] = float(row[5])
                
#print(hormone_genes_preds)

y_test = []
y_pred = []
y_dec_score = []
for hormone in hormone_genes_preds.keys():
    for gene in hormone_genes_preds[hormone]:
        if hormone_genes_preds[hormone][gene][1] != 'x':
            y_test.append(hormone_genes_preds[hormone][gene][0])
            y_pred.append(hormone_genes_preds[hormone][gene][1])
            y_dec_score.append(hormone_genes_preds[hormone][gene][2])
        
#print(len(y_test))
#print(y_pred)
#print(y_dec_score)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

precision, recall, _ = precision_recall_curve(y_test, y_dec_score)
pr_lab = 'AUC=%.2f' % (auc(recall, precision))
axes1.step(recall, precision, label=pr_lab)

fpr, tpr, thresholds = roc_curve(y_test, y_dec_score)
roc_lab = 'AUC=%.2f' % (roc_auc_score(y_test, y_dec_score))
axes2.step(fpr, tpr, label=roc_lab)

tot_ones = 0
for item in y_test:
    if item == 1:
        tot_ones += 1

axes2.plot([0, 1], [0, 1], 'k--',label='Random AUC=0.5', color='black')
axes2.set_xlabel('False Positive Rate')
axes2.set_ylabel('True Positive Rate')
axes2.set_title("Receiver Operating Characteristics Curve")
axes2.legend(loc='lower right', fontsize='large')

#print(tot_ones)
no_skill = tot_ones / len(y_test)
axes1.plot([0, 1], [no_skill, no_skill], linestyle='--', label='Random')    
axes1.set_xlabel('Recall')
axes1.set_ylabel('Precision')
axes1.legend(loc='upper right', fontsize='large')
axes1.set_title("Precision Recall curve")

f2.savefig('./out_of_training_roc_rounded_auc.pdf')


# In[ ]:





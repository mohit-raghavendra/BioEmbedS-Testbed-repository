#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve


f1, axes1 = plt.subplots(figsize=(5,5))
f2, axes2 = plt.subplots(figsize=(5,5))
for i in range(5):
    y_test = np.load('./output/ts_y_fold_'+str(i)+'.npy')
    y_pred = np.load('./output/ts_y_pred_fold_'+str(i)+'.npy')
    y_prob = np.load('./output/ts_y_dec_score_fold_'+str(i)+'.npy')
    
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_lab = 'Fold %d AUC=%.4f' % (i+1, roc_auc_score(y_test, y_prob))
    axes1.step(fpr, tpr, label=roc_lab)
    
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_lab = 'Fold %d AUC=%.2f' % (i+1, auc(recall, precision))
    axes2.step(recall, precision, label=pr_lab)
    
axes1.plot([0, 1], [0, 1], 'k--')
axes1.set_xlabel('False Positive Rate')
axes1.set_ylabel('True Positive Rate')
axes1.set_title("Receiver Operating Characteristics Curve")
axes1.legend(loc='lower right', fontsize='small')


no_skill = len(y_test[y_test==1]) / len(y_test)
axes2.plot([0, 1], [no_skill, no_skill], linestyle='--', label='Random AUC=%.2f'%(no_skill), color='black')    
axes2.set_xlabel('Recall')
axes2.set_ylabel('Precision')
axes2.legend(loc='upper right', fontsize='small')
axes2.set_title("Precision Recall curve")
    
f1.savefig('./output/BioEmbedS_roc.pdf')
f2.savefig('./output/BioEmbedS_TS_pr_rounded_auc.pdf')
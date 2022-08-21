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

with open('./dataset/hgv1_hormone_genes.json') as json_file:
#with open('./dataset/hgv1_hormone_genes.json') as json_file:
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

def get_results_for_fold(hormone_fold_results, fold_num):
    with open('./dataset/correct_train_val_marking_fold_'+str(fold_num)+'.json') as json_file:
        train_marked = json.load(json_file)
        
    gene_cnt = []
    for hormone in hormone_genes.keys():
        gene_cnt.append(len(hormone_genes[hormone]))

    df = pd.DataFrame(gene_cnt)
    res,bin_edges = pd.qcut(df[0], q=4,retbins=True)
    print(bin_edges)
    
    bin1_hor_list = []
    bin2_hor_list = []
    bin3_hor_list = []
    bin4_hor_list = []
    cnt1=0
    cnt2=0
    cnt3=0
    cnt4=0
    
    for hormone in hormone_fold_results[fold_num].keys():
        if hormone in train_marked.keys() and train_marked[hormone] == 1:
            length = len(hormone_genes[hormone])
            if length <= int(bin_edges[2]):
                bin1_hor_list.append(hormone)
            elif length > int(bin_edges[2]) and length<= int(bin_edges[3]):
                bin2_hor_list.append(hormone)
            elif length > int(bin_edges[3]) and length< int(bin_edges[4]):
                bin3_hor_list.append(hormone)
            else:
                bin4_hor_list.append(hormone)
            
    get_results_for_bin(hormone_fold_results,fold_num,bin1_hor_list,1)
    get_results_for_bin(hormone_fold_results,fold_num,bin2_hor_list,2)
    get_results_for_bin(hormone_fold_results,fold_num,bin3_hor_list,3)
    get_results_for_bin(hormone_fold_results,fold_num,bin4_hor_list,4)
        
    
def get_results_for_bin(hormone_fold_results,fold_num,bin_hor_list,bin_num):
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for hormone in bin_hor_list:
        tp += hormone_fold_results[fold_num][hormone][0]
        fp += hormone_fold_results[fold_num][hormone][1]
        fn += hormone_fold_results[fold_num][hormone][2]
        tn += hormone_fold_results[fold_num][hormone][3]
    pr = tp/(tp+fp)
    rec = tp/(tp+fn)
    acc = (tp+tn)/(tp+fp+fn+tn)
    if pr == 0 and rec == 0:
        f1_scr = 'undef'
    else:
        f1_scr = 2*pr*rec/(rec+pr)
    print("For fold:"+str(fold_num)+" ,for bin:"+str(bin_num)+" results")
    print(" Pr:"+str(pr)+" Rec:"+str(rec)+" Acc:"+str(acc)+" F1-score:"+str(f1_scr)+" #pairs:"+str(tp+fp+fn+tn))
    
def get_hormone_wise_results_for_fold(hormone_fold_results, fold_num):
    print("Printing hormone wise results for fold: "+str(fold_num))
    
    with open('./dataset/correct_train_val_marking_fold_'+str(fold_num)+'.json') as json_file:
        train_marked = json.load(json_file)
    results = dict()
    for hormone in hormone_fold_results[fold_num].keys():
        if hormone in train_marked.keys() and train_marked[hormone] == 1:
            results[hormone] = []
            tp = hormone_fold_results[fold_num][hormone][0]
            fp = hormone_fold_results[fold_num][hormone][1]
            fn = hormone_fold_results[fold_num][hormone][2]
            tn = hormone_fold_results[fold_num][hormone][3]
            total = tp+fp+fn+tn
            pr = tp/(tp+fp)
            rec = tp/(tp+fn)
            acc = (tp+tn)/total
            if pr == 0 and rec == 0:
                f1_scr = 'undef'
            else:
                f1_scr = 2*pr*rec/(rec+pr)
            print("For hormone: "+hormone+" Pr:"+str(pr)+" Rec:"+str(rec)+" Acc:"+str(acc)+" F1-score:"+str(f1_scr)+" #pairs:"+str(tp+fp+fn+tn)) 
            results[hormone].extend([hormone,pr,rec,acc,f1_scr,total])
            
    #with open('./BioEmbedS_dummy/output/hormone_wise_results_fold'+str(fold_num)+'.json', 'w') as outfile:
    #    json.dump(results,outfile)
    #outfile.close()
    
    with open('./output/hormone_wise_results_fold'+str(fold_num)+'.csv', mode='w') as pred_file:
        writer = csv.writer(pred_file, delimiter=',')
        header = ['Hormone','Precision','Recall','Accuracy','F1-score','# pairs']
        writer.writerow(header)

        for hormone in results.keys():
            writer.writerow(results[hormone])
            
    
with open('./dataset/hgv1_hormone_gene_bins.json') as json_file:
#with open('./dataset/hgv1_hormone_gene_bins.json') as json_file:
    hormone_gene_bins = json.load(json_file)
    
with open('./dataset/hgv1_negative_hormone_gene_bins.json') as json_file:
#with open('./dataset/hgv1_hormone_gene_bins.json') as json_file:
    neg_hormone_gene_bins = json.load(json_file)
    
hormone_fold_results = dict()
for i in range(5):
    hormone_fold_results[i] = dict()
    for hormone in hormone_genes.keys():
        hormone_fold_results[i][hormone] = [0,0,0,0]
    
hormone_results = dict()
for hormone in hormone_genes.keys():
    hormone_results[hormone] = [0,0,0,0]
    
hormone_mark = dict()
hormone_test_genes = dict()
for hormone in hormone_genes.keys():
    hormone_test_genes[hormone] = []
    hormone_mark[hormone] = 0
    
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
    with open('./dataset/correct_train_val_marking_fold_'+str(i)+'.json') as json_file:
    #with open('./dataset/train_val_marking_fold_'+str(i)+'.json') as json_file:
        train_marked = json.load(json_file)
        
    for hormone in hormone_genes.keys():
        if hormone in train_marked.keys() and train_marked[hormone] == 1:
            hormone_mark[hormone] = 1
            #hormone_test_genes[hormone] = []
            X_test_pos,pairs = transform_X_values(hormone,test_data[hormone])
            X_test_neg,pairs = transform_X_values(hormone,neg_test_data[hormone])
            hormone_test_genes[hormone].extend(test_data[hormone])
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
            hormone_results[hormone][0] += int(tp) 
            hormone_results[hormone][1] += int(fp)
            hormone_results[hormone][2] += int(fn)
            hormone_results[hormone][3] += int(tn)

            hormone_fold_results[i][hormone][0] += int(tp) 
            hormone_fold_results[i][hormone][1] += int(fp)
            hormone_fold_results[i][hormone][2] += int(fn)
            hormone_fold_results[i][hormone][3] += int(tn)
            
get_results_for_fold(hormone_fold_results, 1)
get_results_for_fold(hormone_fold_results, 4)

get_hormone_wise_results_for_fold(hormone_fold_results, 1)
get_hormone_wise_results_for_fold(hormone_fold_results, 4)

with open('./output/all_hormone_foldwise_results.json', 'w') as outfile:
    json.dump(hormone_fold_results,outfile)
outfile.close()
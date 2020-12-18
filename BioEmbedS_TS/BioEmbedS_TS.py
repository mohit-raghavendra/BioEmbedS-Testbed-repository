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

# This is a list of genes that are associated with multiple hormones
dup_genes = []
with open('./BioEmbedS_TS/dataset/genes_associated_with_multiple_hormones.txt','r') as f:
    for line in f:
        dup_genes.append(line[:-1])
        
#a dict containing the hormone and the list of source and target genes associated with it in HGv1 database
with open('./BioEmbedS_TS/dataset/hgv1_hormone_src_tgt_genes.json') as json_file:
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

# function that takes and input as a dict containing hormones and its associated list of genes and gets the 
# corresponding word vectors for each of these tokens and returns the set. Only hormomes used in training are considered
def transform_X_values(data_dict,train_marked):
    embeddings = []
    for hormone in data_dict.keys():
        if train_marked[hormone] == 1:
            if "/" in hormone:
                np1 = alias_embeddings[hormone]
            else:
                np1 = model.get_word_vector(hormone)
            for gene in data_dict[hormone]:
                np2 = model.get_word_vector(gene)
                embeddings.append(np.concatenate([np1,np2]))
    return np.array(embeddings)

# same functionality as the above function but also filters genes based on the bins they are assigned to.
def transform_X_values_new(data_dict, bins, train_marked):
    embeddings = []
    for hormone in data_dict.keys():
        if hormone in bins and train_marked[hormone] == 1:
            if "/" in hormone:
                np1 = alias_embeddings[hormone]
            else:
                np1 = model.get_word_vector(hormone)
            for gene in data_dict[hormone]:
                np2 = model.get_word_vector(gene)
                embeddings.append(np.concatenate([np1,np2]))
    return np.array(embeddings)

# Below two functions fit the classifier for given set of parameters and return the result on validation set.
# Add/Remove/Update these functions to try with different classifiers.

# function to fit the SVM model with given set of parameters and return the Kappa score
def get_svm_estimator_results(X_train, y_train, X_val, y_val, params):
    param = {}
    param['kernel'] = params[0]
    param['C'] = params[1]
    if params[0] == 'rbf':
        param['gamma'] = params[2]
    else:
        param['degree'] = params[2]
    svclassifier = SVC()
    svclassifier.set_params(**param)
    svclassifier.fit(X_train,y_train)
    y_pred = svclassifier.predict(X_val)
    score = cohen_kappa_score(y_val,y_pred)
    param_str = "svm/"+params[0]+"/"+str(params[1])+"/"+str(params[2])
    return (param_str, score)

# function to fit the Random Forest model with given set of parameters and return the Kappa score
def get_rf_estimator_results(X_train, y_train, X_val, y_val, params):
    param = {}
    param['bootstrap'] = params[0]
    param['max_depth'] = params[1]
    param['max_features'] = params[2]
    param['min_samples_leaf'] = params[3]
    param['min_samples_split'] = params[4]
    param['n_estimators'] = params[5]
    
    rf = RandomForestClassifier()
    rf.set_params(**param)
    rf.fit(X_train,y_train)
    y_pred = rf.predict(X_val)
    score = cohen_kappa_score(y_val,y_pred)
    param_str = "rf/"+str(params[0])+"/"+str(params[1])+"/"+str(params[2])+"/"+str(params[3])+"/"+str(params[4])+"/"+str(params[5])
    return (param_str, score)

# Apply SMOTE-Tomek links oversampling+undersampling methodology to get a balanced dataset (Same # of genes associated with every hormone)
# Takes input 1.A dictionary with hormones and list of source and target genes associated with them. 
# 2.The list of genes associated with multiple hormones.
# Returns the training sets and a dict indicating which hormones are present in the training sets
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

# A function to divide the hormones into 3 bins depending on the number of genes associated with the hormone.
# Then uses the classifier to get results for the predictions of hormones in each of the bins.
def get_binned_results(src_test_data, tgt_test_data, train_marked, svclassifier):
    gene_cnt = []
    for hormone in src_test_data.keys():
        if train_marked[hormone] == 1:
            length = len(src_test_data[hormone]) + len(tgt_test_data[hormone])
            gene_cnt.append(length)
        
    df = pd.DataFrame(gene_cnt)
    res,bin_edges = pd.qcut(df[0], q=3,retbins=True)
    bin1 = []
    bin2 = []
    bin3 = []

    for hormone in src_test_data.keys():
        length = len(src_test_data[hormone]) + len(tgt_test_data[hormone])
        if length <= int(bin_edges[2]):
            bin1.append(hormone)
        elif length > int(bin_edges[2]) and length< int(bin_edges[3]):
            bin2.append(hormone)
        else:
            bin3.append(hormone)

    all_bins = []
    all_bins.append(bin1)
    all_bins.append(bin2)
    all_bins.append(bin3)
    
    print(all_bins)
    print(bin_edges)
    
    for i, bin_name in zip(range(3),all_bins):
        print("Testing results for bin "+str(i+1))
        X_test_src = transform_X_values_new(src_test_data, bin_name,train_marked)
        X_test_tgt = transform_X_values_new(tgt_test_data, bin_name,train_marked)
        X_test = np.concatenate([X_test_src,X_test_tgt])
        y_test_src = np.ones((X_test_src.shape[0],), dtype=int)
        y_test_tgt = np.zeros((X_test_tgt.shape[0],), dtype=int)
        y_test = np.concatenate([y_test_src, y_test_tgt])
        print("bin_size")
        print(X_test.shape)
        y_pred_test = svclassifier.predict(X_test)
        print(confusion_matrix(y_test, y_pred_test))
        print(classification_report(y_test, y_pred_test))
        
# The genes associated with every hormone are divided into 5 bins. The below files contain the source and the target genes associated with a hormone for each of the 5 bins in a dictionary form. 
with open('./BioEmbedS_TS/dataset/hgv1_source_hormone_gene_bins.json') as json_file:
    src_hormone_gene_bins = json.load(json_file)
    
with open('./BioEmbedS_TS/dataset/hgv1_target_hormone_gene_bins.json') as json_file:
    tgt_hormone_gene_bins = json.load(json_file)
    
# Set the range of parameters for SVM classifier
param_str_set = []
rbf_param_list = []
C_range = np.logspace(-4, 4, 9)
gamma_range = np.logspace(-9, 2, 12)
degree_range = [2, 3, 5, 7]

for C in C_range:
    for gamma in gamma_range:
        rbf_param_list.append(('rbf',C,gamma))
        param_str_set.append("svm/rbf/"+str(C)+"/"+str(gamma))

poly_param_list = []
for C in C_range:
    for degree in degree_range:
        poly_param_list.append(('poly',C,degree))
        param_str_set.append("svm/poly/"+str(C)+"/"+str(degree))
        
svm_param_lst = rbf_param_list + poly_param_list

# Set the range of parameters for Random forest classifier.
bootstrap = [False]
max_depth = [120, 160, 210, 250, 320, 380, 460, 500]
max_features = ['auto', 'sqrt']
min_samples_leaf = [1, 2, 4]
min_samples_split = [2, 3, 5, 7]
n_estimators = [100, 200, 300, 500, 600, 1200]
rf_param_lst = []
for comb in itertools.product(bootstrap,max_depth,max_features,min_samples_leaf,min_samples_split,n_estimators):
    rf_param_lst.append(comb)
    param_str_set.append("rf/"+str(comb[0])+"/"+str(comb[1])+"/"+str(comb[2])+"/"+str(comb[3])+"/"+str(comb[4])+"/"+str(comb[5]))

# Initialize a dictionary to store the score obtained on each of the 4 validation sets for every parameter combination
model_res_dict = {}
for param_str in param_str_set:
    model_res_dict[param_str] = []


all_bins = [0,1,2,3,4]
# This script runs considering bin 0 as the test set.
print("Enter the test bin number, among [0, 1, 2, 3, 4]")
test_bin = int(input())
if test_bin < 0 or test_bin > 4:
    test_bin = 0
print("Fold: "+str(test_bin))
all_bins.remove(test_bin)
print("train+val available bins:")
print(all_bins)

param_list = []
score_list = []
# Nested inner CV to get results on 4 different validation sets and then choose the best model.
for i in range(4):
    print("Nested fold: "+str(i))
    avail_bins = []
    # In each run choose a validation bin
    val_bin = all_bins[i]
    for num in all_bins:
        if num != val_bin:
            avail_bins.append(num)
    
    print("validation bin: "+str(val_bin))
    print("train available bins:")
    print(avail_bins)
   
    # get the source and the target genes associated with hormones for the training bins in this run
    src_train_data = dict()
    for hormone in src_hormone_gene_bins[str(val_bin)].keys():
        src_train_data[hormone] = []
        for bin_no in avail_bins:
            for gene in src_hormone_gene_bins[str(bin_no)][hormone]:
                src_train_data[hormone].append(gene)

    tgt_train_data = dict()
    for hormone in tgt_hormone_gene_bins[str(val_bin)].keys():
        tgt_train_data[hormone] = []
        for bin_no in avail_bins:
            for gene in tgt_hormone_gene_bins[str(bin_no)][hormone]:
                tgt_train_data[hormone].append(gene)
             
    # get the SMOTE oversampled dataset
    X_train, y_train, train_marked = get_oversampled_train_data(src_train_data, tgt_train_data, dup_genes)
    
    # get the validation dataset from the validation bin
    src_val_data = src_hormone_gene_bins[str(val_bin)]
    tgt_val_data = tgt_hormone_gene_bins[str(val_bin)]
    X_val_src = transform_X_values(src_val_data,train_marked)
    X_val_tgt = transform_X_values(tgt_val_data,train_marked)
    X_val = np.concatenate([X_val_src,X_val_tgt])
    y_val_src = np.ones((X_val_src.shape[0],), dtype=int)
    y_val_tgt = np.zeros((X_val_tgt.shape[0],), dtype=int)
    y_val = np.concatenate([y_val_src, y_val_tgt])

    print("Validation shape")
    print(X_val.shape)
    print(y_val.shape)
    
    # execute the function to calculate the scores for multiple paramter combinations in parallel.
    # n_jobs = -1 will use all the avaliable cores. set the cores as per availabilty
    parallel = Parallel(n_jobs=-1,verbose=2)
    svm_results = parallel(delayed(get_svm_estimator_results)(X_train,y_train,X_val,y_val,par)for par in svm_param_lst)

    rf_results = parallel(delayed(get_rf_estimator_results)(X_train,y_train,X_val,y_val,par)for par in rf_param_lst)
    
    results = svm_results + rf_results
    
    # append the score obtained for each parameter combination in this run
    for pair in results:
        model_res_dict[pair[0]].append(pair[1])

# for each parameter combination get the mean score across the 4 validation sets.
model_scores = {}
for param_comb in model_res_dict.keys():
    model_scores[param_comb] = mean(model_res_dict[param_comb])
    
with open('./BioEmbedS_TS/output/model_scores_fold_'+str(test_bin)+'.json', 'w') as outfile:
    json.dump(model_scores,outfile)
outfile.close()

# select the parameter combination having the best average score as our final classifier.
best_combo = max(model_scores.items(), key=operator.itemgetter(1))
best_score = best_combo[1]
best_param = best_combo[0]
classifier_type = best_param.split("/")[0]
print("best parameters: "+best_param)
print("best validation score: "+str(best_score))

if classifier_type == 'svm':
    param = {}
    kernel_type = best_param.split("/")[1]
    param['kernel'] = kernel_type
    param['C'] = float(best_param.split("/")[2])
    if kernel_type == 'rbf':
        param['gamma'] = float(best_param.split("/")[3])
    else:
        param['degree'] = int(best_param.split("/")[3])
    classifier = SVC()
    classifier.set_params(**param)
    print(classifier.get_params())
else:
    param = {}
    param['bootstrap'] = best_param.split("/")[1]
    param['max_depth'] = best_param.split("/")[2]
    param['max_features'] = best_param.split("/")[3]
    param['min_samples_leaf'] = best_param.split("/")[4]
    param['min_samples_split'] = best_param.split("/")[5]
    param['n_estimators'] = best_param.split("/")[6]
    classifier = RandomForestClassifier()
    classifier.set_params(**param)
    print(classifier.get_params())

pickle.dump(classifier, open('./BioEmbedS_TS/models/fold_'+str(test_bin)+'_model.sav', 'wb'))

# get the source and the target genes associated with each hormone in the training+validation bins. 
# Oversample it using SMOTE and then fit our selected model to this dataset.
src_train_val_data = dict()
for hormone in src_hormone_gene_bins[str(val_bin)].keys():
    src_train_val_data[hormone] = []
    for bin_no in all_bins:
        for gene in src_hormone_gene_bins[str(bin_no)][hormone]:
            src_train_val_data[hormone].append(gene)

tgt_train_val_data = dict()
for hormone in tgt_hormone_gene_bins[str(val_bin)].keys():
    tgt_train_val_data[hormone] = []
    for bin_no in all_bins:
        for gene in tgt_hormone_gene_bins[str(bin_no)][hormone]:
            tgt_train_val_data[hormone].append(gene)
            
X_train_all, y_train_all, _train_marked = get_oversampled_train_data(src_train_val_data, tgt_train_val_data, dup_genes)
np.save('./BioEmbedS_TS/dataset/X_train_val_fold_'+str(test_bin)+'.npy',X_train_all)
np.save('./BioEmbedS_TS/dataset/y_train_val_fold_'+str(test_bin)+'.npy',y_train_all)
with open('./BioEmbedS_TS/dataset/train_val_marking_fold_'+str(test_bin)+'.json', 'w') as outfile:
    json.dump(train_marked,outfile)
outfile.close()

classifier.fit(X_train_all,y_train_all)
y_pred_train = classifier.predict(X_train_all)
print("Training results: fold-"+str(test_bin))
print(confusion_matrix(y_train_all, y_pred_train))
print(classification_report(y_train_all, y_pred_train))

# get the test data considering the test bin
src_test_data = src_hormone_gene_bins[str(test_bin)]
tgt_test_data = tgt_hormone_gene_bins[str(test_bin)]
X_test_src = transform_X_values(src_test_data,_train_marked)
X_test_tgt = transform_X_values(tgt_test_data,_train_marked)
X_test = np.concatenate([X_test_src,X_test_tgt])
y_test_src = np.ones((X_test_src.shape[0],), dtype=int)
y_test_tgt = np.zeros((X_test_tgt.shape[0],), dtype=int)
y_test = np.concatenate([y_test_src, y_test_tgt])

print("Test shape")
print(X_test.shape)
print(y_test.shape)

# get results on the test set
y_pred_test = classifier.predict(X_test)
if classifier_type == 'svm':
    y_dec_score_test = classifier.decision_function(X_test)
else:
    y_dec_score_test = classifier.predict_proba(X_test)
print("Testing results: fold-"+str(test_bin))
print(cohen_kappa_score(y_test,y_pred_test))
print(confusion_matrix(y_test, y_pred_test))
print(classification_report(y_test, y_pred_test))
np.save('./BioEmbedS_TS/output/ts_y_fold_'+str(test_bin)+'.npy',y_test)
np.save('./BioEmbedS_TS/output/ts_y_pred_fold_'+str(test_bin)+'.npy',y_pred_test)
np.save('./BioEmbedS_TS/output/ts_y_dec_score_fold_'+str(test_bin)+'.npy',y_dec_score_test)
print("ROC-AUC score: "+str(roc_auc_score(y_test, y_dec_score_test)))
precision, recall, _ = precision_recall_curve(y_test, y_dec_score_test)
print("PR-AUC score: "+str(auc(recall, precision)))
get_binned_results(src_test_data, tgt_test_data, _train_marked, classifier)

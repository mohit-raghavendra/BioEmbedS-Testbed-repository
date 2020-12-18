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

#a dict containing the hormone and the list of genes associated with it in HGv1 database
with open('./BioEmbedS/dataset/hgv1_hormone_genes.json') as json_file:
    hormone_genes = json.load(json_file)
    
# This is a list of genes that are associated with multiple hormones
dup_genes = []
with open('./BioEmbedS/dataset/genes_associated_with_multiple_hormones.txt','r') as f:
    for line in f:
        dup_genes.append(line[:-1])
    
# get embedding for hormones having aliases by add the original and alias embedding
alias_embeddings = dict()
for hormone in hormone_genes.keys():
    if "/" in hormone:
        parts = hormone.split("/")
        w1 = model.get_word_vector(parts[0])
        w2 = model.get_word_vector(parts[1])
        alias_embeddings[hormone] = np.add(w1,w2)

# Below two functions fit the classifier for given set of parameters and return the result on validation set.
# Add/Remove/Update these functions to try with different classifiers.

# function to fit the SVM model with given set of parameters and resutn the Kappa score
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

# function to fit the Random Forest model with given set of parameters and resutn the Kappa score
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

# Apply SMOTE-Tomek links oversampling+undersampling methodology to get a balanced dataset (Same #genes associated with every hormone)
# Takes input a dictionary with hormones and list of genes associated with them and the list of genes associated with multiple hormones.
# Returns the trainig sets and a dict indicating which hormones are present in the training sets
def get_oversampled_train_data(train_data,dup_genes):
    train_marked = dict()
    hor_map = dict()
    X_train_smote = []
    y_train_smote = []
    eligible_genes = dict()
    duplicate_genes = dict()
    cnt = 1
    # get the list of genes for each hormone after removing genes which are associated with multiple hormones (eligible genes).
    for hormone in train_data.keys():
        eligible_genes[hormone] = []
        duplicate_genes[hormone] = []
        for gene in train_data[hormone]:
            if gene in dup_genes:
                duplicate_genes[hormone].append(gene)
            else:
                eligible_genes[hormone].append(gene)
        
        # Consider a hormone only if it has atleast 3 eligible genes (constraint to apply SMOTE) and mark this gene.
        if len(eligible_genes[hormone]) >= 3:
            hor_map[cnt] = hormone
            train_marked[hormone] = 1
            for gene in eligible_genes[hormone]:
                X_train_smote.append(model.get_word_vector(gene))
                y_train_smote.append(cnt)
            cnt += 1
        else:
            train_marked[hormone] = 0

    # tranform dataset using smote-tomek
    smote_strategy = SMOTETomek(random_state=42, smote=SMOTE(k_neighbors=2))
    X_dataset_src_oversampled, y_dataset_src_oversampled = smote_strategy.fit_resample(np.array(X_train_smote),np.array(y_train_smote))
    #counter = Counter(y_dataset_src_oversampled)
    #print(counter)
    
    oversampled_genes_pos = dict()
    oversampled_genes_neg = dict()
    X_train_pos = []
    # get the positive oversampled train data
    for hormone, embedding in zip(y_dataset_src_oversampled, X_dataset_src_oversampled):
        if hor_map[hormone] in oversampled_genes_pos.keys():
            oversampled_genes_pos[hor_map[hormone]].append(embedding)   
        else:
            oversampled_genes_pos[hor_map[hormone]] = [embedding]
        if "/" in hor_map[hormone]:
            w1 = alias_embeddings[hor_map[hormone]]
        else:
            w1 = model.get_word_vector(hor_map[hormone])
        X_train_pos.append(np.concatenate([w1,embedding]))

    # add back the genes associated with multiple hormones that were removed earlier
    for hor in oversampled_genes_pos.keys():
        if "/" in hor:
            w1 = alias_embeddings[hor]
        else:
            w1 = model.get_word_vector(hor)
        for gene in duplicate_genes[hor]:
            w2 = model.get_word_vector(gene)
            X_train_pos.append(np.concatenate([w1,w2]))

    #get negative train data, randomly sample from oversamples embeddings known to be not associated with a hormone
    X_train_neg = []
    for hormone in oversampled_genes_pos.keys():
        if hormone not in oversampled_genes_neg.keys():
            oversampled_genes_neg[hormone] = []

        cnt = len(oversampled_genes_pos[hormone]) + len(duplicate_genes[hormone])
        if "/" in hormone:
            w1 = alias_embeddings[hormone]
        else:
            w1 = model.get_word_vector(hormone)
        rem_genes_embed = []
        for hor in oversampled_genes_pos.keys():
            if hor != hormone:
                for embed in oversampled_genes_pos[hor]:
                    rem_genes_embed.append(embed)
                    
        random.seed(42)
        for embedding in random.sample(rem_genes_embed, cnt):
            oversampled_genes_neg[hormone].append(embedding)
            X_train_neg.append(list(w1)+list(embedding))

    X_train_pos = np.array(X_train_pos)
    X_train_neg = np.array(X_train_neg)
    X_train = np.concatenate([X_train_pos, X_train_neg])

    y_train_pos = np.ones((X_train_pos.shape[0],), dtype=int)
    y_train_neg = np.zeros((X_train_neg.shape[0],), dtype=int)
    y_train = np.concatenate([y_train_pos,y_train_neg])
    
    print("train shape")
    #print(X_train_pos.shape)
    #print(X_train_neg.shape)
    print(X_train.shape)
    print(y_train.shape)
    return X_train, y_train, train_marked

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

# same functionality as the above function but also filters genes based on the bins they are assigned to
def transform_X_values_new(data_dict, bins, train_marked):
    embeddings = []
    for hormone in data_dict.keys():
        if hormone in bins: 
            if train_marked[hormone] == 1:
                if "/" in hormone:
                    np1 = alias_embeddings[hormone]
                else:
                    np1 = model.get_word_vector(hormone)
                for gene in data_dict[hormone]:
                    np2 = model.get_word_vector(gene)
                    embeddings.append(np.concatenate([np1,np2]))
    return np.array(embeddings)

# function that divides the hormones into 4 bins depending on the #genes associated. 
# Prints the results for each bins considering the predictions for hormones that are assigned to that bin.
def get_binned_results(test_data, neg_test_data, train_marked, svclassifier):
    gene_cnt = []
    for hormone in test_data.keys():
        if train_marked[hormone] == 1:
            length = len(test_data[hormone])
            gene_cnt.append(length)
        
    df = pd.DataFrame(gene_cnt)
    res,bin_edges = pd.qcut(df[0], q=4,retbins=True)
    bin1 = []
    bin2 = []
    bin3 = []
    bin4 = []
     
    # divide hormones into bins
    for hormone in test_data.keys():
        length = len(test_data[hormone])
        if length <= int(bin_edges[2]):
            bin1.append(hormone)
        elif length > int(bin_edges[2]) and length<= int(bin_edges[3]):
            bin2.append(hormone)
        elif length > int(bin_edges[3]) and length< int(bin_edges[4]):
            bin3.append(hormone)
        else:
            bin4.append(hormone)

    all_bins = []
    all_bins.append(bin1)
    all_bins.append(bin2)
    all_bins.append(bin3)
    all_bins.append(bin4)
    
    # get bin wise results
    for i, bin_name in zip(range(4),all_bins):
        print("Testing results for bin "+str(i+1))
        X_test_pos = transform_X_values_new(test_data, bin_name,train_marked)
        X_test_neg = transform_X_values_new(neg_test_data, bin_name,train_marked)
        X_test = np.concatenate([X_test_pos,X_test_neg])
        y_test_pos = np.ones((X_test_pos.shape[0],), dtype=int)
        y_test_neg = np.zeros((X_test_neg.shape[0],), dtype=int)
        y_test = np.concatenate([y_test_pos, y_test_neg])
        print("bin_size")
        print(X_test.shape)
        y_pred_test = svclassifier.predict(X_test)
        print(confusion_matrix(y_test, y_pred_test))
        print(classification_report(y_test, y_pred_test))
    
# The genes associated with every hormone are divided into 5 bins. The below files contain the genes associated with a hormone
# for each of the 5 bins in a dictionary form. Bins are also present for negatively associated genes.
with open('./BioEmbedS/dataset/hgv1_hormone_gene_bins.json') as json_file:
    hormone_gene_bins = json.load(json_file)
    
with open('./BioEmbedS/dataset/hgv1_negative_hormone_gene_bins.json') as json_file:
    neg_hormone_gene_bins = json.load(json_file)
    
param_str_set = []
rbf_param_list = []

# Set the range of parameters for SVM classifier
C_range = np.logspace(-4, 4, 9)
poly_C_range = np.logspace(-4, 0, 5)
gamma_range = np.logspace(-9, 2, 12)
degree_range = [2, 3, 5, 7]
for C in C_range:
    for gamma in gamma_range:
        rbf_param_list.append(('rbf',C,gamma))
        param_str_set.append("svm/rbf/"+str(C)+"/"+str(gamma))

poly_param_list = []
for C in poly_C_range:
    for degree in degree_range:
        poly_param_list.append(('poly',C,degree))
        param_str_set.append("svm/poly/"+str(C)+"/"+str(degree))
        
svm_param_lst = rbf_param_list + poly_param_list

# Set the range of parameters for Random forest classifier.
bootstrap = [False]
max_depth = [120, 140, 160, 210, 220, 230, 320, 340, 360]
max_features = ['auto', 'sqrt']
min_samples_leaf = [1, 2, 4]
min_samples_split = [2, 3, 5, 7]
n_estimators = [100, 200, 300, 1000, 1200, 1400, 1600]
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

train_marked_list = []
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
    
    # get the genes associated with hormones for the trianing bins in this run
    train_data = dict()
    for hormone in hormone_gene_bins[str(val_bin)]:
        train_data[hormone] = []
    for bin_no in avail_bins:
        for hormone in hormone_gene_bins[str(bin_no)].keys():
            for gene in hormone_gene_bins[str(bin_no)][hormone]:
                train_data[hormone].append(gene)
    
    # get SMOTE oversampled dataset
    X_train, y_train, train_marked = get_oversampled_train_data(train_data,dup_genes)
    train_marked_list.append(train_marked)
    
    #min_max_scaler = MinMaxScaler()
    #min_max_scaler.fit(X_train)
    #X_train = min_max_scaler.transform(X_train)
    
    # get the validation dataset from the validation bin
    val_data = hormone_gene_bins[str(val_bin)]
    neg_val_data = neg_hormone_gene_bins[str(val_bin)]
    X_val_pos = transform_X_values(val_data,train_marked)
    X_val_neg = transform_X_values(neg_val_data,train_marked)
    X_val = np.concatenate([X_val_pos,X_val_neg])
    y_val_pos = np.ones((X_val_pos.shape[0],), dtype=int)
    y_val_neg = np.zeros((X_val_pos.shape[0],), dtype=int)
    y_val = np.concatenate([y_val_pos, y_val_neg])
    #X_val = min_max_scaler.transform(X_val)
    
    print("Validation shape")
    print(X_val.shape)
    print(y_val.shape) 

    # execute the function to calculate the scores for multiple paramter combinations in parallel.
    # n_jobs = -1 will use all the avaliable cores. set the cores as per availabilty
    parallel = Parallel(n_jobs=25,verbose=2)
    svm_results = parallel(delayed(get_svm_estimator_results)(X_train,y_train,X_val,y_val,par)for par in svm_param_lst)
    
    rf_results = parallel(delayed(get_rf_estimator_results)(X_train,y_train,X_val,y_val,par)for par in rf_param_lst)
    
    print("Done with gridsearch")
    results = svm_results + rf_results
    
    # append the score obtained for each parameter combination in this run
    for pair in results:
        model_res_dict[pair[0]].append(pair[1])
        
# for each parameter combination get the mean score across the 4 validation sets.
model_scores = {}
for param_comb in model_res_dict.keys():
    model_scores[param_comb] = mean(model_res_dict[param_comb])

with open('./BioEmbedS/output/bioembeds_model_scores_fold_'+str(test_bin)+'.json', 'w') as outfile:
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
    
pickle.dump(classifier, open('./BioEmbedS/models/bioembeds_fold_'+str(test_bin)+'_model.sav', 'wb'))

# get the genes associated with each hormone in the trianing+validation bins. 
# Oversample it using SMOTE and then fit our selected model to this dataset.
train_val_data = dict()
for hormone in hormone_gene_bins[str(test_bin)]:
    train_val_data[hormone] = []
for bin_no in all_bins:
    for hormone in hormone_gene_bins[str(bin_no)].keys():
        for gene in hormone_gene_bins[str(bin_no)][hormone]:
            train_val_data[hormone].append(gene)
            
X_train_all, y_train_all, _train_marked = get_oversampled_train_data(train_val_data, dup_genes)

#save the datasets
np.save('./BioEmbedS/dataset/bioembeds_X_train_val_fold_'+str(test_bin)+'.npy',X_train_all)
np.save('./BioEmbedS/dataset/bioembeds_y_train_val_fold_'+str(test_bin)+'.npy',y_train_all)
with open('./BioEmbedS/dataset/train_val_marking_fold_'+str(test_bin)+'.json', 'w') as outfile:
    json.dump(train_marked,outfile)
outfile.close()

classifier.fit(X_train_all,y_train_all)
y_pred_train = classifier.predict(X_train_all)
print("Training results: fold-"+str(test_bin))
print(confusion_matrix(y_train_all, y_pred_train))
print(classification_report(y_train_all, y_pred_train))
    
# get the test data considering the test bin
test_data = hormone_gene_bins[str(test_bin)]
neg_test_data = neg_hormone_gene_bins[str(test_bin)]
X_test_pos = transform_X_values(test_data,_train_marked)
X_test_neg = transform_X_values(neg_test_data,_train_marked)
X_test = np.concatenate([X_test_pos,X_test_neg])
y_test_pos = np.ones((X_test_pos.shape[0],), dtype=int)
y_test_neg = np.zeros((X_test_pos.shape[0],), dtype=int)
y_test = np.concatenate([y_test_pos, y_test_neg])
#X_test = min_max_scaler.transform(X_test)

# get results on the test set
y_pred_test = classifier.predict(X_test)
if classifier_type == 'svm':
    y_dec_score_test = classifier.decision_function(X_test)
else:
    y_dec_score_test = classifier.predict_proba(X_test)
print("Testing results: fold-"+str(test_bin))
print("Kappa score: "+str(cohen_kappa_score(y_test,y_pred_test)))
print(confusion_matrix(y_test, y_pred_test))
print(classification_report(y_test, y_pred_test))
np.save('./BioEmbedS/output/y_fold_'+str(test_bin)+'.npy',y_test)
np.save('./BioEmbedS/output/y_pred_fold_'+str(test_bin)+'.npy',y_pred_test)
np.save('./BioEmbedS/output/y_dec_score_fold_'+str(test_bin)+'.npy',y_dec_score_test)
print("ROC-AUC score: "+str(roc_auc_score(y_test, y_dec_score_test)))
precision, recall, _ = precision_recall_curve(y_test, y_dec_score_test)
print("PR-AUC score: "+str(auc(recall, precision)))
get_binned_results(test_data, neg_test_data, _train_marked, classifier)
print("\n")

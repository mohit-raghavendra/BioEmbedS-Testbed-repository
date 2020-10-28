import json
import csv
import numpy as np
import math
import fasttext
from sklearn.svm import SVC
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report, confusion_matrix
    
with open('./BioEmbedS/dataset/hgv1_hormone_genes.json') as json_file:
    hormone_genes = json.load(json_file)
    
with open('./BioEmbedS/dataset/hgv1_hormone_gene_bins.json') as json_file:
    hormone_gene_bins = json.load(json_file)
    
with open('./BioEmbedS/dataset/hgv1_negative_hormone_gene_bins.json') as json_file:
    neg_hormone_gene_bins = json.load(json_file)

model = fasttext.load_model("BioWordVec_PubMed_MIMICIII_d200.bin")

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

alias_embeddings = dict()
for hormone in hormone_genes.keys():
    if "/" in hormone:
        parts = hormone.split("/")
        w1 = model.get_word_vector(parts[0])
        w2 = model.get_word_vector(parts[1])
        alias_embeddings[hormone] = np.add(w1,w2)
        
for i in range(5):
    X_train = np.load('./BioEmbedS/dataset/bioembeds_X_train_val_fold_'+str(i)+'.npy')
    y_train = np.load('./BioEmbedS/dataset/bioembeds_y_train_val_fold_'+str(i)+'.npy')
    with open('./BioEmbedS/dataset/train_val_marking_fold_'+str(i)+'.json') as json_file:
        train_marked = json.load(json_file)
    classifier = SVC(kernel='poly',C=1.0,degree=3)
    classifier.fit(X_train,y_train)
    
    test_data = hormone_gene_bins[str(i)]
    neg_test_data = neg_hormone_gene_bins[str(i)]
    X_test_pos = transform_X_values(test_data,train_marked)
    X_test_neg = transform_X_values(neg_test_data,train_marked)
    X_test = np.concatenate([X_test_pos,X_test_neg])
    y_test_pos = np.ones((X_test_pos.shape[0],), dtype=int)
    y_test_neg = np.zeros((X_test_pos.shape[0],), dtype=int)
    y_test = np.concatenate([y_test_pos, y_test_neg])
    
    y_pred_test = classifier.predict(X_test)
    y_dec_score_test = classifier.decision_function(X_test)
    
    print("Testing results: fold-"+str(i))
    print(cohen_kappa_score(y_test,y_pred_test))
    print(confusion_matrix(y_test, y_pred_test))
    print(classification_report(y_test, y_pred_test))
    print("ROC-AUC score: "+str(roc_auc_score(y_test, y_dec_score_test)))
    precision, recall, _ = precision_recall_curve(y_test, y_dec_score_test)
    print("PR-AUC score: "+str(auc(recall, precision)))
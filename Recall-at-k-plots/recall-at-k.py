import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys



#hormone = "insulin"
hormone = sys.argv[1]
predictions_file = sys.argv[2]

predictions = pd.read_csv(predictions_file)

if(hormone == "insulin"):
    insulin_novel_predictions = (predictions.loc[((predictions['Hormone']=='insulin') & ((predictions['Is pair in Hgv1?']=='No') & (predictions['Gene type'] == 'gene with protein product') )) & ((predictions['Prediction'] == 1))]).sort_values(by = 'SVM score', ascending=False)
    insulin_predictions = (predictions.loc[((predictions['Hormone']=='insulin')) & ((predictions['Prediction'] == 1) & (predictions['Gene type'] == 'gene with protein product'))]).sort_values(by = 'SVM score', ascending=False)
    disgenet_insulin = pd.read_csv("./DisGeNET_table_insulin.txt", delimiter="\t")
    print(np.shape(disgenet_insulin))
    disgenet_insulin_no_cut_off = pd.read_csv("./DisGeNET_table_noinsulin_cutoff.txt", delimiter="\t")
    print(np.shape(disgenet_insulin_no_cut_off))
    df_novel = disgenet_insulin_no_cut_off
    df_pred_novel = insulin_novel_predictions
    df = disgenet_insulin
    df_pred = insulin_predictions
    disease = "Diabetes Mellitus"
    all_disease_genes = np.loadtxt("./Diabetes_Mellitus_from_DisGeNET.txt", dtype=str)
    k_limit = np.shape(df_pred)[0]
    #k_limit = 6000
elif(hormone == "estrogen"):
    estrogen_novel_predictions = (predictions.loc[(((predictions['Hormone']=='estrogen')) & ((predictions['Is pair in Hgv1?']=='No') & (predictions['Gene type'] == 'gene with protein product'))) & (predictions['Prediction'] == 1)]).sort_values(by = 'SVM score', ascending=False)
    estrogen_predictions = (predictions.loc[((predictions['Hormone']=='estrogen')) & ((predictions['Prediction'] == 1) & (predictions['Gene type'] == 'gene with protein product'))]).sort_values(by = 'SVM score', ascending=False)
    disgenet_estrogen = pd.read_csv("./DisGeNET_table_estrogen.txt", delimiter="\t")
    print(np.shape(disgenet_estrogen))
    disgenet_estrogen_no_cut_off = pd.read_csv("./DisGeNET_table_noestrogen_cutoff.txt", delimiter="\t")
    print(np.shape(disgenet_estrogen_no_cut_off))
    df_novel = disgenet_estrogen_no_cut_off
    df_pred_novel = estrogen_novel_predictions
    df = disgenet_estrogen
    df_pred = estrogen_predictions
    disease = "Fanconi Anemia"
    k_limit = 10000
    all_disease_genes = np.loadtxt("./Fanconi_Anemia_from_DisGeNET.txt", dtype=str)
    k_limit = np.shape(df_pred)[0]
elif(hormone == "adrenaline"):
    adrenaline_novel_predictions = (predictions.loc[((predictions['Hormone']=='adrenaline/epinephrine') & ((predictions['Is pair in Hgv1?']=='No') & (predictions['Gene type'] == 'gene with protein product'))) & (predictions['Prediction'] == 1)]).sort_values(by = 'SVM score', ascending=False)
    adrenaline_predictions = (predictions.loc[(predictions['Hormone']=='adrenaline/epinephrine') & ((predictions['Prediction'] == 1)& (predictions['Gene type'] == 'gene with protein product'))]).sort_values(by = 'SVM score', ascending=False)
    disgenet_adrenaline = pd.read_csv("./DisGeNET_table_adrenaline.txt", delimiter="\t")
    print(np.shape(disgenet_adrenaline))
    disgenet_adrenaline_no_cut_off = pd.read_csv("./DisGeNET_table_noadrenaline_cutoff.txt", delimiter="\t")
    print(np.shape(disgenet_adrenaline_no_cut_off))
    df_novel = disgenet_adrenaline_no_cut_off
    df_pred_novel = adrenaline_novel_predictions
    df = disgenet_adrenaline
    df_pred = adrenaline_predictions
    disease = "Hypertensive disease"
    k_limit = 1500
    all_disease_genes = np.loadtxt("./Hypertensive_disease_from_DisGeNET.txt", dtype=str)
    k_limit = np.shape(df_pred)[0]
elif(hormone == "leptin"):
    leptin_novel_predictions = (predictions.loc[((predictions['Hormone']=='leptin') & (predictions['Is pair in Hgv1?']=='No')) & ((predictions['Prediction'] == 1)& (predictions['Gene type'] == 'gene with protein product'))]).sort_values(by = 'SVM score', ascending=False)
    leptin_predictions = (predictions.loc[(predictions['Hormone']=='leptin') & ((predictions['Prediction'] == 1)& (predictions['Gene type'] == 'gene with protein product'))]).sort_values(by = 'SVM score', ascending=False)
    disgenet_leptin = pd.read_csv("./DisGeNET_table_leptin.txt", delimiter="\t")
    print(np.shape(disgenet_leptin))
    disgenet_leptin_no_cut_off = pd.read_csv("./DisGeNET_table_noleptin_cutoff.txt", delimiter="\t")
    print(np.shape(disgenet_leptin_no_cut_off))
    df_novel = disgenet_leptin_no_cut_off
    df_pred_novel = leptin_novel_predictions
    df = disgenet_leptin
    df_pred = leptin_predictions
    disease = "Metabolic Syndrome X"
    k_limit = 2000
    all_disease_genes = np.loadtxt("./Metabolic_Syndrome X_from_DisGeNET.txt", dtype=str)
    k_limit = np.shape(df_pred)[0]


disease_genes = np.intersect1d(all_disease_genes, df_pred["Gene"].tolist())
disease_genes_novel = np.intersect1d(all_disease_genes, df_pred_novel["Gene"].tolist())

k_vector = np.zeros((np.shape(df_pred)[0],))
k_vector_novel = np.zeros((np.shape(df_pred_novel)[0],))

count = 0
genes_found = 0
for index, row in df_pred.iterrows():
    if(row['Gene'] in disease_genes):
        genes_found = genes_found + 1
    k_vector[count] = genes_found
    count = count + 1
k_vector


count = 0
genes_found = 0
for index, row in df_pred_novel.iterrows():
    if(row['Gene'] in disease_genes):
        genes_found = genes_found + 1
    k_vector_novel[count] = genes_found
    count = count + 1
k_vector_novel


random_vector = np.zeros_like(k_vector)
random_vector_novel = np.zeros_like(k_vector_novel)

for i in range(np.shape(random_vector)[0]):
    random_vector[i] = (len(disease_genes)/np.shape(random_vector)[0]) * (i + 1)
count_vector = (np.arange(np.shape(random_vector)[0])+1)

for i in range(np.shape(random_vector_novel)[0]):
    random_vector_novel[i] = (len(disease_genes_novel)/np.shape(random_vector_novel)[0]) * (i + 1)
count_vector_novel = (np.arange(np.shape(random_vector_novel)[0])+1)



sns.set(style="white")
#data = k_vector[:10000]
#data2 = random_vector[:10000] 
sns_plot = sns.lineplot(data=k_vector[:k_limit], color='red', label = 'all predictions')
sns_plot.lines[0].set_linestyle("-")
sns_plot = sns.lineplot(data=k_vector_novel[:k_limit], color = 'black', label = 'novel predictions')
sns_plot.lines[1].set_linestyle("-")
sns_plot = sns.lineplot(data=random_vector[:k_limit], color = 'red')
sns_plot.lines[2].set_linestyle("--")
sns_plot = sns.lineplot(data=random_vector_novel[:k_limit], color = 'black')
sns_plot.lines[3].set_linestyle("--")


sns_plot.set_title("Disease: " + disease)
sns_plot.set_ylabel('No. of disease genes recovered')
#sns_plot.set_xlabel("Top predictions for " + hormone.capitalize() + " by BioEmbedS" + "\n" + "All predictions: " + str(len(disease_genes)) + "/" + str(np.shape(df_pred)[0]) + "; Novel predictions: " + str(len(disease_genes_novel)) + "/" + str(np.shape(df_pred_novel)[0]))
sns_plot.set_xlabel("Top predictions for " + hormone.capitalize()+ " by BioEmbedS")

sns_plot.figure.savefig("./outputs/" + disease + ".pdf")
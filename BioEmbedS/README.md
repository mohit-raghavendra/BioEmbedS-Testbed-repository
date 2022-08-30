Copyright 2022 BIRDS Group, IIT Madras

This file is part of BioEmbedS project

BioEmbedS is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

BioEmbedS is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with BioEmbedS.  If not, see <https://www.gnu.org/licenses/>.


The results for this Machine learning setup are reported using 5-fold cross validation. This folder contains a seperate file for each fold.
The entire dataset is already divided into 5 bins considering only the hormones having atleast 5 genes. One set of bins is for postive (associated) genes and the other for negative (non-associated or random) genes. Make sure the folder structure for reading the input and writing the output is correct depending on the path from which you execute the script.

- Run the BioEmbedS_fold0.py file to run the experiments. Make sure to change the test_bin number among [0,1,2,3 or 4] in the script to run the required fold. 
The models obtained on each of the 5 folds are saved in the models folder. Also the predictions made for every fold are saved in the output folder.
Used to generate Suppl. Table S1 and S4
- To get the results on applying BioEmbedS using BioBERT embeddings, make the following changes to BioEmbedS_fold0.py file and run.
  Comment lines:   51, 52, 121, 146, 155, 158, 172, 209, 212, 226, 229
  Uncomment lines: 53, 54, 122, 147, 156, 159, 173, 210, 213, 227, 230
  1. For using 768D BioBERT embeddings, uncomment lines 31 and 32 and comment line 30
  2. For using 200D BioBERT embeddings, uncomment lines 33 and 34 and comment line 30
Running files after these changes result in Suppl. Table S3


- BioEmbedS_disease_gene_predictions.py
Run this file to obtain the performance metrics for the disease-gene predictions task on EUADR dataset using fastText word emeddings from BioWordVec.
A 10-fold cross validation method is used to summarize the results.
Used to generate Suppl. Table 3


- cosine_similarity_analysis.py 
Run this file to visualize the performance comparison of our BioEmbedS model vs. cosine similarity method. The same fastText word embeddings are used to obtain word vectors in both the methods.
It uses already generated predictions obtained from running BioEmbedS_fold0.py file.
Used to generate Suppl. Figure S1

- human_vs_mouse 
This file generates a scatter-plot to visualize the similarity between positive mouse genes and human genes.
used to generate Figure 4.

- results_on_out_of_training_hormones.py
Run this file to obtain results on applying BioEmbedS model on hormones which are not used in training and hence are completely unseen.
used to generate Figure 3(b).
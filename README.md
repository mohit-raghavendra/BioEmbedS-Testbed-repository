# Predicting cross-tissue hormone-gene relations using balanced word embeddings.

This is the *testbed* repository of the paper "Predicting cross-tissue hormone-gene relations using balanced word embeddings." by Aditya Jadhav, Tarun Kumar+, Mohit Raghavendra+, Tamizhini Loganathan, and Manikandan Narayanan (+ equal contributions). 

The code was developed by Aditya Jadhav, Mohit R, and Manikandan Narayanan (in consultation with the other co-authors of the paper listed above), and these developers are jointly referred to as the "BIRDS Group, IIT Madras" in the preamble of all code files in this BioEmbedS project.  


## License preamble 

Copyright 2022 BIRDS Group, IIT Madras

BioEmbedS is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

BioEmbedS is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with BioEmbedS.  If not, see <https://www.gnu.org/licenses/>.

## Installation

See ```requirements.txt``` file for the list of dependencies. Setup everything by executing  
```pip install -r requirements.txt```.

## Running

1. Navigate to either BioEmbedS or BioEmbedS-TS  
2. Follow the instructions in the README file of the appropriate folder. 

## Results

Results are available at the project webpage https://cross-tissue-signaling.herokuapp.com/

## Code file mappings in Main text:

--Figures
FIG 3A -> STRING_comparison/STRING-unequal_hormones_out_of_14.ipynb
FIG 3B -> BioEmbedS/results_on_out_of_training_hormones.ipynb
FIG 3C -> BioEmbedS_TS/generate_roc_pr_curves.ipynb
FIG 3D -> BioEmbedS_TS/out_of_training_hormones_analysis.ipynb
FIG 4 -> BioEmbedS/human_vs_mouse.ipynb
FIG 6 -> ???

--Table
Table 3 -> BioEmbedS/BioEmbedS_disease_gene_predictions.py
Table 4 -> BioEmbedS_TS/BioEmbedS_TS_fold_num.py (Replace num with numbers from 0 to 4, to get results for every fold)
TABLE 5 -> ???

		   
## Code file mappings in Supplement:

--Figures
FIG S1: BioEmbedS/cosine_similarity_analysis.ipynb

--Table
Table S1: BioEmbedS/BioEmbedS_fold0.py (Replace num with numbers from 0 to 4, to get results for every fold)
Table S2: ???
Table S3: 
BioBERT(200D) -> averaged across the 5 folds. files:BioEmbedS_biobert/BioEmbedS_fold0.py
BioBERT(768D) -> averaged across the 5 folds. files:BioEmbedS_biobert_700/BioEmbedS_fold0.py

Table S4: Average across 5 folds. File: BioEmbedS/BioEmbedS_fold0.py (Replace num with numbers from 0 to 4, to get results for every fold)
Table S5: Average across 5 folds. File: BioEmbedS_TS/BioEmbedS_TS_fold_num.py (Replace num with numbers from 0 to 4, to get results for every fold)

## Links to BioBERT_200D files

BioBERT_200D: Google drive link
STRING comparision: Google drive link


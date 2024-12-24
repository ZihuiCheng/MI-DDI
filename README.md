# MI-DDI
## Install  
python==3.7  
pytorch 0.11.0+cu111  
torch_geometric 2.0.0  
rdkit 2023.3.2  
numpy  
pandas  
math  
sklearn  
copy 

## Usage  
### Input
The sample data is in the data folder, including the number in the DrugBank database, corresponding SMILES sequences, and the type of DDI.

### Train and test
Command: python train.py &lt;parameters&gt;  
If you want to change the dataset, please modify the code in data_preprocessing.py

### Output
In MI-DDI, you can get the interaction map and the prediction labels.

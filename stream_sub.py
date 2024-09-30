import numpy as np
import pandas as pd
import torch
from torch.utils import data
import json
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors as rdDesc
from sklearn.preprocessing import OneHotEncoder

from subword_nmt.apply_bpe import BPE
import codecs

# vocab_path = './ESPF/drug_codes_chembl.txt'
# bpe_codes_protein = codecs.open(vocab_path)
# pbpe = BPE(bpe_codes_protein, merges=-1, separator='')
# sub_csv = pd.read_csv('./ESPF/subword_units_map_chembl.csv')
#
# idx2word_p = sub_csv['index'].values
# words2idx_p = dict(zip(idx2word_p, range(0, len(idx2word_p))))

#vocab_path = './ESPF/drug_codes_chembl.txt'
vocab_path = './drug_codes_chembl.txt'
bpe_codes_drug = codecs.open(vocab_path)
dbpe = BPE(bpe_codes_drug, merges=-1, separator='')
sub_csv = pd.read_csv('./subword_units_map_chembl.csv')

idx2word_d = sub_csv['index'].values
words2idx_d = dict(zip(idx2word_d, range(0, len(idx2word_d))))

max_d = 205
# max_p = 545
#
#
# def protein2emb_encoder(x):
#     max_p = 545
#     t1 = pbpe.process_line(x).split()  # split
#     try:
#         i1 = np.asarray([words2idx_p[i] for i in t1])  # index
#     except:
#         i1 = np.array([0])
#         # print(x)
#
#     l = len(i1)
#
#     if l < max_p:
#         i = np.pad(i1, (0, max_p - l), 'constant', constant_values=0)
#         input_mask = ([1] * l) + ([0] * (max_p - l))
#     else:
#         i = i1[:max_p]
#         input_mask = [1] * max_p
#
#     return i, np.asarray(input_mask)


def drug2emb_encoder(x,maxd):
    ii = []
    input_maskii = []
    for kk in range(len(x)):
        max_d = maxd
        # max_d = 100
        t1 = dbpe.process_line(x[kk]).split()  # split
        try:
            i1 = np.asarray([words2idx_d[i] for i in t1])  # index
        except:
            i1 = np.array([0])
            # print(x)

        l = len(i1)

        if l < max_d:
            i = np.pad(i1, (0, max_d - l), 'constant', constant_values=0)
            input_mask = ([1] * l) + ([0] * (max_d - l))

        else:
            i = i1[:max_d]
            input_mask = [1] * max_d

        ii.append(i)
        input_maskii.append(input_mask)
    return ii, input_maskii


class BIN_Data_Encoder(data.Dataset):

    def __init__(self, list_IDs, labels, df_dti):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.df = df_dti

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        # Load data and get label
        index = self.list_IDs[index]

        # d = self.df.iloc[index]['DrugBank ID']
        d = self.df.iloc[index]['smiles_1']
        p = self.df.iloc[index]['smiles_2']

        # d_v = drug2single_vector(d)
        d_v, input_mask_d = drug2emb_encoder(d)
        p_v, input_mask_p = drug2emb_encoder(p)
        # print(d_v.shape)
        # print(input_mask_d)
        # print(p_v.shape)
        # print(input_mask_p.shape)
        y = self.labels[index]
        #
        # print(d)
        # print(p)

        return np.asarray(d_v), np.asarray(p_v), np.asarray(input_mask_d), np.asarray(input_mask_p),y

    # , np.asarray(d_node_features), np.asarray(p_node_features),

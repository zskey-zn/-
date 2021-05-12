# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 16:54:42 2020

@author: lenovo
"""
import rdkit
from rdkit import Chem , DataStructs 
from rdkit.Chem import Draw,QED
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import BRICS,AllChem, Draw, MACCSkeys
import pandas as pd
cov=pd.read_csv('../data/cov_data.csv')
warhead=pd.read_csv('../data/warhead.csv')
w=list(zip(warhead['Warhead（弹头）'],warhead['W_SMILES']))
w_dict=dict(w)
warhead_name=list(warhead['Warhead（弹头）'])
tail=[]
for i in warhead_name:
    print(i)
    a=cov[cov['warhead']==i]
    cov_smiles=a['smiles']
    head=Chem.MolFromSmiles(w_dict[i])
    cluster=[Chem.MolFromSmiles(i) for i in cov_smiles]
    b=[AllChem.DeleteSubstructs(cluster[j],head) for j in range(len(cluster))]
    b=[Chem.MolToSmiles(j) for j in b]
    b=list(set(b))
    tail.extend(b)
    tail=list(set(tail))


dot=[i for i in tail if '.' in i]
tail=[i for i in tail if '.'  not in i]
for i in dot:
    res=i.strip('.')
    tail.extend(res)
tail=[i for i in tail if i]
tail=list(set(tail))

f=open('../data/cov_tail.smi','a')
[f.write(i+'\n') for i in tail]
f.close()

f=open('../data/cov_head.smi','a')
[f.write(w_dict[i]+'\n') for i in warhead_name]
f.close()   

































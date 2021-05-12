# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 17:22:29 2021

@author: lenovo
"""
import numpy as np
import pandas as pd
from rdkit import Chem,DataStructs 
from rdkit.Chem import PandasTools,AllChem,Lipinski,Crippen,Descriptors
from rdkit.Chem.Draw import IPythonConsole
from rdkit import Chem , DataStructs 
from rdkit.Chem import Draw,QED
from rdkit.Chem import BRICS,AllChem, MACCSkeys
import random
 
f=open('sample.smi','r')
data=f.read().split()
f.close()
mol=[Chem.MolFromSmiles(i) for i in data]
mol=[i for i in mol if i]
fp=[MACCSkeys.GenMACCSKeys(i) for i in mol]#生成分子指纹
cluster=[]
similar=0.95#相似度
#生成相似度矩阵，把大于相似度的分子归于一类
for i in range(len(fp)):
    s=set()
    for j in range(i+1,len(fp)):
        if DataStructs.DiceSimilarity(fp[i],fp[j])>=similar:
            s.update((i,j))
    if s!=set():
        cluster.append(s)
#把有相同元素的类和为一类
for i in range(len(cluster)):
    for j in range(i+1,len(cluster)):
        if cluster[i]&cluster[j]:
            cluster[i].union(cluster[j])
            cluster[j]=set()
cluster=[i for i in cluster if i!=set()]
allfrag=[]#除余后的碎片
for i in range(len(cluster)):
    c=sorted(list(cluster[i]))
    allfrag.append(data[c[0]])
seq_mol=[Chem.MolFromSmiles(i) for i in allfrag]#
HD=[Lipinski.NumHDonors(i) for i in seq_mol]
HA=[Lipinski.NumHAcceptors(i) for i in seq_mol]
MW=[Descriptors.ExactMolWt(i) for i in seq_mol]
logP=[Descriptors.MolLogP(i) for i in seq_mol]
RO3=[]
for i in range(len(seq_mol)):
    if HD[i]<= 3 and MW[i]<=300 and logP[i]<=3 and HA[i]<=3:
        RO3.append(1)
    else:
        RO3.append(0)
fiter=[allfrag[i] for i in range(len(RO3)) if RO3[i]==1]
def randomSmiles(mol):
    #打乱原子的排布顺序从而得到具有同样意义的SMILES
    mol.SetProp("_canonicalRankingNumbers", "True")
    idxs = list(range(0,mol.GetNumAtoms()))
    random.shuffle(idxs)#打乱列表
    for i,v in enumerate(idxs):
        mol.GetAtomWithIdx(i).SetProp("_canonicalRankingNumber", str(v))
    return Chem.MolToSmiles(mol)
def smile_augmentation(smile, augmentation, min_len, max_len):
    #增强SMILES
    mol = Chem.MolFromSmiles(smile)
    s = set()
    for i in range(1000):
        smiles = randomSmiles(mol)
        if len(smiles)<=max_len:
            s.add(smiles)
            if len(s)==augmentation:
                break
    return list(s)
new=[set() for i in fiter]
for i in range(len(fiter)):
    a=smile_augmentation(fiter[i],20,0,150)
    new[i].update(a)
f=open('./data/tail_filter.smi','r')
l=f.read().split()
l=set(l)
f.close()

nn=[]
for i in range(len(fiter)):
    if len(new[i]-l)==len(new[i]):
        nn.append(fiter[i])



f=open('new20000.smi','w')
[f.write(i+'\n') for i in nn]
f.close()




"""

19028
1952
156
136
"""

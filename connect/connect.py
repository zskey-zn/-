# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 19:34:53 2021

@author: lenovo
"""
import sys
import re
import numpy as np
import pandas as pd
from rdkit import Chem,DataStructs 
from rdkit.Chem import PandasTools,AllChem,Lipinski,Crippen,Descriptors,QED
from rdkit.Chem.Draw import IPythonConsole
from rdkit import Chem , DataStructs 
from rdkit.Chem import Draw,QED
from rdkit.Chem import BRICS,AllChem, MACCSkeys
import random
from math import exp
def random_pick(some_list, probabilities):
    '''
    特定概率取样
    '''
    x = random.uniform(0,1) 
    cumulative_probability = 0.0 
    for item, item_probability in zip(some_list, probabilities): 
        cumulative_probability += item_probability 
        if x < cumulative_probability:break 
    return item 
def softmax(a_list):
    a_exp=[exp(i) for i in a_list]
    a_sum=sum(a_exp)
    a_lv=[i/a_sum for i in a_exp]
    return(a_lv) 
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
def mis_score(score,chem_str):
    if chem_str=='#':
        score-=3
    elif chem_str=='=':
        score-=2
    else:
        score-=1
    return(score)
def find_saturated_carbon_atom(smiles):
    if re.match('c12',smiles):
        postion=(0,3)
    elif re.match('c1',smiles):
        postion=(1,2)
    elif re.match('C12',smiles):
        postion=(1,3)
    elif re.match('C1',smiles):
        postion=(2,2)
    elif re.match('C',smiles):
        postion=(4,1)
    saturated=postion[0]
    p=re.finditer('\(',smiles)
    p=[m.span()[0] for m in p]
    n=re.finditer('\)',smiles)
    n=[m.span()[0] for m in n]
    try:
        if p[0]!=postion[1] or p==[]:
            saturated=mis_score(saturated,smiles[postion[1]])
        else:
            saturated=mis_score(saturated,smiles[p[0]+1])
            bracket_st=len([s for s in p if s<n[0]])
            if smiles[n[bracket_st-1]+1]=='(':
                saturated=mis_score(saturated,smiles[n[bracket_st-1]+2])
                bracket_nd=len([s for s in p[bracket_st:] if s<n[bracket_st]])
                saturated=mis_score(saturated,smiles[n[bracket_st+bracket_nd-1]+1])
            else:
                saturated=mis_score(saturated,smiles[n[bracket_st-1]+1])
    except:
        saturated=saturated
    if saturated<=0:
        return(1)
    else:
        return(0)
    
warhead=pd.read_csv('warhead.csv')
f=open('new20000.smi','r')
tail=f.read().split()
f.close()
tail=sorted(tail)
tail_aug=set()
for i in range(len(tail)):
    a=smile_augmentation(tail[i],20,0,150)
    tail_aug.update(a)
tail_aug=list(tail_aug)
tail_aug=[i for i in tail_aug if i[0]=='C' or i[0]=='c']
tail_aug=[i for i in tail_aug if i.find('\\')==-1]#去掉 \ 的smiles
#识别饱和碳原子
tail=[]
for i in range(len(tail_aug)):
    print(i)
    if find_saturated_carbon_atom(tail_aug[i])==0:
        tail.append(tail_aug[i])
tail_mol=[Chem.MolFromSmiles(i) for i in tail]
tail_MW=[1/Descriptors.ExactMolWt(i) for i in tail_mol]
tail_lv=softmax(tail_MW)
tail_mol=[]
tail_MW=[]
link=list(warhead['link'])
num=int(sys.argv[1])
a=link[num]
def connect(con,tails):
    n=len(re.findall('\[R\]',con))
    result=con
    for i in range(n):
        rd_tail=random_pick(tail,tail_lv)
        #print(f'con: {result}  ')
        #print(f'tail: {rd_tail}')
        result=re.sub('\[R\]',rd_tail,result,1)
    return(result)

mol_list=set()
epoch=int(sys.argv[2])
for j in range(epoch):
    n_c=0
    print("start run epoch {}\n".format(j))
    for epoch in range(1000):
        try:
            smi=connect(a,tail) 
            mol=Chem.MolFromSmiles(smi)
            if mol:
                mol_list.add(mol)
        except:
            n_c+=1
        print("epoch {} end\nWaild:\t{}".format(j,1-n_c/1000))

mol_list=list(mol_list)
HD=[Lipinski.NumHDonors(i) for i in mol_list]
HA=[Lipinski.NumHAcceptors(i) for i in mol_list]
MW=[Descriptors.ExactMolWt(i) for i in mol_list]
logP=[Descriptors.MolLogP(i) for i in mol_list]
smiles_list=[Chem.MolToSmiles(i) for i in mol_list]
RO5=[]
for i in range(len(mol_list)):
    if HD[i]<= 5 and MW[i]<=500 and logP[i]<=5 and HA[i]<=10:
        RO5.append(1)
    else:
        RO5.append(0)
qed=[QED.qed(i) for i in mol_list]
f=open('./new/new_covlent.csv','w')
f.write('warhead,smiles,RO5,QED\n')
for i in range(len(HA)):
    f.write(a+','+smiles_list[i]+','+str(RO5[i])+','+str(qed[i])+'\n')
f.close()















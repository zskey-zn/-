# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 16:01:50 2020

@author: lenovo
"""
import sys
import re
import numpy as np
import pandas as pd
from rdkit import Chem,DataStructs 
from rdkit.Chem import PandasTools,AllChem,Lipinski,Crippen,Descriptors,MACCSkeys
from rdkit.Chem.Draw import IPythonConsole
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn import svm
cov=pd.read_csv('cov_despcript.csv')
non=pd.read_csv('non_descript.csv')
X=cov.append(non)
X=np.mat(X)
Y=[]
for i in range(len(X)):
    if i<len(cov):
        Y.append(1)
    else:
        Y.append(0)
#构建分类器
clf = svm.SVC(kernel='rbf',gamma='auto') 
clf.fit(X,Y)
#预测
a=[clf.predict(X[i]) for i in range(len(X))]
#测试
import random
r_c=[random.randint(0,len(cov)) for i in range(200)]
r_c=list(set(r_c))[:100]
test=[clf.predict(X[i]) for i in r_c]
smiles_file=sys.argv[1]
f=open(smiles_file,'r')
smiles_list=f.read().split()
f.close()
mol=[Chem.MolFromSmiles(i) for i in smiles_list]
HD=[Lipinski.NumHDonors(i) for i in mol]
HA=[Lipinski.NumHAcceptors(i) for i in mol]
MW=[Descriptors.ExactMolWt(i) for i in mol]
logP=[Descriptors.MolLogP(i) for i in mol]
a=[HD,HA,MW,logP]
a=np.mat(a)
a=a.T
result=clf.predict(a)
f=open("svm_out.csv",'w')
f.write("smiles,cov\n")
for i in range(len(a)):
    f.write(smiles_list[i]+','+str(result[i])+'\n')
f.close()




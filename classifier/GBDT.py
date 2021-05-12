# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 11:53:10 2021

@author: lenovo
"""
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
gbdt = GradientBoostingClassifier(loss='deviance', learning_rate=0.90, n_estimators=95, subsample=1
                                  , min_samples_split=2, min_samples_leaf=1, max_depth=4
                                  , init=None, random_state=None, max_features=None
                                  , verbose=0, max_leaf_nodes=None, warm_start=False
                                  )

'''
调参：
loss：损失函数。有deviance和exponential两种。deviance是采用对数似然，exponential是指数损失，后者相当于AdaBoost。
n_estimators:最大弱学习器个数，默认是100，调参时要注意过拟合或欠拟合，一般和learning_rate一起考虑。
learning_rate:步长，即每个弱学习器的权重缩减系数，默认为0.1，取值范围0-1，当取值为1时，相当于权重不缩减。较小的learning_rate相当于更多的迭代次数。
subsample:子采样，默认为1，取值范围(0,1]，当取值为1时，相当于没有采样。小于1时，即进行采样，按比例采样得到的样本去构建弱学习器。这样做可以防止过拟合，但是值不能太低，会造成高方差。
init：初始化弱学习器。不使用的话就是第一轮迭代构建的弱学习器.如果没有先验的话就可以不用管

由于GBDT使用CART回归决策树。以下参数用于调优弱学习器，主要都是为了防止过拟合
max_feature：树分裂时考虑的最大特征数，默认为None，也就是考虑所有特征。可以取值有：log2,auto,sqrt
max_depth：CART最大深度，默认为None
min_sample_split：划分节点时需要保留的样本数。当某节点的样本数小于某个值时，就当做叶子节点，不允许再分裂。默认是2
min_sample_leaf：叶子节点最少样本数。如果某个叶子节点数量少于某个值，会同它的兄弟节点一起被剪枝。默认是1
min_weight_fraction_leaf：叶子节点最小的样本权重和。如果小于某个值，会同它的兄弟节点一起被剪枝。一般用于权重变化的样本。默认是0
min_leaf_nodes：最大叶子节点数
'''
import sys
import re
import random
import pandas as pd
from rdkit import Chem,DataStructs 
from rdkit.Chem import PandasTools,AllChem,Lipinski,Crippen,Descriptors,MACCSkeys
from rdkit.Chem.Draw import IPythonConsole
import matplotlib.pyplot as plt
cov=pd.read_csv('cov_despcript.csv')
non=pd.read_csv('non_descript.csv')
train_feat=cov.append(non)
train_feat=np.mat(train_feat)
train_label=[]
for i in range(len(train_feat)):
    if i<len(cov):
        train_label.append(1)
    else:
        train_label.append(0)
train_label = np.array(train_label).ravel()
def cal_f1(l,n):
    gbdt = GradientBoostingClassifier(loss='deviance', learning_rate=l/100, n_estimators=n, subsample=1
                                  , min_samples_split=2, min_samples_leaf=1, max_depth=4
                                  , init=None, random_state=None, max_features=None
                                  , verbose=0, max_leaf_nodes=None, warm_start=False
                                  )
    gbdt.fit(train_feat, train_label)
    a=[]
    for i in train_feat:
        pre=gbdt.predict(i)
        a.append(pre)
    PT=sum(1 for i in range(len(cov)) if a[i]==train_label[i])
    NT=sum(1 for i in range(len(non)) if a[-i-1]==train_label[-i-1])
    R=PT/4500
    P=PT/(5993-NT+PT)
    f1=2/(1/P+1/R)
    return(P,R,f1)
gbdt.fit(train_feat, train_label)
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
result=gbdt.predict(a)
f=open("gbdt_out.csv",'w')
f.write("smiles,cov\n")
for i in range(len(a)):
    f.write(smiles_list[i]+','+str(result[i])+'\n')
f.close()









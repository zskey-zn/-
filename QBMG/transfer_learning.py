#!/usr/bin/env python
import torch
from torch.utils.data import DataLoader
import pickle
from rdkit import Chem
from tqdm import tqdm
import data_struct as ds
from data_struct import MolData, Vocabulary
from data_struct import Variable, decrease_learning_rate
from model import RNN
import sys

def Transfer(restore_from = None):
    """Trains the Prior RNN(深度递归神经网络)"""

    voc = Vocabulary(init_from_file="./Voc")
    moldata = MolData("tl_filtered.smi", voc)#Dataset containing 2239 structures.A custom PyTorch dataset for training the Prior
    data = DataLoader(moldata, batch_size=32, shuffle=True, drop_last=True,
                      collate_fn=MolData.collate_fn)#把数据读入模本
    Prior = RNN(voc)
    
    # Can restore from a saved RNN
    if restore_from:
        Prior.rnn.load_state_dict(torch.load(restore_from))#读入ckpt

    optimizer = torch.optim.Adam(Prior.rnn.parameters(), lr = 0.001)#create 优化器
    for epoch in range(1, 101):
        for step, batch in tqdm(enumerate(data), total=len(data)):#len(data)=69:	2239/69=32.4?

            # Sample from DataLoader
            seqs = batch.long()#按照voc格式的smi

            # Calculate loss
            log_p, _ = Prior.likelihood(seqs)#log_p:32个数 对应 batch_size=32，优先的可能性
            loss = - log_p.mean()

            # Calculate gradients and take a step
            optimizer.zero_grad()#梯度置零
            loss.backward()#反向传播
            optimizer.step()#对所有的参数进行更新

            # Every 2 epoch we decrease learning rate and print some information
            if epoch % 2 == 0 and step == 1:
                #decrease_learning_rate(optimizer, decrease_by=0.03)
                decrease_learning_rate(optimizer, decrease_by=0.03)
            if epoch % 10 == 0 and step == 1:
                tqdm.write("*" * 50)#50个*            
                tqdm.write("Epoch {:3d}   step {:3d}    loss: {:5.2f}\n".format(epoch, step, loss.item()))
                seqs, likelihood, _ = Prior.sample(100)
                valid = 0
                f = open('tran_output.smi', 'a')
                for i, seq in enumerate(seqs.cpu().numpy()):
                    smile = voc.decode(seq)
                    if Chem.MolFromSmiles(smile):#是否符合SMILES
                        valid += 1
                        f.write(smile + "\n")
                    if i < 10:
                        tqdm.write(smile)
                f.close()
                tqdm.write("\n{:>4.1f}% valid SMILES".format(100 * valid / len(seqs)))
                tqdm.write("*" * 50 + "\n")               
        # Save the Prior
        torch.save(Prior.rnn.state_dict(), "data/tail.ckpt")

if __name__ == "__main__":
    smiles_file = sys.argv[1]
    voc_file = './Voc'
    print("Reading smiles...")
    smiles_list = ds.canonicalize_smiles_from_file(smiles_file)
    ds.filter_file_on_chars(smiles_list,voc_file)#产生tl_filtered.smi
    Transfer('data/tail.ckpt')

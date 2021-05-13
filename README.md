



example:
1,classifier
cd classifier
python GBDT.py test.smi
python SVM.py test.smi
2,RNN train and sample
cd ../QBMG
python transfer_learning.py ../data/cov_tail.smi
python sample.py ./data/tail.ckpt 20000
3,connect and evaluation
cd ../connect
python connect.py 1 100


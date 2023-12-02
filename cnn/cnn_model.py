import cnnLayers_v2 as layer
import imgPreprocess
import genConvNet
import pickle as pkl
import optimizer as opt
import cv2
import numpy as np
import gradCheck
if __name__ == '__main__':
    import sys
    from os import path
    print(path.dirname(path.dirname(path.abspath(__file__))))
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from data.mnist import load_mnist
(x_train, t_train), (x_test,t_test)= \
    load_mnist(normalize=True,flatten=False,one_hot_label=True) # x_train.shape=(60000,1,28,28)
train_loss_list=[]
iters_num=1000
train_size=x_train.shape[0]
batch_size=100
# model structure: data->A->A->A->A->A(with no pool)->B->Affine->Loss
# 1. Data Preprocessing
#try:
#    with open('/pkl/data/mean.pickle','rb') as fr:
#        mean=pkl.load(fr)
#except:
#     print('GEN: pkl/data/mean.pickle')
#     with open('/pkl/data/mean.pickle','wb') as fw:
#         mean=imgPreprocess.meanTotal(batch)
#         pkl.dump(mean,fw)
# batch=imgPreprocess.zeromean(batch,mean)
# 2. updating W /project/python_project/carDetect/project/cnn/pkl/model/data
rootDir='/project/python_project/carDetect/project/cnn/'
try:
    with open(rootDir+'/pkl/model/modelA.pickle','rb') as fr:
        A,optA=pkl.load(fr)
except FileNotFoundError:
    print('GEN: pkl/model/modelA.pickle')
    A1=layer.modelA(30,5,5,1,2)
    A=[A1]
    optA1=opt.Momentum()
    optA=[optA1]
    with open(rootDir+'/pkl/model/modelA.pickle','wb') as fr:
        pkl.dump([A,optA],fr)
try:
    with open(rootDir+'/pkl/model/modelB.pickle','rb') as fr:
        B,optB=pkl.load(fr)
except FileNotFoundError:
    print('GEN: pkl/model/modelB.pickle')
    B=layer.modelB(12*12*30,100)
    optB=opt.Momentum()
    with open(rootDir+'/pkl/model/modelB.pickle','wb') as fr:
        pkl.dump([B,optB],fr)
try:
    with open(rootDir+'/pkl/model/modelC.pickle','rb') as fr:
        C,optC=pkl.load(fr)
except FileNotFoundError:
    print('GEN: pkl/model/modelC.pickle')
    C=layer.modelC(100,10)
    optC=opt.Momentum()
    with open(rootDir+'/pkl/model/modelC.pickle','wb') as fr:
        pkl.dump([C,optC],fr)
neuralNet=genConvNet.genNet(A,B,C)
infoA1=[(batch_size*28,28),5,5,1,1,((0,0),(0,0))]
infoA=[infoA1]    # dxShape,FH,FW,C,s,p=InfoA
for i in range(iters_num):
    batchMask=np.random.choice(train_size,batch_size)
    batch=x_train[batchMask].transpose(0,2,3,1)
    answer=t_train[batchMask]
    L=neuralNet.forward(answer,batch,batch_size,infoA)
    print('L(',i,'): ',L)
    neuralNet.backward(batch_size,optA,optB,optC,infoA,0.01)
try:
    with open(rootDir+'pkl/model/modelA.pickle','wb') as fr:
        pkl.dump([A,optA],fr)
except FileNotFoundError:
    print('error')
try:
    with open(rootDir+'pkl/model/modelB.pickle','wb') as fr:
        pkl.dump([B,optB],fr)
except FileNotFoundError:
    print('error')
try:
    with open(rootDir+'pkl/model/modelC.pickle','wb') as fr:
        pkl.dump([C,optC],fr)
except FileNotFoundError:
    print('error')
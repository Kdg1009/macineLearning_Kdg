import cnnLayers_v2 as layer
import imgPreprocess
import genConvNet
import pickle as pkl
import optimizer as opt
import cv2
import numpy as np
# model structure: data->A->A->A->A->A(with no pool)->B->Affine->Loss
batchMask=np.random.choice(2000,10) 
batchMap=list(map(lambda x:(int(x/100),int(x%100)),batchMask)) # 100->train_set subtree file num
try:
    with open('/pkl/data/batch.pickle','rb') as fr:
        batchMat=pkl.load(fr)
except FileNotFoundError:
    print('ERROR: no batchMat !!!')
batch=[]
for bM in batchMap:
    batch.append(cv2.imread(batchMat[bM[0]][bm[1]]))
batch=np.array(batch)
N,H.W,C=batch.shape
try:
    with open('/pkl/data/answer.pickle','rb') as fr:
        answerMat=pkl.load(fr)
except FileNotFoundError:
    print('ERROR: no answerMat!!!')
answerBatch=answerMat[batchMask]
# 1. Data Preprocessing
try:
    with open('/pkl/data/mean.pickle','rb') as fr:
        mean=pkl.load(fr)
except:
    print('GEN: pkl/data/mean.pickle')
    with open('/pkl/data/mean.pickle','wb') as fw:
        mean=imgPreprocess.meanTotal(batch)
        pkl.dump(mean,fw)
batch=imgPreprocess.zeromean(batch,mean)
# 2. updating W
try:
    with open('/pkl/model/modelA.pickle','rb') as fr:
        A,optA=pkl.load(fr)
except FileNotFoundError:
    print('GEN: pkl/model/modelA.pickle')
    A1=layer.modelA()
    A2=layer.modelA()
    A3=layer.modelA()
    A4=layer.modelA()
    A5=later.modelA()
    A=[A1,A2,A3,A4,A5]
    optA1=opt.AdaGrad()
    optA2=opt.AdaGrad()
    optA3=opt.AdaGrad()
    optA4=opt.AdaGrad()
    optA5=opt.AdaGrad()
    optA=[optA1,optA2,optA3,optA4,optA5]
    pkl.dump([A,optA],'/pkl/model/modelA.pickle','wb')
try:
    with open('/pkl/model/modelB.pickle','rb') as fr:
        B,optB=pkl.load(fr)
except FileNotFoundError:
    print('GEN: pkl/model/modelB.pickle')
    B=layer.modelB()
    optB=opt.AdaGrad()
    pkl.dump([B,optB],'/pkl/model/modelB.pickle','wb')
try:
    with open('/pkl/model/modelC.pickle','rb') as fr:
        C,optC=pkl.load(fr)
except FileNotFoundError:
    print('GEN: pkl/model/modelC.pickle')
    C=layer.modelC()
    optC=opt.AdaGrad()
    pkl.dump([C,optC],'/pkl/model/modelC.pickle')
neuralNet=genConvNet.genNet(A,B,C)
infoA=[]
for i in range(1000):
    L=neuralNet.forward(answer,batch,N,infoA)
    print('L(',i,'): ',L)
    if L < 0.01:
        break
    neuralNet.backward(N,optA,optB,optC,infoA,0.01)

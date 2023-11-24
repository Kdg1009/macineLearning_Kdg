import cnnLayers_v2 as layer
import imgPreprocess
import genConvNet
import pickle as pkl
# model structure: data->A->A->A->A->A(with no pool)->B->Affine->Loss
batch=None
N,H.W,C=batch.shape
# 1. Data Preprocessing
try:
    with open('/pkl/mean.pickle','rb') as fr:
        mean=pkl.load(fr)
except:
    print('GEN: pkl/mean.pickle')
    with open('pkl/mean.pickle','wb') as fw:
        mean=imgPreprocess.meanTotal(batch)
        pkl.dump(mean,fw)
batch=imgPreprocess.zeromean(batch,mean)
# 2. updating W
try:
    with open('pkl/modelA.pickle','rb') as fr:
        A=pkl.load(fr)
except FileNotFoundError:
    print('GEN: pkl/modelA.pickle')
    A1=layer.modelA()
    A2=layer.modelA()
    A3=layer.modelA()
    A4=layer.modelA()
    A5=later.modelA()
    A=[A1,A2,A3,A4,A5]
    pkl.dump(A,'pkl/modelA.pickle','wb')
try:
    with open('pkl/modelB.pickle','rb') as fr:
        B=pkl.load(fr)
except FileNotFoundError:
    print('GEN: pkl/modelB.pickle')
    B=layer.modelB()
    pkl.dump(B,'pkl/modelB.pickle','wb')
try:
    with open('pkl/modelC.pickle','rb') as fr:
        C=pkl.load(fr)
except FileNotFoundError:
    print('GEN: pkl/modelC.pickle')
    C=layer.modelC()
    pkl.dump(C,'pkl/modelC.pickle')
answer=None
neuralNet=genConvNet.genNet(A,B,C,answer)
for i in range(1000):
    L=neuralNet.forward(answer,batch,N,5,5,3,1,0)
    neuralNet.backward(N)
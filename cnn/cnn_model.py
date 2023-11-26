import cnnLayers_v2 as layer
import imgPreprocess
import genConvNet
import pickle as pkl
import optimizer as opt
# model structure: data->A->A->A->A->A(with no pool)->B->Affine->Loss
answer=None
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
        A,optA=pkl.load(fr)
except FileNotFoundError:
    print('GEN: pkl/modelA.pickle')
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
    pkl.dump([A,opt],'pkl/modelA.pickle','wb')
try:
    with open('pkl/modelB.pickle','rb') as fr:
        B,optB=pkl.load(fr)
except FileNotFoundError:
    print('GEN: pkl/modelB.pickle')
    B=layer.modelB()
    optB=opt.AdaGrad()
    pkl.dump([B,optB],'pkl/modelB.pickle','wb')
try:
    with open('pkl/modelC.pickle','rb') as fr:
        C,optC=pkl.load(fr)
except FileNotFoundError:
    print('GEN: pkl/modelC.pickle')
    C=layer.modelC()
    optC=opt.AdaGrad()
    pkl.dump([C,optC],'pkl/modelC.pickle')
neuralNet=genConvNet.genNet(A,B,C)
infoA=[]
for i in range(1000):
    L=neuralNet.forward(answer,batch,N,infoA)
    print('L(',i,'): ',L)
    if L < 0.01:
        break
    neuralNet.backward(N,optA,optB,optC,infoA,0.01)
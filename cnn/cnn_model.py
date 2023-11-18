import cnnLayers_v2 as layer
import imgPreprocess
import pickle as pkl
# model structure: data->A->A->A->A->A(with no pool)->B->Affine->Loss
batch=None
batch.transpose(0,3,1,2) #batch.shape(N,C,H,W)
# 1. Data Preprocessing
try:
    with open('/pkl/mean.pickle','rb') as fr:
        mean=pkl.load(fr)
except:
    print('GEN: pkl/mean.pickle')
    with open('pkl/mean.pickle','wb') as fw:
        mean=imgPreprocess.meanTotal(batch)
        pkl.dump(mean,fw)
batch=imgPreprocess.zeromean(batch,mean,'total')
# 2. Forward-> 2.1 evaluate loss value
try:
    with open('pkl/modelA.pickle','rb') as fr:
        A=pkl.load(fr)
except:
    print('GEN: pkl/modelA.pickle')
    A1=layer.modelA()
    A2=layer.modelA()
    A3=layer.modelA()
    A4=layer.modelA()
    A5=later.modelA()
    A=[A1,A2,A3,A4,A5]
    pkl.dump(A,'pkl/modelA.pickle'.'wb')
try:
    with open('pkl/modelB.pickle','rb') as fr:
        B=pkl.load(fr)
except:
    print('GEN: pkl/modelB.pickle')
    B=layer.modelB()
    pkl.dump(B,'pkl/modelB.pickle','wb')
# 3. backward

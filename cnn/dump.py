import numpy as np
import cnnLayers_v2 as layer
import optimizer as opt
import pickle as pkl
from collections import OrderedDict
if __name__ == '__main__':
    import sys,os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.mnist import load_mnist
(x_train,t_train),(x_test,t_test)=load_mnist(normalize=True, flatten=False,one_hot_label=True)
convInfo={'FH':5,'FW':5,'FN':30}
layers={}
params=OrderedDict()
params['W1']=np.random.randn(25,30)*0.01
params['b1']=np.zeros(30)
layers['conv'] = layer.conv(params['W1'],params['b1'])
layers['relu1']=layer.relu()
layers['pool']=layer.pooling(2)
params['W2']=np.random.randn(4320,100)*0.01
params['b2']=np.zeros(100)
layers['aff1']=layer.Affine(params['W2'],params['b2'])
layers['relu2']=layer.relu()
params['W3']=np.random.randn(100,10)*0.01
params['b3']=np.zeros(10)
layers['aff2']=layer.Affine(params['W3'],params['b3'])
layers['loss']=layer.softmaxWithLoss()

def predict(batch,answer,layers,convInfo):
    t=layers['conv'].forward(batch,1,convInfo['FH'],convInfo['FW'],1,1)
    t=layers['relu1'].forward(t)
    t=layers['pool'].forward(t,1,convInfo['FN'])
    t=layers['aff1'].forward(t,1)
    t=layers['relu2'].forward(t)
    t=layers['aff2'].forward(t,1)
    L=layers['loss'].forward(t,answer)
    return L
dataSize=60000
batchSize=1
batchMask=np.random.choice(dataSize,batchSize)
batch=x_train[batchMask].transpose(0,2,3,1).reshape(28,28)
answer=t_train[batchMask]
backpropGrad={}
L=predict(batch,answer,layers,convInfo)
dy=layers['loss'].backward()
backpropGrad['b3']=dy
dx,dw=layers['aff2'].backward(dy,1)
backpropGrad['W3']=dw
dx=layers['relu2'].backward(dx)
backpropGrad['b2']=dx
dx,dw=layers['aff1'].backward(dx,1)
backpropGrad['W2']=dw
dx=layers['pool'].backward(dx,convInfo['FN'])
dx=layers['relu1'].backward(dx)
backpropGrad['b1']=dx
dx,dw=layers['conv'].backward(dx)
backpropGrad['W1']=dw
#print(backpropGrad)
'''
Grad={}
h=1e-4
for key in params:
    param=params.get(key)
    grad=np.zeros_like(param)
    it=np.nditer(param,flags=['multi_index'])
    while not it.finished:
        idx=it.multi_index
        tmp_val=param[idx]
        param[idx]=float(tmp_val)+h
        fxh1=predict(batch,answer,layers,convInfo)
        param[idx]=float(tmp_val)-h
        fxh2=predict(batch,answer,layers,convInfo)
        grad[idx]=(fxh1-fxh2)/(2*h)

        param[idx]=tmp_val
        it.iternext()
    print(key)
    Grad[key]=grad
try:
    with open('pkl/Grad.pickle','wb') as fr:
        pkl.dump(Grad,fr)
except:
    print('error')
print(Grad)
'''
with open('/project/python_project/carDetect/project/cnn/pkl/Grad.pickle','rb') as fr:
    Grad=pkl.load(fr)
'''
print('bp[W1]: ',backpropGrad['W1'].shape)
print('G[W1]: ',Grad['W1'].shape)
print('bp[b1]: ',backpropGrad['b1'].shape)
print('G[b1]: ',Grad['b1'].shape)
print('bp[W2]: ',backpropGrad['W2'].shape)
print('G[W2]: ',Grad['W2'].shape)
print('bp[b2]: ',backpropGrad['b2'].shape)
print('G[b2]: ',Grad['b2'].shape)
print('bp[W3]: ',backpropGrad['W3'].shape)
print('G[b3]: ',Grad['W3'].shape)'''
print('bp[b3]: ',backpropGrad['b3'])
print('G[b3]: ',Grad['b3'])

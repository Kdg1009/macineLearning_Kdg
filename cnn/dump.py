import numpy as np
import cnnLayers_v2 as layer
import imgPreprocess as img
import optimizer as opt
import cv2
from genConvNet import genNet
import gradCheck
import pickle as pkl
if __name__ == '__main__':
    import sys,os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.mnist import load_mnist
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


(x_train,t_train),(x_test,t_test)=load_mnist(normalize=True,flatten=False,one_hot_label=True)
""" data=x_train[0].transpose(1,2,0).reshape(28,28)
answer=t_train[0]
x=A[0].forward(data,1,5,5,1,1)
x=B.forward(x,1)
L=C.forward(answer,x,1)

infoA1=[(28,28),5,5,1,1,((0,0),(0,0))]
dy=C.backward(1,optC)
dy=B.backward(dy,1,optB)
dy=A[0].backward(dy,optA[0],1,infoA1)
 """
correct=0
for i in range(50):
    data=x_test[i].transpose(1,2,0).reshape(28,28)
    answer=t_test[i]
    x=A[0].forward(data,1,5,5,1,1)
    x=B.forward(x,1)
    t=C.aff.forward(x,1)
    t=C.softmax.forward(t)
    if np.argmax(t)==np.argmax(answer):
        print(i)
        correct+=1
print(correct)
import numpy as np
import cnnLayers_v2 as cnn
import imgPreprocess as img
import optimizer as opt
import cv2
from genConvNet import genNet
data=cv2.imread('/project/python_project/carDetect/project/test_set/test_1.labled/1.[[(212, 216), (290, 290)], [(340, 205), (380, 248)], [(100, 197), (142, 231)]].dfec4efae9812bb6db3de0bf9ea8ab96d0a65d3b.jpg')
data=data.reshape(427,-1)
A=cnn.modelA(5,7,7,3,3)
B=cnn.modelB(16685,3)
C=cnn.modelC(3,5)
answer=[[0,0,0,0,0],]
optA=opt.Momentum()
optB=opt.AdaGrad()
optC=opt.Adam()
InfoA=[[(427,1920),7,7,3,3,((1,1),(1,1))],] # dxShape,FH,FW,C,s,p
singleNet=genNet([A,],B,C)
for i in range(1000):
    L=singleNet.forward(answer,data,1,InfoA)
    print(L)
    #print('L(',i,'): ',L)
    if L<0.01:
        break
    singleNet.backward(1,[optA],optB,optC,InfoA,0.01)
import numpy as np
import cv2

class conv:
    def __init__(self,filter,bias):
        # F.shape=(C*FH*FW,FN)
        self.F=filter
        # b.shape=(1,FN)
        self.b=bias
        # data.shape=(N*OH*OW,FN)
        self.data=None
    def forward(self,data,batch_size,FH,FW,channel=3,stride=1,padding=0):
        # ready2dot: (N*H,W*C) => (N*OH*OW,C*FH*FW)
        self.data=ready2dot(data,stride,padding,FH,FW,channel,batch_size)
        ret=np.dot(data,self.F)+self.b
        return ret
    def backward(self,diff_y,learning_rate=0.1):
        F-=learning_rate*np.dot(self.data.T,diff_y)
        f=lambda x: x if len(x[0])==1 else x[0]+f(x[1:])
        temp=f(diff_y)/diff_y.shape[0]
        b-=learning_rate*temp
        return np.dot(diff_y,self.F.T)

def ready2dot(data,stride,padding,FH,FW,C,N):
    # data.shape=(N*H,W*C)
    # add padding
    data=np.concatenate((np.zeros((W*C,padding),dtype=int),data),axis=1)
    data=np.concatenate((data,np.zeros((W*C,padding),dtype=int)),axis=1)
    data=np.concatenate(np.zeros((padding,N*H+2*padding),dtype=int),data,axis=0)
    data=np.concatenate(data,np.zeros((padding,N*H+2*padding),dtype=int),axis=0)
    OH=N*int((H-FH+2*padding)/stride)+1
    OW=N*int((W-FW+2*padding)/stride)+1
    ret=[]
    for i in range(OH):
        temp=data[i*stride:i*stride+FH]
        for j in range(OW):
            L=temp[j*stride*C:j*stride*C+C*FW].reshape(1,C*FH*FW)
            ret.append(L)
    return ret
   
class relu:
    def __init__(self):
        self.mask=None
    def forward(self,data):
        self.mask=(data<=0)
        ret=data.copy()
        ret[self.mask]=0
        return ret
    def backward(self,diff_y):
        diff_y[self.mask]=0
        return diff_y

class pooling:
    def __init__(self,stride):
        self.stride=stride
    def forward(self,data):
        ret=[]
        H,W=data.shape
        for i in range(H):
            temp=data[i*self.stride:(i+1)*self.stride]
            L=[]
            for j in range(W):
                max_val=max(temp[j*self.stride:(j+1)*self.stride])
                L.append(max_val)
            ret.append(L)
        return ret
    def backward(self):
        return
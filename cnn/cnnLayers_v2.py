import numpy as np
import cv2
import math

class conv:
    def __init__(self,filter,bias):
        # F.shape=(C*FH*FW,FN)
        self.F=filter
        # b.shape=(1,FN)
        self.b=bias
        self.data=None
    def forward(self,data,batch_size,FH,FW,channel=3,stride=1,padding=0):
        # ready2dot: (N*H,W*C) => (N*OH*OW,C*FH*FW)
        self.data=ready2dot(data,batch_size,FH,FW,channel,stride,padding)
        ret=np.dot(data,self.F)+self.b
        self.data=ret
        return ret
    def backward(self,diff_y,learning_rate=0.1):
        F-=learning_rate*np.dot(self.data.T,diff_y)
        f=lambda x: x if len(x[0])==1 else x[0]+f(x[1:])
        temp=f(diff_y)/diff_y.shape[0]
        b-=learning_rate*temp
        return np.dot(diff_y,self.F.T)

def ready2dot(data,N,FH,FW,C=3,s=1,p=0):
    ret=[]
    # data.shape=(N*H,W*C)
    NH,WC=data.shape
    # split data into N
    H=int(NH/N)
    W=int(WC/C)
    OH=int((H-FH+2*p)/s)+1
    OW=int((W-FW+2*p)/s)+1
    jLoop=H+2*p-FH+1
    kLoop=WC+C*(2*p-FW)+1
    for i in range(0,NH,H): # loop N
        tmp=data[i:i+H] # tmp.shape=(H,WC)
        # add padding
        tmp=np.pad(tmp,p,mode='constant')
        # modify to shape(OH,OW)
        for j in range(0,jLoop,s):
            tmp2=np.array(tmp[j:j+FH]).T
            for k in range(0,kLoop,s*C):
                tmp3=tmp2[k:k+FW*C]
                ret.append(tmp3)
    ret=np.array(ret).transpose(0,2,1)
    return ret.reshape(N*OH*OW,FH*FW*C)

class relu:
    def __init__(self):
        self.mask=None
    # data.type=np.array
    def forward(self,data):
        self.mask=(data <= 0)
        data[self.mask]=0
        return data
    def backward(self,diff_y):
        diff_y[self.mask]=0
        return diff_y

class pooling:
    def __init__(self,stride):
        self.s=stride
        self.mask=None # (1,NOHOW/ss*FN)
    # using im2col
    def forward(self,data,N):
        NOHOW,FN=data.shape
        ss=self.s**2
        OHOW=int(NOHOW/N)
        tmp=ready2dot(data,N,ss,FN,C=1,s=ss).reshape(-1,ss,FN).transpose(0,2,1).reshape(-1,ss) # tmp.shape=(NOHOW/ss*FN,ss)
        ret=np.max(tmp,axis=1).reshape(-1,FN) #(NOHOW/ss,FN)
        self.mask=np.argmax(tmp,axis=1).reshape(-1,FN) #(NOHOW/ss,FN)
        return ret # ret.shape=(NOHOW/ss,FN)
    def backward(self,diff_y): # diff_y.shape=(NOHOW/ss,FN)
        H,FN=diff_y.shape
        ss=self.s**2
        tmp=np.zeros(H*ss*FN,dtype='float64') #shape(1,H*ss*FN)
        tmp2=diff_y.T.reshape(1,-1) #shape(1,H*FN)
        mask=self.mask.T.reshape(1,-1) #shape(1,H*FN)
        for i in range(H*FN):
            tmp[i*ss+mask[0][i]]=tmp2[0][i]
        ret=tmp.reshape(FN,H*ss).T
        return ret

class Affine:
    def __init__(self,W,b):
        # batch.shape=(N*BH,BW)=>(N,BH*BW)
        self.batch=None
        # W.shape=(BH*BW,FN)
        self.W=W
        # b.shape=(1,FN)
        self.b=b
    def forward(self,batch,N):
        self.batch=batch
        ret=np.dot(batch.reshape(N,-1),self.W)+self.b
        return ret
    def backward(self,diff_y,N,learning_rate=0.1):
        self.b-=learning_rate*diff_y
        self.W-=learning_rate*np.dot(self.batch.T,diff_y)
        return np.dot(diff_y,self.W.T).reshape(self.batch.shape[0],-1)
    def lossGrad(self,y):
        # y.shape=(N,H*W), x*W.shape=(N,FN)
        temp=self.batch.reshape(y.shape[0],-1)
        ret=y-np.dot(temp,self.W)*(-1*temp)
        return ret

class modelA:
    def __init__(self,FN,FH,FW,C,ps):
        filters=self.genFilters(FN,FH,FW,C) # for each channels, F init identical, but by doing gradient descent, execute separately
        bias=np.zeros((1,FN))
        self.conv=conv(filters,bias)
        self.relu=relu()
        self.pool=pooling(ps)
    # batch.shape=(N*H,W*C)
    def forward(self,batch,N,FH,FW,C=3,s=1,p=1):
        # x1.shape=(N*OH*OW,FN)
        x1=self.conv.forward(batch,N,FH,FW,C,s,p)
        x2=self.relu.forward(x1)
        # ret.shape=(N*OH*OW/s,FN/s)
        ret=self.pool.forward(x2,N)
        return ret
    # y.shape=(N*OH*OW/s,FN/s)
    def backward(self,y,lr=0.1):
        # y1.shape=(N*OH*OW,FN)
        y1=self.pool.backward(y)
        y2=self.relu.backward(y1)
        # ret.shape=(N*OH*OW,C*FH*FW)
        ret=self.conv.backward(y2,lr)
        return ret
    def genFilters(FN,FH,FW,C=3): # using He init
        # F.shape=(C*FH*FW,FN)
        tmp=np.random.randn(FN,FH*FW)*math.sqrt(2.0/FH*FW) # N=FH*FW
        ret=np.repeat(tmp,repeats=C,axis=1).T
        return ret

class modelB:
    def __init__(self,BHBW,FN):
        filters=self.genFilters(FN,BHBW)
        b=np.zeros(1,FN)
        self.aff=Affine(filters,b)
        self.relu=relu()
    def forward(self,batch,N):
        tmp=self.aff.forward(batch,N)
        ret=self.relu.forward(tmp)
        return ret
    def backward(self,diff_y,N,lr=0.1):
        tmp=self.relu.backward(diff_y)
        ret=self.aff.backward(diff_y,N,learning_rate=lr)
        return ret
    def genFilters(self,FN,BHBW):
        ret=np.random.randn(BHBW,FN)*math(1/BHBW)
        return ret

a=np.array([[1,9,17],[2,10,18],[3,12,19],[4,11,20],[5,13,21],[6,14,22],[7,15,23],[8,16,24]])
b=np.array([[4,12,20],[8,16,24]])
p=pooling(2)
print(p.forward(a,1))
print(p.backward(b))

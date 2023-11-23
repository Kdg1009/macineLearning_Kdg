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
    def forward(self,data,N,FH,FW,channel=3,stride=1,padding=0):
        NH,WC=data.shape
        # ready2dot: (N*H,W*C) => (N*OH*OW,C*FH*FW)
        OH,OW,data=ready2dot(data,N,FH,FW,channel,stride,padding)
        ret=np.dot(data,self.F)+self.b
        FN=self.F.shape[1]
        self.data=ret.reshape(N*OH,OW*FN)
        return self.data # data.shape=(N*OH,OW*FN)
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
    ret=np.array(ret).transpose(0,2,1).reshape(N*OH*OW,FH*FW*C)
    return OH,OW,ret

class relu:
    def __init__(self):
        self.mask=None
    # data.type=np.array
    def forward(self,data):
        self.mask=(data <= 0)
        data[self.mask]=0
        return np.array(data)
    def backward(self,diff_y):
        diff_y[self.mask]=0
        return diff_y

class pooling:
    def __init__(self,stride):
        self.s=stride
        self.mask=None #mask.shape=(N*OH/s*OW/s,FN)
        self.ss=stride**2
    # using im2col
    def forward(self,data,N,FN):
        NOH,OWFN=data.shape
        OH,OW,tmp=ready2dot(data,N,self.s,self.s,FN,self.s,0)
        tmp=tmp.reshape((-1,self.ss,FN)) # tmp.shape=(N*OH/s*OW/s,ss,FN)
        ret=np.max(tmp,axis=1)
        ret=ret.reshape(-1,OW*FN) # ret.shape=(N*OH/s,OW/s*FN)
        self.mask=np.argmax(tmp,axis=1).reshape(-1,FN)
        return ret

    def backward(self,diff_y,FN): # diff_y.shape=(N*OH/s,OW/s*FN)
        H,W=diff_y.shape
        tmp=np.zeros(self.ss,H*W) #tmp.shape=(ss,N*OHOW/ss*FN)
        tmp2=diff_y.reshape(1,-1) #tmp2.shape=(1,N*OHOW/ss*FN)
        for i in range(H*W):
            tmp[tmp2[int(i/FN)][int(i%FN)]][i]=tmp2[0][i]
        #tmp.shape=(NOH,OWFN)
        tmp3=tmp.T
        ret=[]
        for i in range(0,H*W,FN):
            tmp4=np.array(tmp3[i:i+FN]).reshape(self.s,-1)
            ret.append(tmp4)
        ret=np.array(ret).reshape(H*self.s,W*self.s) #(NOHOW/ss*s,FN*s)=>(NOH,OWFN)
        return ret
class Affine:
    def __init__(self,W,b):
        # batch.shape=(N*BH,BW*FN)|(N,M)
        self.batch=None
        # W.shape=(BH*BW*FN,M)
        self.W=W
        # b.shape=(1,M)
        self.b=b
    def forward(self,batch,N): # output.shape=(1,N) output has to be an vector
        self.batch=batch
        ret=np.dot(batch.reshape(N,-1),self.W)+self.b
        return ret #ret.shape=(N,M)
    def backward(self,diff_y,N,learning_rate=0.1):
        self.b-=learning_rate*diff_y
        self.W-=learning_rate*np.dot(self.batch.reshape(N,-1).T,diff_y)
        return np.dot(diff_y,self.W.T).reshape(self.batch.shape)
class LossSSE:
    def __init__(self):
        self.loss=None
        self.N=None
        self.W=None
    def forward(t,y):
        N,W=t.shape
        self.N=N
        self.W=W
        self.loss=y-t
        ret=self.loss
        ret=np.sum(ret*ret,axis=1)
        return ret/2
    def backward(dy): # dy.shape=(N,1)
        ret=np.repeat(dy,(1,W))
        ret=ret*self.loss
        return ret
    def grad(y):

class modelA:
    def __init__(self,FN,FH,FW,C,ps):
        filters=self.genFilters(FN,FH,FW,C) # for each channels, F init identical, but by doing gradient descent, execute separately
        bias=np.zeros((1,FN))
        self.conv=conv(filters,bias)
        self.relu=relu()
        self.pool=pooling(ps)
    # batch.shape=(N*H,W*C)
    def forward(self,batch,N,FH,FW,C=3,s=1,p=0):
        # x1.shape=(N*OH*OW,FN)
        x1=self.conv.forward(batch,N,FH,FW,C,s,p)
        x2=self.relu.forward(x1)
        # ret.shape=(N*OH*OW/s,FN/s)
        ret=self.pool.forward(x2,N,self.conv.F.shape[1])
        return ret
    # y.shape=(N*OH*OW/s,FN/s)
    def backward(self,y,lr=0.1):
        # y1.shape=(N*OH*OW,FN)
        y1=self.pool.backward(y,self.conv.F.shape[1])
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
    def __init__(self,BHBWFN,M):
        filters=self.genFilters(M,BHBWFN)
        b=np.zeros(1,M)
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

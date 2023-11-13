import numpy as np
import cv2

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
        tmp=np.concatenate((np.zeros((p,WC),dtype=int),tmp,np.zeros((p,WC),dtype=int)),axis=0)
        tmp=np.concatenate((np.zeros((H+2*p,C*p),dtype=int),tmp,np.zeros((H+2*p,C*p),dtype=int)),axis=1)
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
        self.mask=[]
    # using im2col
    def forward(self,data,N):
        NH,W=data.shape
        OH=int(NH/self.s)
        OW=int(W/self.s)
        tmp=ready2dot(data,N,self.s,self.s,1,self.s,0)
        ret=np.max(tmp,axis=1).reshape(OH,OW)
        self.mask=np.argmax(tmp,axis=1) # mask.shape=(1,OHOW)
        # ret.shape=(OH,OW)
        return ret
    def backward(self,diff_y):
        OH,OW=diff_y.shape
        ss=self.s*self.s
        OHOW=OH*OW
        ret=np.ones(OHOW,ss)
        diff_y=diff_y.reshape(1,OHOW)
        for i in range(OHOW):
            ret[i][self.mask[i]]=diff[i]
        ret=ret.reshape(OH*self.s,OW*self.s)
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
    def loss(self,y):
        # y.shape=(N,H*W), x*W.shape=(N,FN)
        temp=self.batch.reshape(y.shape[0],-1)
        ret=y-np.dot(temp,self.W)*(-1*temp)
        return ret

class modelA:
    def __init__(self,FH,FW,FN,ps):
        filters=genFilters(FH,FW,FN)
        bias=np.random()
        self.conv=conv(filters,bias)
        self.relu=relu()
        self.pool=pooling()
    # batch.shape=(N*H,W*C)
    def forward(self,batch,N,FH,FW,C=3,s=1,p=1):
        # x1.shape=(N*OH*OW,FN)
        x1=self.conv.forward(batch,N,FH,FW,C,s,p)
        x2=self.relu.forward(x1)
        # ret.shape=(N*OH*OW/s,FN/s)
        ret=self.pool.forward(x2)
        return ret
    # y.shape=(N*OH*OW/s,FN/s)
    def backward(self,y,lr):
        # y1.shape=(N*OH*OW,FN)
        y1=self.pool.backward(y)
        y2=self.relu.backward(y1)
        # ret.shape=(N*OH*OW,C*FH*FW)
        ret=self.conv.backward(y2,lr)
        return ret

def genFilters(FH,FW,FN):
    tmp=np.random(FN,FH*FW)

data='/project/python_project/carDetect/project/test_set/test_1.labled/1.[[(212, 216), (290, 290)], [(340, 205), (380, 248)], [(100, 197), (142, 231)]].dfec4efae9812bb6db3de0bf9ea8ab96d0a65d3b.jpg'
re=relu()
a=np.array([[1,2,3,-1],
    [4,5,-1,6],
    [1,3,4,5],
    [0,-50,2,3]])
p=pooling(2)
print(re.forward(a))
#print(p.forward(a,1))
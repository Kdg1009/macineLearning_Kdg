import numpy as np
import cv2

class conv:
    def __init__(self,filter,bias):
        # F.shape=(C*FH*FW,FN)
        self.F=filter
        # b.shape=(1,FN)
        self.b=bias
        # data.shape=(N*OH*OW,FN) *OH: about single data
    def forward(self,data,batch_size,FH,FW,channel=3,stride=1,padding=0):
        # ready2dot: (N*H,W*C) => (N*OH*OW,C*FH*FW)
        # self.data=ready2dot(data,stride,padding,FH,FW,channel,batch_size)
        self.data=ready2dot(data,batch_size,FH,FW,channel,stride,padding)
        ret=np.dot(data,self.F)+self.b
        return ret
    def backward(self,diff_y,learning_rate=0.1):
        F-=learning_rate*np.dot(self.data.T,diff_y)
        f=lambda x: x if len(x[0])==1 else x[0]+f(x[1:])
        temp=f(diff_y)/diff_y.shape[0]
        b-=learning_rate*temp
        return np.dot(diff_y,self.F.T)

# if (OH-1)s+FH=H, clean
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
        self.mask=[]
    def forward(self,data):
        ret=[]
        H,W=data.shape
        OH=int(H/self.stride)
        OW=int(W/self.stride)
        for i in range(OH):
            # stride=FH,FW.size
            temp=data[i*self.stride:(i+1)*self.stride]
            for j in range(OW):
                tmp2=temp[j*self.stride:(j+1)*self.stride]
                max_val=max(tmp2)
                # mask,shape=((H/s)*(W/s),s,s)
                self.mask.append(tmp2<max_val)
                ret.append(max_val)
        self.mask.reshape(H,W)
        return ret.reshape(OH,OW)
    def backward(self,diff_y):
        H,W=self.mask.shape
        OH=int(H/self.stride)
        OW=int(W/self.stride)
        ret=np.ones(H,W)
        for i in range(OH):
            tmp=self.mask[i*self.stride:(i+1)*self.stride]
            for j in range(OW):
                tmp2=tmp[j*self.stride:(j+1)*self.stride]
                ret[tmp2]=0
                ret[not tmp2]=diff_y[i][j]
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

im1=cv2.imread('/project/python_project/carDetect/project/test_set/test_1.labled/1.[[(212, 216), (290, 290)], [(340, 205), (380, 248)], [(100, 197), (142, 231)]].dfec4efae9812bb6db3de0bf9ea8ab96d0a65d3b.jpg')
im2=cv2.imread('/project/python_project/carDetect/project/train_set/train_1.labled/1.[[(155, 74), (563, 370)]].0b561d7ae41f9aeecfdbb9032021596d057c25c3.jpg')
data=np.array([im1,im2])
print(ready2dot(data.reshape(2*427,640*3),2,3,4,s=2,p=1).shape) # N,FH,FW,C,s,p

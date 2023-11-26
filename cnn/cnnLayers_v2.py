import numpy as np
import cv2
import math

class conv:
    def __init__(self,filter,bias):
        self.F=filter   # F.shape=(C*FH*FW,FN)
        self.b=bias     # b.shape=(1,FN)
        self.data=None  # data.shape=(NOHOW,CFHFW)
    def forward(self,data,N,FH,FW,channel=3,stride=1,padding=((0,0),(0,0))):
        NH,WC=data.shape
        OH,OW,data=ready2dot(data,N,FH,FW,channel,stride,padding) # ready2dot: (NH,WC) => (NOHOW,CFHFW)
        ret=np.dot(data,self.F)+self.b
        FN=self.F.shape[1]
        self.data=data   # data.shape=(NOHOW,CFHFW)
        return ret.reshape(N*OH,OW*FN) 
    def backward(self,dy,learning_rate=0.1):
        dy=dy.reshape(-1,self.F.shape[1])           # dy.shape=(NOH,OWFN)=>(NOHOW,FN)
        # ave b
        db=np.sum(dy,axis=0)/self.b.shape[1]
        self.b-=learning_rate*db
        # dx,dw
        dx=np.dot(dy,self.F.T) # dx.shape=(NOHOW,CFHFW)
        dF=np.dot(self.data.T,dy) #dF.shape=(CFHFW,FN)
        return dx,dF
def ready2dot(data,N,FH,FW,C=3,s=1,p=((0,0),(0,0))):
    ret=[]
    # data.shape=(N*H,W*C)
    NH,WC=data.shape
    # split data into N
    H=int(NH/N)
    W=int(WC/C)
    pcol=p[0][0]+p[0][1]
    prow=p[1][0]+p[1][1]
    OH=int((H-FH+pcol)/s)+1
    OW=int((W-FW+prow)/s)+1
    jLoop=H+pcol-FH+1
    kLoop=WC+C*(prow-FW)+1
    for i in range(0,NH,H): # loop N
        tmp=data[i:i+H] # tmp.shape=(H,WC)
        # add padding
        tmp=np.pad(tmp,((p[0][0],p[0][1]),(C*p[1][0],C*p[1][1])),mode='constant')
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
    def backward(self,dy):
        dy[self.mask]=0
        return dy
class pooling:
    def __init__(self,stride):
        self.s=stride
        self.mask=None #mask.shape=(N*OH/s*OW/s,FN)
        self.ss=stride**2
        self.pad=None
    # using im2col
    def forward(self,data,N,FN):
        NOH,OWFN=data.shape
        OH=int(NOH/N)
        OW=int(OWFN/FN)
        pcol=int((self.s-OH%self.s)%self.s)
        prow=int((self.s-OW%self.s)%self.s)
        self.pad=(N*pcol,FN*prow)
        OH,OW,tmp=ready2dot(data,N,self.s,self.s,FN,self.s,((0,pcol),(0,prow)))
        tmp=tmp.reshape((-1,self.ss,FN)) # tmp.shape=(N*OH/s*OW/s,ss,FN)
        ret=np.max(tmp,axis=1)
        ret=ret.reshape(-1,OW*FN) # ret.shape=(N*OH/s,OW/s*FN)
        self.mask=np.argmax(tmp,axis=1).reshape(-1,FN)
        return ret
    def backward(self,dy,FN): # dy.shape=(N*OH/s,OW/s*FN)
        H,W=dy.shape
        tmp=np.zeros((self.ss,H*W)) #tmp.shape=(ss,N*OHOW/ss*FN)
        tmp2=dy.reshape(1,-1) #tmp2.shape=(1,N*OHOW/ss*FN)
        for i in range(H*W):
            tmp[self.mask[int(i/FN)][int(i%FN)]][i]=tmp2[0][i]
        #tmp.shape=(NOH,OWFN)
        tmp3=tmp.T
        ret=[]
        for i in range(0,H*W,FN):
            tmp4=np.array(tmp3[i:i+FN]).reshape(self.s,-1)
            ret.append(tmp4)
        ret=np.array(ret).reshape(H*self.s,W*self.s) #(NOHOW/ss*s,FN*s)=>(NOH,OWFN)
        if self.pad[0]>0 and self.pad[1]>0:
            return ret[:-self.pad[0],:-self.pad[1]]
        elif self.pad[0]>0 and self.pad[1]==0:
            return ret[:-self.pad[0],:]
        elif self.pad[0]==0 and self.pad[1]>0:
            return ret[:,:-self.pad[1]]
        else:
            return ret
class Affine:
    def __init__(self,W,b):
        # batch.shape=(N*BH,BW*FN)
        self.batch=None
        # W.shape=(BH*BW*FN,M)
        self.W=W
        # b.shape=(1,M)
        self.b=b
    def forward(self,batch,N):
        self.batch=batch
        ret=np.dot(batch.reshape(N,-1),self.W)+self.b
        return ret
    def backward(self,dy,N,learning_rate=0.1):
        #self.b-=learning_rate*diff_y
        db=np.sum(dy,axis=0)/self.b.shape[1]
        self.b-=learning_rate*db
        #self.W-=learning_rate*np.dot(self.batch.reshape(N,-1).T,diff_y)
        return np.dot(dy,self.W.T).reshape(self.batch.shape),np.dot(self.batch.reshape(N,-1).T,dy)
class LossSSE:
    def __init__(self):
        self.loss=None
        self.N=None
        self.W=None
    def forward(self,t,y):
        N,W=t.shape
        self.N=N
        self.W=W
        self.loss=y-t
        ret=self.loss
        ret=np.sum(ret*ret,axis=1)
        return ret/2
    def backward(self,dy): # dy.shape=(N,1)
        ret=np.repeat(dy,(1,W))
        ret=ret*self.loss
        return ret
class modelA:
    def __init__(self,FN,FH,FW,C,pstride):
        filters=self.genFilters(FN,FH,FW,C) # for each channels, F init identical, but by doing gradient descent, execute separately
        bias=np.zeros((1,FN))
        self.conv=conv(filters,bias)
        self.relu=relu()
        self.pool=pooling(pstride)
    def forward(self,batch,N,FH,FW,C=3,s=1,p=((0,0),(0,0))):        # batch.shape=(N*H,W*C)
        x1=self.conv.forward(batch,N,FH,FW,C,s,p)       # x1.shape=(N*OH*OW,FN)
        x2=self.relu.forward(x1)
        ret=self.pool.forward(x2,N,self.conv.F.shape[1])# ret.shape=(N*OH*OW/s,FN/s)
        return ret
    def backward(self,y,optimizer,N,InfoA,lr=0.1):      # y.shape=(N*OH*OW/s,FN/s)
        dxShape,FH,FW,C,s,p=InfoA
        y1=self.pool.backward(y,self.conv.F.shape[1])   # y1.shape=(N*OH*OW,FN)
        y2=self.relu.backward(y1)
        dx,dF=self.conv.backward(y2,lr)
        optimizer.update(self.conv.F,dF)
        return self.col2im(dx,dxShape,N,FH,FW,C,s,p)    # dx.shape=(NH,WC)
    def genFilters(self,FN,FH,FW,C=3): # using He init / F.shape=(C*FH*FW,FN)
        tmp=np.random.randn(FN,FH*FW)*math.sqrt(2.0/FH*FW) # N=FH*FW
        ret=np.repeat(tmp,repeats=C,axis=1).T
        return ret
    def col2im(self,batch,dxShape,N,FH,FW,C=3,s=1,p=((0,0),(0,0))):
        NH,WC=dxShape
        NOHOW,CFHFW=batch.shape
        OHOW=int(NOHOW/N)
        H=int(NH/N)
        W=int(WC/C)
        pcol=p[0][0]+p[0][1]
        prow=p[1][0]+p[1][1]
        dx=np.zeros((NH+pcol*N,WC+prow*C))    # dx.shape=(N(H+pcol),C(W+prow))
        OH=int((H-FH+pcol)/s)+1
        OW=int((W-FW+prow)/s)+1
        for i in range(N):
            batTmp=batch[i*OHOW:(i+1)*OHOW]
            dxTmp=dx[i*(H+pcol):(i+1)*(H+pcol)]
            for j in range(0,OHOW,s):
                col=int(j/OW)
                row=int(j%OW)
                dxTmp[col:col+FH,row*C:C*(row+FW)]+=batTmp[j].reshape(FH,-1)
        dx=dx.reshape(N,H+pcol,-1)
        dx=dx[:,p[0][0]:-p[0][1],C*p[1][0]:-1*C*p[1][1]]
        return dx.reshape(NH,WC)
class modelB:
    def __init__(self,BHBWFN,M):
        filters=self.genFilters(BHBWFN,M)
        b=np.zeros((1,M))
        self.aff=Affine(filters,b)
        self.relu=relu()
    def forward(self,batch,N):
        tmp=self.aff.forward(batch,N)
        ret=self.relu.forward(tmp)
        return ret
    def backward(self,dy,N,optimizer,lr=0.01):
        dy=self.relu.backward(dy)
        dx,dw=self.aff.backward(dy,N,lr)
        optimizer.update(self.aff.W,dw)
        return dx
    def genFilters(self,BHBW,FN):
        ret=np.random.randn(BHBW,FN)*math.sqrt(1/BHBW)
        return ret
class modelC:
    def __init__(self,W,M):
        filters=self.genFilters(W,M)
        b=np.zeros((1,M))
        self.aff=Affine(filters,b)
        self.loss=LossSSE()
        self.y=None
        self.xw=None
        self.x=None
    def forward(self,answer,batch,N):
        tmp=self.aff.forward(batch,N)
        ret=self.loss.forward(tmp,answer)
        self.x=batch
        self.y=answer
        self.xw=tmp
        return ret
    def backward(self,N,optimizer,lr=0.01):
        dy=self.y-self.xw
        dx,dw=self.aff.backward(dy,N,lr)
        optimizer.update(self.aff.W,-1*dw)
        return -1*dx
    def genFilters(self,BHBW,FN):
        ret=np.random.randn(BHBW,FN)*math.sqrt(1/BHBW)
        return ret
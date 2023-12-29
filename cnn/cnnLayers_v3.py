import numpy as np
from cnnLayers_v2 import ready2dot
if __name__ == '__main__':
    import sys
    from os import path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from data.mnist import load_mnist

# col[N,C,FH,FW,OH,OW]
# group of datas(OH,OW) that multiplied by certain element(FH,FW) in certain Filter's channel(C) in certain data(N)
def im2col(input_data, FH, FW, s=1,p=0):
    N,C,H,W=input_data.shape
    OH=(H-FH+2*p)//s+1
    OW=(W-FW+2*p)//s+1

    img=np.pad(input_data,[(0,0),(0,0),(p,p),(p,p)], 'constant')
    col=np.zeros((N,C,FH,FW,OH,OW))

    for y in range(FH):
        y_max=y+s*OH
        for x in range(FW):
            x_max=x+s*OW
            col[:,:,y,x,:,:]=img[:,:,y:y_max:s,x:x_max:s]
    col=col.transpose(0,4,5,1,2,3).reshape(N*OH*OW,-1)
    return col

def col2im(col,input_shape,FH,FW,s=1,p=0):
    N,C,H,W=input_shape
    OH=(H+2*p-FH)//s+1
    OW=(W+2*p-FW)//s+1
    col=col.reshape(N,OH,OW,C,FH,FW).transpose(0,3,4,5,1,2)

    img=np.zeros((N,C,H+2*p+s-1,W+2*p+s-1))
    for y in range(FH):
        y_max=y+s*OH
        for x in range(FW):
            x_max=x+s*OW
            img[:,:,y:y_max:s,x:x_max:s]+=col[:,:,y,x,:,:]
    return img[:,:,p:H+p,p:W+p]
class Relu:
    def __init__(self):
        self.mask=Non
    def forward(self,x):
        self.mask=(x<=0)
        out=x.copy()
        out[self.mask]=0
        return out
    def backward(self,dy):
        dy[self.mask]=0
        dx=dy
        return dx

class Sigmoid:
    def __init__(self):
        self.out=None
    def forward(self,x):
        out=sigmoid(x)
        self.out=out
        return out
    def backward(self,dy):
        dx=dy*(1.0-self.out)*self.out
        return dx

class Affine:
    def __init__(self,W,b):
        self.W=W
        self.b=b
        self.x=None
        self.x_shape=None
        self.db=None
        self.dW=None
    def forward(self,x):
        self.x_shape=x.shape
        x=x.reshape(x.shape[0],-1)
        self.x=x
        out=np.dot(self.x,self.W)+self.b
        return out
    def backward(self,dy):
        dx=np.dot(dy,self.W.T)
        self.dW=np.dot(self.x.T,dy)
        self.db=np.sum(dy,axis=0)
        dx=dx.reshape(*self.x_shape)
        return dx

class SoftmaxWithLoss:
    def __init__(self):
        self.loss=None
        self.t=None
        self.y=None
    def forward(self,x,y):
        self.y=y
        self.t=softmax(x)
        self.loss=CEE(self.t,self.y)
        return self.loss
    def backward(self,dy=1):
        batch_size=self.t.shape[0]
        if self.t.size == self.y.size:
            dx=(self.t-self.y) / batch_size
        else:
            dx=self.t.copy()
            dx[np.arange(batch_size),self.y]-=1
            dx=dx/batch_size
        return dx

class Dropout:
    def __init__(self,d_ratio=0.5):
        self.d_ratio=d_ratio
        self.mask=None
    def forward(self,x,train_flg=True):
        assert type(train_flg) is bool
        if train_flg:
            self.mask=np.random.rand(*x.shape) > self.d_ratio # if mask <= d_ratio, then dropout
            return x*self.mask
        else:
            return x*(1.0-self.d_ratio)
    def backward(self,dy):
        return dy*self.mask

class BatchNorm:
    def __init__(self,gamma=1,beta=0,momentum=0.9,running_mean=None,running_var=None):
        self.gamma=gamma
        self.beta=beta
        self.momentum=momentum
        self.input_shape=None

        self.running_mean=running_mean
        self.running_var=running_var

        self.batch_size=None
        self.xc=None
        self.std=None
        self.dgamma=None
        self.dbeta=None
    def forward(self,x,train_flg=True):
        assert type(train_flg) is bool
        self.input_shape=x.shape
        if x.ndim != 2:
            N,C,H,W=x.shape
            x=x.reshape(N,-1)
        out = self.__forward(x,train_flg)
        return out.reshape(*self.input_shape)
    def __forward(self,x,train_flg):
        if self.running_mean is None:
            N,D = x.shape
            self.running_mean=np.zeros(D)
            self.running_var=np.zeros(D)
        if train_flg:
            mu=x.mean(axis=0)
            xc=x-mu
            var=np.mean(xc**2,axis=0)
            std=np.sqrt(var + 10e-7)
            xn = xc / std

            self.batch_size=x.shape[0]
            self.xc=xc
            self.xn=xn
            self.std=std
            self.running_mean=self.momentum * self.running_mean + (1-self.momentum) * mu
            self.running_var=self.momentum * self.running_var + (1-self.momentum) * var
        else:
            xc = x - self.running_mean
            xn = xc / ((np.sqrt(self.running_var + 10e-7)))

        out=self.gamma*xn + self.beta
        return out
    def backward(self,dy):
        if dy.ndim != 2:
            N,C,H,W = dy.shape
            dy = dy.reshape(N,-1)

        dx=self.__backward(dy)
        dx=dx.reshape(*self.input_shape)
        return dx
    def __backward(self,dy):
        dbeta=dout.sum(axis=0)
        dgamma=np.sum(self.xn*dy,axis=0)
        dxn=self.gamma * dy
        dxc=dxn/self.std
        dstd=-np.sum((dxn*self.xc)/(self.std*self.std),axis=0)
        dvar=0.5*dstd/self.std
        dxc+=(2.0/self.batch_size)*self.xc*dvar
        dmu=np.sum(dxc,axis=0)
        dx=dxc-dmu/self.batch_size

        self.dgamma=dgamma
        self.dbeta=dbeta

        return dx

class Conv:
    def __init__(self,W,b,s=1,p=0):
        self.W=W
        self.b=b
        self.s=s
        self.p=p

        self.x=None
        self.col=None
        self.col_W=None

        self.dW=None
        self.db=None
    def forward(self,x):
        FN,C,FH,FW=self.W.shape
        N,C,H,W=x.shape

        OH=1+int((H-FH+2*self.p)/self.s)
        OW=1+int((W-FW+2*self.p)/self.s)

        col=im2col(x,FH,FW,self.s,self.p)
        col_W=self.W.reshape(FN,-1).T

        out=np.dot(col,col_W) + self.b
        out=out.reshape(N,OH,OW,-1).transpose(0,3,1,2)
        self.x=x
        self.col=col
        self.col_W=col_W

        return out
    def backward(self,dy):
        FN,C,FH,FW=self.W.shape
        dy=dy.transpose(0,2,3,1).reshape(-1,FN)

        self.db=np.sum(dy,axis=0)
        self.dW=np.dot(self.col.T,dy)
        self.dW=self.dW.transpose(1,0).reshape(FN,C,FH,FW)

        dcol=np.dot(dy,self.col_W.T)
        dx=col2im(dcol,self.x.shape,FH,FW,self.s,self.p)

        return dx
class Pool:
    def __init__(self,pool_h,pool_w,s=1,p=0):
        self.pool_h=pool_h
        self.pool_w=pool_w
        self.s=s
        self.p=p
        self.x=None
        self.arg_max=None

    def forward(self,x):
        N,C,H,W=x.shape
        OH=int(1+(H-self.pool_h)/self.s)
        OW=int(1+(W-self.pool_w)/self.s)

        col=im2col(x,self.pool_h,self.pool_w,self.s,self.p)
        col=col.reshape(-1,self.pool_h*self.pool_w)

        arg_max=np.argmax(col,axis=1)
        out=np.max(col,axis=1)
        out=out.reshape(N,OH,OW,C).transpose(0,3,1,2)

        self.x=x
        self.argmax=argmax
        
        return out
    def backward(self,dy):
        dy=dy.transpose(0,2,3,1)

        pool_size=self.pool_h*self.pool_w
        dmax=np.zeros((dy.size,pool_size))
        dmax[np.arange(self.arg_max.size),self.arg_max.flatten()]=dy.flatten()
        dmax=dmax.reshape(dy.shape + (pool_size,))

        dcol=dmax.reshape(dmax.shape[0]*dmax.shape[1]*dmax.shape[2] , -1)
        dx=col2im(dcol,self.x.shape,self.pool_h,self.pool_w,self.s,self.p)
        return dx
class AvePool:
    def __init__(self,pool_h,pool_w,s=1,p=0):
        self.pool_h=pool_h
        self.pool_w=pool_w
        self.s=s
        self.p=p
        self.x=None
    def forward(self,x):
        N,C,H,W=x.shape
        OH=int(1+(H-self.pool_h)/self.s)
        OW=int(1+(W-self.pool_w)/self.s)

        col=im2col(x,self.pool_h,self.pool_w,self.s,self.p)
        col=col.reshape(-1,self.pool_h*self.pool_w)
        # ave pool
        out=np.mean(col,axis=1)
        out=out.reshape(N,OH,OW,C).transpose(0,3,1,2)
        self.x=x
        return out
    def backward(self,dy):
        dy=dy.transpose(0,2,3,1)
        # reverse average pooling
        pool_size=self.pool_h*self.pool_w
        dave=np.zeros((pool_size,dy.size))
        dave+=(1/pool_size)*dy.flatten()
        dave=dave.T.reshape(dy.shape+(pool_size,))

        dcol=dave.reshape(dave.shape[0]*dave.shape[1]*dave.shape[2],-1)
        dx=col2im(dcol,self.x.shape,self.pool_h,self.pool_w,self.s,self.p)
        return dx
from collections import OrderedDict
from googleNet import Conv2d
class conv_block:
    def __init__(self,convInfo):
        self.layers=OrderedDict()
        self.layers['conv']=Conv(*convInfo)
        self.layers['batchNorm']=BatchNorm()
        self.layers['relu']=Relu()
    def forward(self,x):
        for layer in self.layers.values():
            x=layer.forward(x)
        return x


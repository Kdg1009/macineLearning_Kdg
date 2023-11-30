import numpy as np

class SGD:
    def __init__(self,lr=0.01):
        self.lr=lr
    def update(self,params,grads):
        params-=grads*self.lr
class Momentum:
    def __init__(self,lr=0.01,momentum=0.9):
        self.lr=lr
        self.momentum=momentum
        self.v=None
    def update(self,params,grads):
        if self.v is None:
            self.v=np.zeros(params.shape)
        self.v=self.momentum*self.v-self.lr*grads
        params=params+self.v
class AdaGrad:
    def __init__(self,lr=0.01):
        self.lr=lr
        self.h=None
    def update(self,params,grads):
        if self.h is None:
            self.h=np.zeros(params.shape)
        self.h+=grads*grads
        params=params-self.lr*grads/(np.sqrt(self.h)+1e-7)
class Adam:
    def __init__(self,lr=0.01,b1=0.9,b2=0.999,e=1e-8):
        self.lr=lr
        self.b1=b1
        self.b2=b2
        self.m=None
        self.h=None
        self.t=0
        self.e=e
    def update(self,params,grads):
        if self.m is None:
            self.m=np.zeros(params.shape)
            self.h=np.zeros(params.shape)
        self.t+=1
        self.m=self.b1*self.m+(1-self.b1)*grads
        self.h=self.b2*self.h+(1-self.b2)*grads*grads
        mb=self.m/(1-self.b1**self.t)
        mh=self.h/(1-self.b2**self.t)
        params-=self.lr*mb/(np.sqrt(mh)+self.e)

import numpy as np
import cnnLayers_v2 as cnn
import imgPreprocess as img
F=np.ones((27,2),dtype='float64')
b=np.zeros((1,2),dtype='float64')
conv=cnn.conv(F,b)
re=cnn.relu()
pool=cnn.pooling(2)
# data.shape=(2,6,6,3)
data=np.arange(1,37,dtype='float64')
data2=data+1
data3=data2+1
data=np.concatenate((data,data2,data3)).reshape(3,36).T
data=np.tile(data,(2,1))
data=data.reshape(2*6,6*3)
print(data)
x1=conv.forward(data,2,3,3,3,1,0)
print(x1)
x2=re.forward(x1)
print(x2)
x3=pool.forward(x2,2,2)
print(x3)
W=np.tile([1,2],(8,1))
affine=cnn.Affine(W,np.zeros(2))
x4=affine.forward(x3,2)
print(x4)
answer=[[4960,9940],[4961,9939]]
L=cnn.Loss(x4,answer)
print(L)
class genNet:
    def __init__(self,A,B,affine):
        self.A=A
        self.B=B
        self.affine=affine
    def forward(self,batch,N,FH,FW,C=3,s=1,p=0):
        N,H,W,C=batch.shape
        x=batch.reshape(N*H,W*C)
        for layer in self.A:
            x=layer.forward(x,N,FH,FW,C,s,p)
        x=self.B.forward(x,N)
        x=self.affine.forward(x,N)
        # loss func
        # ret=loss(x,N)
        # return ret
    def backward(self,dy,N,lr):
        y=self.affine.backward(dy,N,lr)
        y=self.B.backward(y,N,lr)
        for layer in self.A:
            y=layer.backward(y,lr)
            return y

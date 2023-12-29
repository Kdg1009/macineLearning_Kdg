import numpy as np
import cnnLayers_v3 as layers
from collections import OrderedDict
import optimizer as opt
class Conv2d:
    def __init__(self,in_channels,out_channels,filter_size,stride,padding=None): # in_channels=C, out_channels=FN, filter_size=FH/FW
        self.W=(2/(filter_size**2))*np.random.randn(out_channels,in_channels,filter_size,filter_size)
        self.b=np.zeros(out_channels)
        self.s=stride
        self.layers=OrderedDict()
        self.layers['conv']=layers.Conv(self.W,self.b,s=stride,p=padding)
        self.layers['relu']=layers.Relu()
        self.layers['batchNorm']=layers.BatchNorm()
        self.x=None
    def forward(self,x):
        self.x=x
        FN,C,FH,FW=self.W.shape
        N,C,H,W=x.shape
        if self.conv.p==None:
            p=FH-int((H-FH)%self.s)
            self.conv.p=p
        for layer in self.layers.values():
            x=layer.forward(x)
        return x
    def backward(self,dy):
        dx=dy.copy()
        layers=list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dx=layer.backward(dx)
        return dx
class Linear:
    def __init__(self,in_channels,out_channels):
        self.W=(2/in_channels)*np.random.randn(in_channels,out_channels)
        self.b=np.zeros(out_channels)
        self.layers=OrderedDict()
        self.layers['aff']=layers.Affine(self.W,self.b)
        self.layers['relu']=layers.Relu()
        self.x=None
    def forward(self,x):
        self.x=x
        for layer in self.layers.values():
            x=layer.forward(x)
        return x
    def backward(self,dy):
        dx=dy.copy()
        layers=list(self.layers.valuse())
        layers.reverse()
        for layer in layers:
            dx=layer.backward(dx)
        return dx
class InceptionAux:
    def __init__(self,in_channels,num_classes):
        self.conv=OrderedDict()
        self.fc=OrderedDict()
        
        self.conv['avepool']=layers.AvePool(5,5,3)
        self.conv['conv']=Conv2d(in_channels,128,3)

        self.fc['aff1']=Linear(2048,1024)
        self.fc['aff2']=Linear(1024,num_classes)
        self.fc['dropout']=layers.Dropout(0.7)
        self.fc['relu']=layers.Relu()
        
        self.x=None
    def forward(self,x):
        self.x=x
        for layer in self.conv.valuse():
            x=layer.forward(x)
        for layer in self.fc.valuse():
            x=layer.forward(x)
        return x
    def backward(self,y):
        return
class Inception:
    def __init__(self,in_channels,out_1x1,red_3x3,out_3x3,red_5x5,out_5x5,pit_1x1pool):
        self.x1_layers=Conv2d(in_channels,out_1x1,1)
        self.x2_layers=OrderedDict()
        self.x3_layers=OrderedDict()

        self.x2_layers['red_3x3']=Conv2d(in_channels,red_3x3,1)
        self.x2_layers['out_3x3']=Conv2d(red_3x3,out_3x3,3,1,1)
        self.x3_layers['red_5x5']=Conv2d(in_channels,red_5x5,1)
        self.x3_layers['out_5x5']=Conv2d(red_5x5,out_5x5,5,1,2)

        self.pool_3x3=layers.Pool(3,3,1,1)
        self.pit_1x1pool=Conv2d(in_channels,pit_1x1pool,1)
    def forward(self,x):
        x1=self.x1_layers.forward(x)
        x2=x.copy()
        x3=x.copy()
        for x2_layer,x3_layer in zip(self.x2_layers.values(),self.x3_layers.values()):
            x2=x2_layer.forward(x2)
            x3=x3_layer.forward(x3)
        x4=self.pool_3x3.forward(x)
        x4=self.pit_1x1pool.forward(x4)

        concat=np.concatenate((x1,x2,x3,x4))
        return concat

class GoogleNet:
    def __init__(self,num_classes,aux_logits=True):
        assert aux_logits==True or aux_logits==False
        self.aux_logits=aux_logits

        self.net=OrderedDict()
        self.net['conv_7x7']=Conv2d(3,64,7,2,3)
        self.net['maxpool_7x7']=layers.Pool(3,3,2,1)
        self.net['LRNorm']=None
        self.net['conv_3x3']=Conv2d(64,192,3,1,1)
        self.net['maxpool_3x3']=layers.Pool(3,3,2,1)
        self.net['incept_3a']=layers.Inception(192,64,96,128,16,32,32)
        self.net['incept_3b']=layers.Inception(256,128,128,192,32,96,64)
        self.net['maxpool_3']=layers.Pool(3,3,2,1)
        self.net['incept_4a']=layers.Inception(480,192,96,208,16,48,64)
        # auxiliary classifier
        self.net['incept_4b']=layers.Inception(512,160,112,224,24,64,64)
        self.net['incept_4c']=layers.Inception(512,128,128,256,24,64,64)
        self.net['incept_4d']=layers.Inception(512,112,144,288,32,64,64)
        # auxiliary classifier
        self.net['incept_4e']=layers.Inception(528,256,160,320,32,128,128)
        self.net['maxpool_4']=layers.Pool(3,3,2,1)
        self.net['inception_5a']=layers.Inception(832,256,160,320,32,128,128)
        self.net['inception_5b']=layers.Inception(832,384,192,384,48,128,128)

        self.net['avepool']=layers.AvePool(7,7)
        self.net['dropout']=layers.Dropout(0.7)
        #self.net['FC1']=layers.Affine() # using extra linear layer makes it easy to adapting & fine-tuning networks for other label
        self.net['softmax']=layers.Softmax()

        if self.aux_logits:
            self.aux1=InceptionAux(512,num_classes)
            self.aux2=InceptionAux(528,num_classes)
        else:
            self.aux1=self.aux2=None
    def forward(self,x):
        return
    def backward(self,y,optimizer):
        return

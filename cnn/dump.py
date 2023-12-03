import numpy as np
import cnnLayers_v2 as layers
import optimizer as opt
import pickle as pkl
from collections import OrderedDict
if __name__ == '__main__':
    import sys.os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.mnist import load_mnist
(x_train,t_train),(x_test,t_test)=load_mnist(normalize=True, flatten=False,one_hot_lable=True)
convInfo={'FH':5,'FW':5,'FN':30}
layers={}
params=OrderedDict()
params['W1']=np.random.randn(25,30)*0.01
params['b1']=np.zeros(30)
layers['conv']=layers.conv(params['W1'],params['b1'])
layers['relu1']=layers.relu()
layers['pool']=layers.pooling(2)
params['W2']=np.random.randn(4320,100)*0.01
params['b2']=np.zeros(100)
layers['aff1']=layers.Affine(params['W2'],params['b2'])
layers['relu2']=layers.relu()
params['W3']=np.random.randn(100,10)*0.01
params['b3']=np.zeros(10)
layers['aff2']=layers.Affine(params['W3'],params['b3'])
layers['loss']=layers.

import numpy as np
import cnnLayers_v3 as layers

a=np.arange(96).reshape(2,3,4,4)
da=np.random.rand(2,3,2,2)
ap=layers.AvePool(2,2,2,0)
print(ap.forward(a))
print(ap.backward(da))

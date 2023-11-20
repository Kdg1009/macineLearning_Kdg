import numpy as np
import cnnLayers_v2 as cnn
def meanTotal(data):
    return np.mean(data)
def meanChannel(data): # data.dtype=float64
    N,H,W,C=data.shape
    data=data.reshape(N*H,W*C)
    OH,OW,data=cnn.ready2dot(data,N,H,1,C=3,s=1,p=0)
    data=data.reshape(-1,C) #data.shape=(NWH,C)
    ret=np.mean(data,axis=0)
    ret=np.tile(ret,W) # ret.shape=(1,WC)
    return ret
def zeromean(batch,mean):
    mtypes=['total','channel']
    try:
        mtypes.index(mtype)
    except:
        print('no such mtype')
        return None
    batch-=mean
    return batch
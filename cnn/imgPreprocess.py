import numpy as np
def meanTotal(data):
    return np.mean(data)
def meanChannel(data): # data.dtype=float64
    N,H,W,C=data.shape
    data=data.transpose(0,3,1,2)
    return np.mean(data.reshape(N,C,1,H*W),axis=3)
    # to apply zero maen to each of channel, for i in C: data[0][i]-=ret[0][i]
def zeromean(batch,mean,mtype='total'):
    mtypes=['total','channel']
    try:
        mtypes.index(mtype)
    except:
        print('no such mtype')
        return None
    if mtype=='total':
        return batch-=mean
    else:
        C=batch.shape[3]
        for i in range(C):
            batch[0][i]-=mean[0][i]

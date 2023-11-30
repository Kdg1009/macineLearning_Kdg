import numpy as np
import os
import pickle as pkl
import re
rootDir='/project/python_project/carDetect/project/train/'
batchMat_train=np.zeros((20,100))
answerMat_train=np.zeros((20,100))
dirInfo=list(rootDir.split('/'))
for sub in os.listdir(rootDir):
    dirInfo.append(sub)
    subDir='/'+'/'.join(dirInfo)
    if os.isfile(subdir):
        del dirInfo[-1]
        continue
    col=int(sub.split('.')[0])
    for sub2 in os.listdir(subDir):
        fileInfo=sub2.split('.')
        row=int(fileInfo[0])
        answer=re.findall(r"[0-9]{1,3}",fileInfo[1])
        answer=list(map(int,answer))
        answerMat_train[col][row]=answer # answer.shape(1,4*carLocation)
        batchMat_train[col][row]='/'+sub+'/'+sub2
    del dirInfo[-1]
pkl.dump(answerMat_train,'/pkl/data/answer.pickle','wb')
pkl.dump(batchMat_train,'pkl/data/batch.pickle','wb')

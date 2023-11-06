import cv2
import numpy as np
import time

initA=1732584193
initB=4023233417
initC=2562383102
initD=271733878
initE=3285377520

def genblocks(img,imgHeight=427,imgWidth=640):
    blocks=[]
    for i in range(imgHeight):
        for j in range(0,imgWidth,64):
            blocks.append(img[i][j:j+64])
    return blocks
def list2dec(list):
    ret=0
    mult=1
    for i in range(4):
        ret += mult*list[3-i]
        mult*=256
    return ret
def mainLoop(a,b,c,d,e,block):
    prevVal=(a,b,c,d,e)
    # const k
    K=(1518500249,1859775393,2400959708,3395469782)
    # decompose block const W
    W=[]
    for i in range(0,64,4):
        W.insert(0,list2dec(block[i:i+4]))
    
    for i in range(80):
        # each round
        # cal w[t]
        w=lRotate(W[0]^W[2]^W[8]^W[13],1)
        W.append(w)
        del W[0]
        #cal a,b,c,d,e
        if i>=0 and i<=19:
            F=(b&c)|((~b)&d)
            constK=K[0]
        elif i>=20 and i<=39:
            F=b^c^d
            constK=K[1]
        elif i>=40 and i<=59:
            F=(b&c)|(b&d)|(c&d)
            constK=K[2]
        else:
            F=b^c^d
            constK=K[3]
        #temp=(a<<5)+F+e+constK+W[-1]
        temp=addBytes([lRotate(a,5),F,e,constK,W[-1]])
        e=d
        d=c
        c=lRotate(b,30)
        b=a
        a=temp
    return digest(prevVal,[a,b,c,d,e])

def addBytes(bList,divider=4294967296):
    f=lambda x:x[0] if len(x)==1 else x[0]+f(x[1:])
    temp=f(bList)
    temp=temp%divider
    return temp 

def lRotate(a,times,divisor=4294967296):
    mult=2**times
    a=a*mult
    head=int(a%divisor)
    tail=int(a/divisor)
    return head+tail

def digest(prevHash,presentHash):
    transpose=np.transpose([prevHash,presentHash])
    ret=list(map(addBytes,transpose))
    return ret
def hash(img):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blocks=genblocks(img)
    a,b,c,d,e=mainLoop(initA,initB,initC,initD,initE,blocks[0])
    for block in blocks[1:]:
        a,b,c,d,e=mainLoop(a,b,c,d,e,block)
    f=lambda x:hex(x)
    ret=list(map(f,[a,b,c,d,e]))
    return list2hash(ret)
    
def list2hash(hashHex):
    ret=''
    for i in hashHex:
        iLen=len(i)
        if iLen<10:
            for time in range(10-iLen):
                ret+='0'
        ret+=i[2:]
    return ret

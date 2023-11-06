import os
import cv2
import binascii
import time
import numpy as np

path='/project/python_project/carDetect/project/test_set/test_3.labled/3.[[(0,0), (0,0)]].jpg'
img=cv2.imread(path)
grayScale=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# step 1. degrade input img data
# img size: 427 * 640 * 3 => 6558720 bits
# need 12810 blocks for img data, need i block for padding (block size: 512 bits)

def genBlocks(img):
    hexImg=binascii.hexlify(img)
    blocks=[]
    hexImgSize=int(len(hexImg)/128)
    while not len(blocks)==hexImgSize:
        i=len(blocks)
        blocks.append(hexImg[i*128:(i+1)*128])
    """ tail='00000000'
    for i in range(14):
        tail += '00000000'
    tail += '000c8280'
    blocks.append(bytes(tail,'utf-8')) """
    return blocks

# step 2. main loop
# init values
initA=b'67452301'
initB=b'efcdab89'
initC=b'98badcfe'
initD=b'10325476'
initE=b'c3d2e1f0'

def mainLoop(a,b,c,d,e,block):
    previous_val=(a,b,c,d,e)
    #constance k
    k=(b'5a827999',b'6ed9eba1',b'8f1bbcdc',b'ca62c1d6')
    #decompose block into 16 bytes
    W=[]
    for i in range(16):
        W.insert(0,block[8*i:8*(i+1)])

    for i in range(80):
        #cal w[t]
        if i>15:
            #w=(decomp[i-3]^decomp[i-8]^decomp[i-14]^decomp[i-16])<<1
            w_t=lRotate(xorBytes(xorBytes(xorBytes(W[0],W[2]),W[8]),W[13]),1)
            W.append(w_t)
            del W[0]
        #cal a,b,c,d,e
        if i>=0 and i<=19:
            #F=(b&c)|((~b)&d)
            F=orBytes(andBytes(b,c),andBytes(notBytes(b),d))
            constK=k[0]
        elif i>=20 and i<=39:
            #F=b^c^d
            F=xorBytes(xorBytes(b,c),d)
            constK=k[1]
        elif i>=40 and i<=59:
            #F=(b&c)|(b&d)|(c&d)
            F=orBytes(orBytes(andBytes(b,c),andBytes(b,d)),andBytes(c,d))
            constK=k[2]
        else:
            #F=b^c^d
            F=xorBytes(xorBytes(b,c),d)
            constK=k[3]
        #temp=(a<<5)+F+e+constK+W[-1]
        #temp=addBytes(addBytes(addBytes(addBytes(F,e),lRotate(a,5)),W[-1]),constK)
        temp=addBytes([lRotate(a,5),F,e,constK,W[-1]])
        e=d
        d=c
        #c=b<<30
        c=lRotate(b,30)
        b=a
        a=temp
    #return hash
    return digestFunction(previous_val,[a,b,c,d,e])

def andBytes(abyte,bbyte):
    ret=[]
    for i in range(8):
        a=abyte[i]
        b=bbyte[i]
        if a>=97:
            a-=87
        if b>=97:
            b-=87
        temp=a&b
        if temp<16:
            temp += 87
        ret.append(temp)
    return bytes(ret)
def orBytes(abyte,bbyte):
    ret=[]
    for i in range(8):
        a=abyte[i]
        b=bbyte[i]
        if a>=97:
            a-=87
        if b>=97:
            b-=87
        temp=a|b
        if temp<16:
            temp += 87
        ret.append(temp)
    return bytes(ret)
def xorBytes(abyte,bbyte):
    ret=[]
    for i in range(8):
        a=abyte[i]
        b=bbyte[i]
        if a>=97:
            a-=87
        else:
            a-=48
        if b>=97:
            b-=87
        else:
            b-=48
        temp=a^b
        if temp>9:
            temp+=87
        else:
            temp+=48
        ret.append(temp)
    return bytes(ret)
def notBytes(byte):
    ret=[]
    for i in range(8):
        b=byte[i]
        if b>=97:
            b-=87
        else:
            b-=48
        temp=15-b
        if temp>9:
            temp+=87
        else:
            temp+=48
        ret.append(temp)
    return bytes(ret)
def lRotate(byte,times):
    ret=[]
    upper=0
    for i in range(8):
        b=byte[7-i]
        if b>97: 
           b-=87
        else:
            b-=48
        temp=b*2+upper
        upper=int(temp/16)
        temp=int(temp%16)
        if i==7:
            ret[-1]+=upper
        if temp>9:
            temp+=87
        else:
            temp+=48
        ret.insert(0,temp)
    for time in range(1,times):
        upper=0
        for i in range(8):
            b=ret[7-i]
            if b>97:
                b-=87
            else:
                b-=48
            temp=b*2+upper
            upper=int(temp/16)
            temp=int(temp%16)
            if i==7:
                ret[-1]+=upper
            if temp>9:
                temp+=87
            else:
                temp+=48
            ret[7-i]=temp
    return bytes(ret)
""" def addBytes(abyte,bbyte):
    ret=[]
    upper=0
    for i in range(0,8):
        a=abyte[7-i]
        b=bbyte[7-i]
        if a>=97:
            a-=87
        else:
            a-=48
        if b>=97:
            b-=87
        else:
            b-=48
        temp=a+b+upper
        upper=int(temp/16)
        temp=int(temp%16)
        if temp>9:
            temp+=87
        else:
            temp+=48
        ret.insert(0,temp)
    return bytes(ret) """
def addBytes(bytesArr):
    ret=[]
    upper=0
    transformedArr=np.transpose(list(map(str2byte,bytesArr)))
    addNum = lambda x:x[0] if len(x)==1 else x[0]+addNum(x[1:])
    for i in range(0,8):
        temp=addNum(transformedArr[7-i])
        temp+=upper
        upper=int(temp/16)
        temp=int(temp%16)
        ret.insert(0,temp)
    ret=byte2str(ret)
    return ret
def str2byte(str):
    ret=[]
    for i in range(0,8):
        if str[i]>=97:
            ret.append(str[i]-87)
        else:
            ret.append(str[i]-48)
    return ret
def byte2str(byte):
    for i in range(0,8):
        if byte[i]>9:
            byte[i]+=87
        else:
            byte[i]+=48
    return bytes(byte)
def digestFunction(previous_hash,present_hash):
    #a=addBytes(previous_hash[0],present_hash[0])
    #b=addBytes(previous_hash[1],present_hash[1])
    #c=addBytes(previous_hash[2],present_hash[2])
    #d=addBytes(previous_hash[3],present_hash[3])
    #e=addBytes(previous_hash[4],present_hash[4])
    a=addBytes([previous_hash[0],present_hash[0]])
    b=addBytes([previous_hash[1],present_hash[1]])
    c=addBytes([previous_hash[2],present_hash[2]])
    d=addBytes([previous_hash[3],present_hash[3]])
    e=addBytes([previous_hash[4],present_hash[4]])
    return a,b,c,d,e

def hash(img):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    start=time.time()
    blocks=genBlocks(img)
    a,b,c,d,e=mainLoop(initA,initB,initC,initD,initE,blocks[0])
    for block in blocks[1:]:
        a,b,c,d,e=mainLoop(a,b,c,d,e,block)
    hash=a+b+c+d+e
    fin=time.time()
    total=fin-start
    print(time.localtime(total))
    return hash

a=b'11111110'
b=b'1111111b'
c=b'1111111c'
print(hash(img))
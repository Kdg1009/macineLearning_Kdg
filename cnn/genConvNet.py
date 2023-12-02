from gradCheck import numerical_grad

class genNet:
    def __init__(self,A,B,C):
        self.A=A
        self.B=B
        self.C=C
    def forward(self,answer,batch,N,InfoA):
        xShape,FH,FW,C,s,p=InfoA[0]
        x=batch.reshape(xShape)
        for i in range(len(self.A)):
            xShape,FH,FW,C,s,p=InfoA[i]
            x=self.A[i].forward(x,N,FH,FW,C,s,p)
        x=self.B.forward(x,N)
        L=self.C.forward(answer,x,N)
        return L
    def backward(self,N,optA,optB,optC,InfoA,lr):
        dL=self.C.backward(N,optC,lr)
        dL=self.B.backward(dL,N,optB,lr)
        for i in range(len(self.A)):
            dL=self.A[i].backward(dL,optA[i],N,InfoA[i],lr)

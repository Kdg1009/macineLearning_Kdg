class genNet:
    def __init__(self,A,B,C,answer):
        self.A=A
        self.B=B
        self.C=C
    def forward(self,answer,batch,N,FH,FW,C=3,s=1,p=0):
        N,H,W,C=batch.shape
        x=batch.reshape(N*H,W*C)
        for layer in self.A:
            x=layer.forward(x,N,FH,FW,C,s,p)
        x=self.B.forward(x,N)
        L=self.C.forward(answer,x,N)
        return L
    def backward(self,N,optA,optB,optC,N,InfoA,lr):
        dL=self.C.backward(N,optC,lr)
        dL=self.B.backward(dL,N,B,lr)
        for i in len(self.A):
            dL=self.A[i].backward(dL,optA[i],N,InfoA[i],lr)
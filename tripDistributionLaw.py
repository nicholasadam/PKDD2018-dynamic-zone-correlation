import numpy as np

def GravExp(dij,mi,mj,beta,OD):
    dij=dij*OD.sum()/dij.sum()
    n=len(dij)
    predictedOD=np.array([[0]*n]*n).astype('float64')
    for i in range(n):
        for j in range(n):
            predictedOD[i][j]=mi[i]*mj[j]*np.exp(dij[i][j]*(-beta))
    return predictedOD

def NGravExp(dij,mi,mj,beta,OD):
    dij=dij*OD.sum()/dij.sum()
    n=len(dij)
    predictedOD=np.array([[0]*n]*n).astype('float64')
    for i in range(n):
        for j in range(n):
            predictedOD[i][j]=mi[i]*mj[j]*np.exp(dij[i][j]*(-beta))/sum(mj*np.exp(dij[i][j]*(-beta)))
    return predictedOD

def GravPow(dij,mi,mj,beta,OD):
    dij=dij*OD.sum()/dij.sum()
    n=len(dij)
    eps=1e-05
    predictedOD=np.array([[0]*n]*n).astype('float64')
    for i in range(n):
        for j in range(n):
            predictedOD[i][j]=mi[i]*mj[j]*np.power(dij[i][j]+eps,(-beta))
    return predictedOD

def NGravPow(dij,mi,mj,beta,OD):
    dij=dij*OD.sum()/dij.sum()
    n=len(dij)
    eps=1e-05
    predictedOD=np.array([[0]*n]*n).astype('float64')
    for i in range(n):
        for j in range(n):
            predictedOD[i][j]=mi[i]*mj[j]*np.power(dij[i][j]+eps,(-beta))/sum(mj*np.power(dij[i][j]+eps,(-beta)))
    return predictedOD

def Schneider(sij,mi,mj,beta,OD):
    sij=sij*OD.sum()/sij.sum()
    n=len(sij)
    predictedOD=np.array([[0]*n]*n).astype('float64')
    for i in range(n):
        for j in range(n):
            predictedOD[i][j]=np.exp(-beta*sij[i][j])-np.exp(-beta*(sij[i][j]+mj[j]))
    return predictedOD

def Rad(sij,mi,mj,beta,OD):
    sij=sij*OD.sum()/sij.sum()
    eps=1e-05
    n=len(sij)
    predictedOD=np.zeros([n,n]).astype('float64')
    proij=np.zeros([n,n])
    for i in range(n):
        for j in range(n):
            proij[i][j]=mi[i]*mj[j]/(mi[i]+sij[i][j]+eps)/(mi[i]+mj[j]+sij[i][j]+eps)
    for i in range(n):
        for j in range(n):
            predictedOD[i][j]=mi[i]*proij[i][j]/(proij.sum(axis=0)[i]+eps)
    return predictedOD

def RadExt(sij,mi,mj,beta,OD):
    sij=sij*OD.sum()/sij.sum()
    #eps=1e-04
    n=len(sij)
    predictedOD=np.zeros([n,n]).astype('float64')
    proij=np.zeros([n,n])
    for i in range(n):
        for j in range(n):
            part1=np.power((mi[i]+mj[j]+sij[i][j]),beta)
            part2=np.power((mi[i]+sij[i][j]),beta)
            part3=np.power(mi[i],beta)+1
            part4=np.power((mi[i]+sij[i][j]),beta)+1
            part5=np.power((mi[i]+mj[j]+sij[i][j]),beta)+1
            proij[i][j]=(part1-part2)*part3/part4/part5
    for i in range(n):
        for j in range(n):
            predictedOD[i][j]=mi[i]*proij[i][j]/proij.sum(axis=0)[i]
    return predictedOD

def rowNormalization(predictedOD):
    n=len(predictedOD)
    wi=np.array([0]*n).astype('int64')
    for i in range(n):
        for j in range(n):
            wi[i]+=predictedOD[i][j]
    for i in range(n):
        if wi[i]!=0:
            for j in range(n):
                predictedOD[i][j]=mi[i]*predictedOD[i][j]/wi[i]
    return predictedOD

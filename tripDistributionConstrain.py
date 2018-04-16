import numpy as np

def UM(predictedOD,Oi):
    n=len(Oi)
    nbCommuters =0
    for i in range(n):
        nbCommuters +=Oi[i]

    sumt=np.zeros(1).astype('float64')
    for i in range(n):
        for j in range(n):
            sumt+=predictedOD[i][j]
            
    sumi=np.zeros(n).astype('float64')
    for i in range(n):
        for j in range(n):
            sumi[i]+=predictedOD[i][j]
            
    nb=0
    S=np.zeros([n,n]).astype('int64')
    for i in range(n):
        for j in range(n):
            S[i][j]=np.floor(nbCommuters*predictedOD[i][j]/sumt)
            nb+=S[i][j]
    idx=Multinomial_ij(nbCommuters-nb,predictedOD,sumi)
    
    for k in idx:
        S[k[0]][k[1]]+=1
    
    return S

def PCM(predictedOD,Oi):
    n=len(Oi)
    S=np.zeros([n,n]).astype('int64')
    sumi=np.zeros(n).astype('float64')
    for i in range(n):
        for j in range(n):
            sumi[i]+=predictedOD[i][j]
            
    nb=np.zeros(n).astype('int64')
    for i in range(n):
        if sumi[i]>0:
            for j in range(n):
                S[i][j]=np.floor(Oi[i]*predictedOD[i][j]/sumi[i])
                nb[i]+=S[i][j]
    
    for i in range(n):
        if Oi[i]!=0:
            idx=Multinomial_i(Oi[i]-nb[i],predictedOD[i],sumi[i])
            for k in idx:
                S[i][k]+=1
    
    return S

def ACM(predictedOD,Dj):
    n=len(Dj)
    S=np.zeros([n,n]).astype('int64')
    tweights=S=np.zeros([n,n]).astype('float64')
    for i in range(n):
        for j in range(n):
            tweights[i][j]=predictedOD[j][i]
            
    sumi=np.zeros(n).astype('float64')
    for i in range(n):
        for j in range(n):
            sumi[i]+=tweights[i][j]
            
    nb=np.zeros(n).astype('int64')
    for i in range(n):
        if sumi[i]>0:
            for j in range(n):
                S[i][j]=np.floor(Dj[i]*tweights[i][j]/sumi[i])
                nb[i]+=S[i][j]
    
    for i in range(n):
        if Dj[i]!=0:
            idx=Multinomial_i(Dj[i]-nb[i],tweights[i],sumi[i])
            for k in idx:
                S[k][i]+=1
    
    return S

def DCM(predictedOD,Oi,Dj,maxIter, closure):
    n=len(Oi)
    marg =np.zeros([n,2]).astype('float64')
    for i in range(n):
        marg[i][0] = Oi[i]
        marg[i][1] = Dj[i]
        if marg[i][0] == 0:
            marg[i][0] = 0.01
        if marg[i][1] == 0:
            marg[i][1] = 0.01
            
    weights = np.zeros([n,n]).astype('float64')
    for i in range(n):
        for j in range(n):
            weights[i][j] = predictedOD[i][j]
            if weights[i][j] == 0:
                weights[i][j] = 0.01
    iter =0
    critOut=1
    critIn=1
    sout=np.array([0]*n).astype('float64')
    sin=np.array([0]*n).astype('float64')
    
    while ((critOut > closure or critIn > closure) and (iter <= maxIter)):
        for i in range(n):
            sout[i]=0
            for k in range(n):
                sout[i]+=weights[i][k]
                
        for i in range(n):
            for j in range(n):
                weights[i][j] = marg[i][0] * weights[i][j] / sout[i]
                
        for i in range(n):
            sin[i]=0
            for k in range(n):
                sin[i]+=weights[k][i]
                
        for i in range(n):
            for j in range(n):
                weights[i][j] =marg[j][1] * weights[i][j] / sin[j]
                
        critOut = 0
        critIn = 0
        for i in range(n):
            sout[i]=0
            sin[i]=0
            for k in range(n):
                sout[i] += weights[i][k]
                sin[i] += weights[k][i]
            critOut = max(critOut, abs(1 - (sout[i] / marg[i][0])))
            critIn = max(critIn, abs(1 - (sin[i] / marg[i][1])))
        iter+=1
        
    S=UM(weights,Oi)
    return S

def Multinomial_ij(n, weights, s):
    randomIdx=np.array([[0]*2]*n).astype('int64')
    sumt=np.zeros(1).astype('float64')
    
    for k in range(len(s)):
        sumt+=s[k]
    random=np.zeros(n).astype('float64')
    randomi=np.zeros(n).astype('float64')
    
    for k in range(n):
        random[k]=np.random.random()*sumt
        randomi[k]=random[k]
    
    for k in range(n):
        for i in range(len(s)):
            randomi[k]-=s[i]
            random[k]-=s[i]
            if randomi[k]<=0:
                random[k]+=s[i]
                randomIdx[k][0]=i
                break
        for j in range(len(weights)):
            random[k]-=weights[randomIdx[k][0]][j]
            if random[k]<=0:
                randomIdx[k][1]=j
                break

    return randomIdx

def Multinomial_i(n,weights,s):
    randomIdx=np.zeros(n).astype('int64')
    random=np.zeros(n).astype('float64')
    for k in range(n):
        random[k]=np.random.random()*s
        
    for k in range(n):
        for i in range(len(weights)):
            random[k]-=weights[i]
            if random[k]<=0:
                randomIdx[k]=i
                break
    return randomIdx

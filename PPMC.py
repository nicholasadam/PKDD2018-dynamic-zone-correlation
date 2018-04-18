import numpy as np

def PPMC(mi, mj, friction,inter):
    '''Calculates singly constrained trip distribution for a given friction factor matrix
    ProdA = Production array
    AttrA = Attraction array
    F = Friction factor matrix
    Resutrns trip table
    '''
    eps=1e-05
    n=len(mi)
    mjadj=mj
    mi=mi+eps
    mj=mj+eps
    mj=mjadj
    friction=friction+eps
    sumj = (mj*friction).sum(1)
    S=mi*(mjadj*friction).transpose()/sumj
    for i in range(inter):
        mi=np.sum(S, axis=0)+eps
        mjout=np.sum(S, axis=1)+eps
        mjadj=mjadj*mj/mjout
        sumj = (mjadj*friction).sum(1)
        S=mi*(mjadj*friction).transpose()/sumj
    return S.astype('int')

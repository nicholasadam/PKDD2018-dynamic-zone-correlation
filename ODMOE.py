import numpy as np
import matplotlib.pyplot as plt

def tripLengthFrequency(OD,distance):
    distanceRange=np.arange(0, np.floor(distance.max())+1,0.5)
    tlf=np.zeros(len(distanceRange)-1)
    n=len(OD)
    for k in range(len(distanceRange)-1):
        for i in range(n):
            for j in range(n):
                if distanceRange[k]<=distance[i][j]<distanceRange[k+1]:
                    tlf[k]=tlf[k]+OD[i][j]
    tlf=tlf/tlf.sum()
    return tlf
    
def dR(distance):
    return np.arange(0, np.floor(distance.max())+1,0.5)[1:]

def CR(predicttlf,referencetlf):
    part1=0
    part2=0
    for i in range(len(predicttlf)):
        part1=part1+min(predicttlf[i],referencetlf[i])
        part2=part2+max(predicttlf[i],referencetlf[i])
    return part1/part2

def MOEs(predict,reference,dij):
    MAE=np.mean(abs(np.array(predict)-np.array(reference)))
    NRMSE=((np.array(predict)-np.array(reference))**2).sum()/np.array(reference).sum()
    tlfd=CR(tripLengthFrequency(reference,dij),tripLengthFrequency(predict,dij))
    return MAE,NRMSE,tlfd

import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

def processdata(datafile,dimension):
    
    data=np.loadtxt(datafile)
    serie= pd.Series(data)
    laggeddata = pd.concat( [serie.shift(i) for i in range(dimension+1)],axis=1 )
    trainset = laggeddata.iloc[dimension:int(np.floor(0.8*len(laggeddata))),1:dimension+1]
    traintarget = laggeddata.iloc[dimension:int(np.floor(0.8*len(laggeddata))),0]
    
    testset = laggeddata.iloc[int(np.floor(0.8*len(laggeddata))):len(laggeddata),1:dimension+1]
    testtarget = laggeddata.iloc[int(np.floor(0.8*len(laggeddata))):len(laggeddata),0]
    
    return (trainset,traintarget,testset,testtarget)

def createmodel(trainset,traintarget,dimension):
    
    trainset[dimension+1]=1 
    coefs = np.linalg.pinv(trainset).dot(traintarget)
    print coefs
    return coefs

def predict(coefs,testset,testtarget,dimension):
    testset[dimension+1]=1
    preds = testset.dot(coefs)
    
    erro = metrics.mean_squared_error(preds,testtarget)
    x= range(len(testtarget))    
    plt.plot(x,testtarget,'r--',label='real')
    plt.plot(x,preds,label='predicted')
    plt.legend()
    return (erro,preds)

(trainset,traintarget,testset,testtarget) = processdata('eletric.txt',12)
coefs = createmodel(trainset,traintarget,12)
(erro,preds) = predict(coefs,testset,testtarget,12)  









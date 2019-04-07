import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.metrics as metrics

def processdata(datafile,dimension,nPassos):
    
    data = np.loadtxt(datafile)
    dataNormalizado = normalizacao(data)
    serie = pd.Series(dataNormalizado)
    lagged = pd.concat([serie.shift(i) for i in range(dimension+1) ],axis=1) # dimension +1 : previsão de um passo à frente
     
    trainIndex=int(np.floor(0.6*len(data)))  
    
    trainSet = lagged.iloc[dimension:trainIndex,1:dimension+1]
    trainTarget = lagged.iloc[dimension:trainIndex,0]

    testSet = lagged.iloc[trainIndex:len(data),1:dimension+1]
    testTarget = lagged.iloc[(trainIndex + nPassos):len(data),0]
    
    return (trainSet,trainTarget,testSet,testTarget,min(data),max(data))

def predict(testSet,testTarget,coefs,valorMinimo,valorMaximo,nPassos,dimension):
    predictsIterativo = np.zeros(shape = (nPassos,1))
    testSet[dimension+1]=1
    predicts = testSet.dot(coefs)
    
    for i in range(len(testSet)):
        testSetNovo = testSet.iloc[i:i+1,0:dimension+1]
        firstPredict = model.predict(testSetNovo)
        predictsIterativo[0] = float(firstPredict[-1])
        
        testSetIterativo = [testSetNovo.iloc[-1,(-j-2)] for j in range(dimension-1)]
        testSetIterativo.append(firstPredict[-1])
        testSetIterativo.reverse()
        testSetIterativoDataframe = pd.DataFrame(testSetIterativo)
        testSetIterativoDataframe = testSetIterativoDataframe.transpose()
        
        for k in range(nPassos - 1):
            secondPredict = model.predict(testSetIterativoDataframe)
            predictsIterativo[k+1] = float(secondPredict)
            testSetIterativo = [testSetIterativoDataframe.iloc[-1,(-j-2)] for j in range(dimension-1)]
            testSetIterativo.append(float(predicts[k+1]))
            testSetIterativo.reverse()
            testSetIterativoDataframe = pd.DataFrame(testSetIterativo)
            testSetIterativoDataframe = testSetIterativoDataframe.transpose()   
        
        predicts[i] = predictsIterativo[-1]
        
    testTargetDesnormalizado = desnormalizacao(testTarget,valorMinimo,valorMaximo)
    predictsDesnormalizado = desnormalizacao(predicts,valorMinimo,valorMaximo)
    predictsIterativo = desnormalizacao(predictsIterativo,valorMinimo,valorMaximo)
    predictsComResultado = predictsDesnormalizado[0:len(testTarget)]
    
    erro = metrics.mean_squared_error(testTargetDesnormalizado,predictsComResultado)
    x=range(len(testTarget))
    x2 = [i + len(testTarget) - 1 for i in range(6)]
    predictsPlot = [i for i in predictsIterativo]
    predictsPlot.insert(0,predictsComResultado[-1])
    plt.plot(x,predictsComResultado,'b--',label='predicts')
    plt.plot(x,testTargetDesnormalizado,'r',label='real')
    plt.plot(x2,predictsPlot,'orange',label='previsao n passos')
    plt.legend()
    plt.show()
    print "O Erro eh: {}\nAs Proximas 5 previsões são:\n{}".format(erro,predictsIterativo)
    return (erro,predictsDesnormalizado,testTargetDesnormalizado)

def coeficientes(trainset,traintarget,dimension):
    
    trainset[dimension+1]=1 
    coefs = np.linalg.pinv(trainset).dot(traintarget)
    print coefs
    return coefs

def normalizacao(dados):
    dadosNormalizados = dados.copy()
    index = 0
    for i in dados:
        dadosNormalizados[index] = (i - min(dados))/(max(dados) - min(dados))
        index += 1
        
    return dadosNormalizados    
    
def desnormalizacao(dados,valorMinimo,valorMaximo):
    dadosDesnormalizados = dados.copy()
    if type(dados) == np.ndarray:
        index = 0
    elif type(type(dados) == pd.core.series.Series):
        index = dados.first_valid_index()
        
    for i in dados:    
        dadosDesnormalizados[index] = i*(valorMaximo - valorMinimo) + valorMinimo
        index += 1
        
    return dadosDesnormalizados

(trainSet,trainTarget,testSet,testTarget,valorMinimo,valorMaximo) = processdata('airlines2.txt',12,5)

coefs = coeficientes(trainSet,trainTarget,12)

(erro,predicts,testTarget) = predict(testSet,testTarget,coefs,valorMinimo,valorMaximo,5,12)


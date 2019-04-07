import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.metrics as metrics
import blindSearch as bs

def processdata(datafile,dimension,nPassos,previsaoNPassos):
    
    nPassos = nPassos - 1
    data = np.loadtxt(datafile)
    datanormalizado = normalizacao(data)
    serie = pd.Series(datanormalizado)
    lagged = pd.concat([serie.shift(i) for i in range(dimension+1) ],axis=1) # dimension +1 : previsão de um passo à frente
     
    trainindex=int(np.floor(0.6*len(data)))
    valindex=int(np.floor(0.8*len(data)))
    
    
    trainset = lagged.iloc[dimension:trainindex,1:dimension+1]
    traintarget = lagged.iloc[(dimension + nPassos):(trainindex + nPassos),0]
    
    valset = lagged.iloc[trainindex:valindex,1:dimension+1]
    valtarget = lagged.iloc[(trainindex + nPassos):(valindex + nPassos),0]
    
    testset = lagged.iloc[valindex:(len(data) - nPassos),1:dimension+1]
    testtarget = lagged.iloc[(valindex + nPassos):len(data),0]

    return (trainset,traintarget,valset,valtarget,testset,testtarget,min(data),max(data))

def predict(testset,testtarget,model,valorminimo,valormaximo,nPassos):
     
    predicts = model.predict(testset)
    
    testtargetdesnormalizado = desnormalizacao(testtarget,valorminimo,valormaximo)
    predictsdesnormalizado = desnormalizacao(predicts,valorminimo,valormaximo)
    
    erro = metrics.mean_squared_error(testtargetdesnormalizado,predictsdesnormalizado)
    x=range(len(testtarget))
    plt.plot(x,predictsdesnormalizado,'b--',label='predicts')
    plt.plot(x,testtargetdesnormalizado,'r',label='real')
    plt.legend()
    plt.show()
    return (erro,predictsdesnormalizado,testtargetdesnormalizado)

def createmodel(nPassos,previsaoNPassos,randomState):
    (trainset,traintarget,valset,valtarget,testset,testtarget,valorminimo,valormaximo) = processdata('airlines2.txt',12,nPassos,previsaoNPassos)
    (bestModel,bestPredicts,bestError,bestParam) = bs.gridSVR(trainset,traintarget,valset,valtarget)
    (erro,predicts,testtarget) = predict(testset,testtarget,bestModel,valorminimo,valormaximo,nPassos)
    return erro,predicts,testtarget

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

nPassos = 5
previsaoNPassos = [None for i in range(nPassos)]
randomState = int(np.random.random()*10000)

for i in range(5):
    (erro,predicts,testtarget) = createmodel(i+1,previsaoNPassos,randomState)
    previsaoNPassos[i] = predicts[-1]
    print "Previsao de {} Passos: {}\nErro {}: {}\n".format(i+1,predicts[-1],i+1,erro)
    
x2 = [i + len(testtarget) - 1 for i in range(6)]
previsaoNPassos.insert(0,predicts[-1])
x=range(len(testtarget))
plt.plot(x,predicts,'b--',label='predicts')
plt.plot(x,testtarget,'r',label='real')
plt.plot(x2,previsaoNPassos,'orange',label='previsao n passos')
plt.legend()
plt.show()

print "Proximas 5 previsoes: {}".format(previsaoNPassos)
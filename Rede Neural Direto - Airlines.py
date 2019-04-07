from sklearn.neural_network import MLPRegressor
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.metrics as metrics

def processdata(datafile,dimension,nPassos,previsaoNPassos):
    
    nPassos = nPassos - 1
    data = np.loadtxt(datafile)
    datanormalizado = normalizacao(data)
    serie = pd.Series(datanormalizado)
    lagged = pd.concat([serie.shift(i) for i in range(dimension+1) ],axis=1) # dimension +1 : previsão de um passo à frente
     
    trainindex=int(np.floor(0.6*len(data)))    
    
    trainset = lagged.iloc[dimension:trainindex,1:dimension+1]
    traintarget = lagged.iloc[(dimension + nPassos):(trainindex + nPassos),0]
    
    testset = lagged.iloc[trainindex:(len(data) - nPassos),1:dimension+1]
    testtarget = lagged.iloc[(trainindex + nPassos):len(data),0]

    return (trainset,traintarget,testset,testtarget,min(data),max(data))

def predict(testset,testtarget,model,valorminimo,valormaximo,nPassos):
    
    testtargetdesnormalizado = testtarget.copy()    
    predicts = model.predict(testset)
    predictsdesnormalizado = predicts.copy()
    
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
    (trainset,traintarget,testset,testtarget,valorminimo,valormaximo) = processdata('airlines2.txt',12,nPassos,previsaoNPassos)

    model = MLPRegressor(hidden_layer_sizes=(12), activation='relu', solver='lbfgs', learning_rate='adaptive', 
                      learning_rate_init=0.01, alpha=0.01,random_state = randomState)       #Solver='adam' é o padrao mas serve para muitos dadosm lbfgs é melhor para poucos dados
    model.fit(trainset,traintarget)

    (erro,predicts,testtarget) = predict(testset,testtarget,model,valorminimo,valormaximo,nPassos)
    
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
    print "Previsao de {} Passo: {}\nErro {}: {}\n".format(i+1,predicts[-1],i+1,erro)
    
x2 = [i + len(testtarget) - 1 for i in range(6)]
previsaoNPassos.insert(0,predicts[-1])
x=range(len(testtarget))
plt.plot(x,predicts,'b--',label='predicts')
plt.plot(x,testtarget,'r',label='real')
plt.plot(x2,previsaoNPassos,'orange',label='previsao n passos')
plt.legend()
plt.show()

print "Erro 5: {}\nProximas 5 previsoes: {}".format(erro,previsaoNPassos)
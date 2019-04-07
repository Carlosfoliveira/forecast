# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 18:56:47 2018

@author: faust
"""

from sklearn import metrics
from sklearn import svm


def gridSVR(X_train,Y_train,X_val,Y_val):

    C_r = range(-5,5)
    Epsilon_r = range(-5,5)
    Gamma_r = range(-5,5)
    
    bestModel=1
    bestError=9999999999999999
    bestPredicts=1
    bestParam=(0,0,0)
    for c in C_r:
        for e in Epsilon_r:
            for g in Gamma_r:
                model = svm.SVR(C=10**c,gamma=10**g,epsilon=10**e)
                model.fit(X_train,Y_train)
                predicts = model.predict(X_val)
                erro = metrics.mean_squared_error(Y_val,predicts)
                if (erro<bestError):
                    bestError=erro
                    bestModel=model
                    bestPredicts=predicts
                    bestParam=(c,g,e)
                    print bestError
                    print bestParam
    return (bestModel,bestPredicts,bestError,bestParam)
                
    

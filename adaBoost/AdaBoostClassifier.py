#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 23:47:58 2020

@author: poorvaja
"""


import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn import metrics

class DecisionTree():
    def __init__(var1):
        var1.polarity = 1
        var1.alpha = 0
        var1.findex = 0
        var1.threshold = 0

class AdaBoostClassif():
    def __init__(var1, numclassif=4):
        var1.numclassif = numclassif
        
    def predict(var1, x):
        sample = np.shape(x)[0]
        
        y_pred = np.zeros((sample, 1))

        for j in var1.classifs:
            pred = np.ones(np.shape(y_pred))
            negindex = (j.polarity * x[:, j.findex] < j.polarity * j.threshold)
            
            pred[negindex] = -1
            y_pred = y_pred + (j.output * pred)
            
        y_pred = np.sign(y_pred).flatten()
        
        return y_pred
    
    def fit(var1, x, y):
        sample, feature = np.shape(x)
        
        weight = np.full(sample, (1/sample))
        
        var1.classifs = []
        
        for _ in range(var1.numclassif):
            classifier = DecisionTree()
            
            minimumerr = float('inf')
            
            for k in range(feature):
                featureval = np.expand_dims(x[:, k], axis =1)
                uniqueval = np.unique(featureval)
                
                for i in uniqueval:
                    p = 1
                    pred = np.ones(np.shape(y))
                    pred[x[:, k] < i] = -1
                    err = sum(weight[y != pred]) 
                    
                    if err > 0.5:
                        err = 1-err
                        p = -1
                        
                    if err < minimumerr:
                        classifier.polarity = p
                        classifier.threshold = i
                        classifier.findex = k
                        minimumerr = err
                        
            classifier.output = 0.5 * math.log((1.0 - minimumerr) / (minimumerr + 1e-10))
            
            pred = np.ones(np.shape(y))
            
            negindex = (classifier.polarity +x[:, classifier.findex] < classifier.polarity * classifier.threshold)
            
            pred[negindex] = -1
            
            weight *= np.exp(-classifier.output * y * pred)
            
            weight /= np.sum(weight)
            
            var1.classifs.append(classifier)
    

def main():
    phishd = pd.read_csv("dataset.csv") #Please set the path to the dataset directory
    
    x=phishd.iloc[:,1:31].values
    y=phishd.iloc[:,31].values

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    
    classifier = AdaBoostClassif(numclassif = 4)
    
    classifier.fit(x_train, y_train)
    
    y_pred = classifier.predict(x_test)
    
    print('Confusion Matrix:')
    print(metrics.confusion_matrix(y_test, y_pred))
    
    print(metrics.classification_report(y_test, y_pred))
    
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print ("Accuracy:", accuracy)

if __name__ == "__main__":
    main()
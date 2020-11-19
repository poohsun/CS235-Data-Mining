"""
bayes.py

by Josiah R. Anderson

An implementation of a Bernoulli Naive Bayesian classifier for website phishing data

University of California, Riverside
Winter 2020
CS 235 Data Mining Techniques

Bayes' Theorem:
P(class∣data) = (P(data∣class) * P(class)) / P(data)

Step-by-Step Bernoulli Naive Bayesian classifier:
https://iq.opengenus.org/bernoulli-naive-bayes/

Data set:
https://www.kaggle.com/akashkr/phishing-website-dataset

"""

import pandas
from sklearn import metrics


CSV_PATH_TO_DATA = '/home/darkstar/Downloads/UCR/CS_235/project/project_phishing_dataset_cleaned.csv'
TRAINING_SET_PERCENTAGE = 0.80 # We will use 80% of our data for training, and the rest for testing
PHISHING = -1
LEGITIMATE = 1
LAPLACE_SMOOTHING = 0.0000000000001


def loadDataSet(csvPath):
    
    dataSet = pandas.read_csv(csvPath)
            
    return dataSet



def partitionDataIntoSets(dataSet):
    
    trainingSet = dataSet.sample(frac = TRAINING_SET_PERCENTAGE)
    testSet = dataSet.drop(trainingSet.index)
    
    return trainingSet, testSet



def groupDataByClasses(data):
    
    phishingSites = data[data['result'] == PHISHING]
    legitimateSites = data[data['result'] == LEGITIMATE]
    
    return phishingSites, legitimateSites



def calculateClassProbabilities(trainingSet):
    
    phishingSites, legitimateSites = groupDataByClasses(trainingSet)
        
    totalSites = len(phishingSites) + len(legitimateSites)
    
    phishingClassProbability = len(phishingSites) / totalSites
    legitimateClassProbability = len(legitimateSites) / totalSites
    
    return phishingClassProbability, legitimateClassProbability



def calculateFeatureProbabilitiesByClass(data):
    
    phishingSites, legitimateSites = groupDataByClasses(data)
        
    numberOfPhishingSites = phishingSites['result'].count()
    numberOfLegitimateSites = legitimateSites['result'].count()
    phishingFeaturesProbability = []
    legitimateFeaturesProbability = []
    
    # These are irrelevant to feature probabilities
    phishingSites.drop(columns=['site'])
    phishingSites.drop(columns=['result'])
    legitimateSites.drop(columns=['site'])
    legitimateSites.drop(columns=['result'])
    
    for feature in list(phishingSites):
    
        phishingFeature = phishingSites.apply(lambda phishing: True if phishing[feature] == PHISHING else False, axis=1)
        numberOfRows = len(phishingFeature[phishingFeature == True].index)
        
        featureProbability = numberOfRows / numberOfPhishingSites
        
        if featureProbability == 0.0:
            featureProbability = LAPLACE_SMOOTHING
        
        phishingFeaturesProbability.append(featureProbability)
        
    for feature in list(legitimateSites):
    
        legitimateFeature = legitimateSites.apply(lambda legitimate: True if legitimate[feature] == LEGITIMATE else False, axis=1)
        numberOfRows = len(legitimateFeature[legitimateFeature == True].index)
        
        featureProbability = numberOfRows / numberOfLegitimateSites
        
        if featureProbability == 0.0:
            featureProbability = LAPLACE_SMOOTHING
        
        legitimateFeaturesProbability.append(featureProbability)
    
    return phishingFeaturesProbability, legitimateFeaturesProbability



def naiveBayesClassifier(trainingSet, testSet):

    phishingClassPriorProbability, legitimateClassPriorProbability = calculateClassProbabilities(trainingSet)
    phishingFeaturesProbability, legitimateFeaturesProbability = calculateFeatureProbabilitiesByClass(trainingSet)
    
    dataFeatureList = list(testSet)
    phishingProbabilities = []
    legitimateProbabilities = []
    classificationLabels = []
    
    # The below algorithm runs in O(n*features) due to the double for loops. This should
    # be avoided to keep runtime complexity low. But it will suffice for now.
    
    for index, row in testSet.iterrows():
        
        phishingProbability = 0.0
        legitimateProbability = 0.0
        featureIndex = 0
        
        for feature in dataFeatureList:
            
            # Because of the way we filtered the data, if feature == PHISHING then we
            # must reverse it by (1.0 - feature probability) to get the corresponding
            # probability for "legitimate". Conversely, if feature is LEGITIMATE, we
            # must do (1.0 - feature probability) to get the "phishing" probability.
            # Yeah, it's ugly. I know.
            
            if row[feature] == PHISHING:
            
                if phishingProbability == 0:
                    phishingProbability = phishingFeaturesProbability[featureIndex]
                else:
                    phishingProbability = phishingProbability * phishingFeaturesProbability[featureIndex]
                    
                if legitimateProbability == 0:
                    legitimateProbability = (1.0 - legitimateFeaturesProbability[featureIndex])
                else:
                    legitimateProbability = legitimateProbability * (1.0 - legitimateFeaturesProbability[featureIndex])
            
            elif row[feature] == LEGITIMATE:
                
                if phishingProbability == 0:
                    phishingProbability = (1.0 - phishingFeaturesProbability[featureIndex])
                else:
                    phishingProbability = phishingProbability * (1.0 - phishingFeaturesProbability[featureIndex])
                    
                if legitimateProbability == 0:
                    legitimateProbability = legitimateFeaturesProbability[featureIndex]
                else:
                    legitimateProbability = legitimateProbability * legitimateFeaturesProbability[featureIndex]
                    
                                              
            featureIndex = featureIndex + 1
        
        # Here we check for 0 values, because multiplying by zero is always zero.
        # We replace with a  Laplacian smoothing factor very close to zero to
        # to avoid this problem.
        
        if phishingProbability != 0.0:
            phishingProbabilities.append(phishingClassPriorProbability * phishingProbability)
        else:
            phishingProbabilities.append(phishingClassPriorProbability * LAPLACE_SMOOTHING)
        
        if legitimateProbability != 0.0:
            legitimateProbabilities.append(legitimateClassPriorProbability * legitimateProbability)
        else:
            legitimateProbabilities.append(legitimateClassPriorProbability * LAPLACE_SMOOTHING)
    
            
    for index in range(len(phishingProbabilities)):
        
        phishingProbability = phishingProbabilities[index]
        legitimateProbability = legitimateProbabilities[index]
        
        if phishingProbability > legitimateProbability:
            classificationLabels.append(PHISHING)
            
        elif phishingProbability < legitimateProbability:
            classificationLabels.append(LEGITIMATE)
    
    return classificationLabels


 
def evaluateClassifier(testData, classificationLabels):
    
    actualLabels = testData['result']
    
    print("Confusion Matrix: ")
    print()
    print(metrics.confusion_matrix(actualLabels, classificationLabels))
    print()
    print("Report: ")
    print(metrics.classification_report(actualLabels, classificationLabels))
    print()
    accuracy = metrics.accuracy_score(actualLabels, classificationLabels)
    print("Classification Accuracy:", accuracy)



def run():
    
    phishingData = loadDataSet(CSV_PATH_TO_DATA)
    
    trainingSet, testSet = partitionDataIntoSets(phishingData)
    
    classificationLabels = naiveBayesClassifier(trainingSet, testSet)
    
    evaluateClassifier(testSet, classificationLabels)        
    
run()
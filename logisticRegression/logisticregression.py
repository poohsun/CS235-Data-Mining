'''
logisticregression.py
Implemented by Nathan Gootee
For CS235 Data Mining Techniques at UC Riverside

A logistic regression algorithm that is fit with gradient descent.
Designed for project in phishing website classification.
'''

#Imports
import numpy as np

#logit/sigmoid activation function
def sigmoid(X):
    return 1.0/(1.0+np.exp(-X))

#Returns dot product of inputs and weights
def weight_function(theta, X):
    return np.dot(X, theta.T)

#Scales inputs to their weights and passes through sigmoid activation, returning probability
def probability(theta, X):
    return sigmoid(weight_function(theta,X))

#cost function (Inverse of likelihood function)
def cost_function(theta,X,Y):
    hypothesis = probability(theta, X)
    #Removes dimensionality to set as (predictions,) and (labels,) for cost computation
    hypothesis=np.squeeze(hypothesis)
    Y=np.squeeze(Y)
    #cost computation
    cost = -Y*np.log(hypothesis)-(1-Y)*np.log(1-hypothesis)
    #Return mean cost of each point
    return np.mean(cost)

#gradient of cost/likelihood function for gradient descent
def gradient(theta, X, Y):
    return np.dot((probability(theta, X) - Y).T, X)

#gradient descent used to minimize cost function through updating input weights
def gradient_descent(theta, X, Y, learn_rate=0.1, convergance=0.001):
    #initialize cost
    curr_cost=cost_function(theta, X, Y)
    change_cost=1
    #For testing, records number of steps in gradient descent until convergance
    i=0
    #Continue to update theta and cost until cost reaches convergance
    while(change_cost > convergance):
        prev_cost=curr_cost
        #update theta and cost
        theta=theta-(learn_rate*gradient(theta, X, Y))
        curr_cost=cost_function(theta, X, Y)
        #update change in cost to check for convergance
        change_cost=prev_cost-curr_cost
        i+=1
    return theta, i

#Uses labelled input data to fit optimal input weights (theta)
#X input as array (n,f) and Y input as dimensionless labels (n,)
# Where n is number of observations and f is features
def fit(X, Y, learn_rate=0.01, convergance=0.01):
    #Initialize Inputs
    #Add first column of input variable matrix as 1s for first (constant) weight (theta_0)
    X=np.hstack((np.ones((X.shape[0],1)),X))
    #Transform y to 1 column numpy array
    Y=Y.to_numpy()
    Y=Y.reshape(Y.shape[0],1)
    #Initialize weights as zero
    theta=np.zeros((1,X.shape[1]))

    #Fit logistic algorithm with gradient descent
    theta, num_steps = gradient_descent(theta, X, Y, learn_rate, convergance)
    return theta, num_steps

#The predicted values for input set X
def predict(theta, X):
    #Add first column of input variable matrix as 1s for first (constant) weight (theta_0)
    X=np.hstack((np.ones((X.shape[0],1)),X))
    #Compute probabilities
    hypothesis = probability(theta, X)
    #Cast probabilities to categories
    predictions = np.where(hypothesis>=0.5, 1, 0)
    #return as (n,) for comparison with labels stored in same format
    predictions=np.squeeze(predictions)
    return predictions
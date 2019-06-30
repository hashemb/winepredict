# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 19:31:33 2018

@author: Hashem
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def Accuracy(data,th,alpha, its):
    count = 0
    
    for i in range(len(data)):
        
     a = np.array(data.iloc[i])
     
     qual = a[a.shape[0] - 1]
     
     a = np.delete(a, -1)
     
     a = a.reshape((1,12))
     
     a = a.astype(np.float)
     
     p = predictQuality(th,a)
     
     
     if str(p) == qual:  
         count = count + 1
    acc = (count*100)/1600
    
    print("Accuracy when Alpha = " + str(alpha) + " and Iterations = " + str(its) + " is " + str(acc) + "%")
    return acc


def Divide(dset,trainPar):
        """
        takes a paramater trainPar: that determines the ratio of the training set of 100%, takes paramater
                                   imgs which is the whole set
                          dset : the dataset to be divided
        """
        trainElms = int((trainPar * len(dset)) / 100)
        
        if trainPar == 100:
            trainRes = dset.iloc[1:trainElms,:]

            return (trainRes)
        
        
        
        trainRes = dset.iloc[1:trainElms,:]
        testRes = dset.iloc[trainElms: len(dset) - 1,:]
        


        
        result = {'tr':trainRes , 'te': testRes}

        return result

def predictQuality(thetas,vals):
    """
    takes thetas: the paramaters calculated with gradient descent
          vals: the attributes values to predict the quality for it
          returns r: the quality 
    
    """
    h = np.dot(vals,thetas.T)
    r = h[0]
    r = float(r)
    r = round(r)
    return r
    

def readData(fileName):
    """
    takes fileName: name of the file from which the data set will be taken
    returns data: processed, separated and has extra column of ones, for the multiplication later
    
    """
    data = pd.read_csv(fileName,sep=';',header= None)
    data.insert(0, 'Ones', 1)
    return data

def computeCost(x,y,theta): # previously explained
    inner = np.power(((x * theta.T) - y),2)
    result = np.sum(inner) * ( 1/2*len(x))
    return result
def plotCostVSIterations(cost,its):
    """
    takes cost: np.array(number of itrerations): the cost value changing with each iteration
          its: # of iterations
          plots the two of them to see how does the cost have been changed with number 
          of iterations increasing
    """
    fig, ax = plt.subplots(figsize=(12,8))  
    ax.plot(np.arange(its), cost, 'r')  
    ax.set_xlabel('Iterations')  
    ax.set_ylabel('Cost') 
    
def gradientDescent(x,y,theta,alpha,its): # previously explained
    temp = np.array(np.zeros(theta.shape))
    cost = np.zeros(its)
    for i in range(its):
      err = (x * theta.T) - y
      for j in range(12):
         deriv = np.multiply(err,x[:,j])
         temp[0,j] = theta[0,j] - ((alpha / len(x))*np.sum(deriv))
         
      theta = temp
      cost[i] = computeCost(x,y,theta)
      print("J = " + str(cost[i]))

    return theta, cost
  
data = readData('white_wine.csv')

d = Divide(data,70)
d = d['tr']  #you can take the testing set with 'te'
cols = data.shape[1]
x = d.iloc[1:,0:cols-1]
y = d.iloc[1:,cols-1:cols]

x = np.matrix(x.values)
y = np.matrix(y.values)

x = x.astype(np.float) #converting dtype to float
y = y.astype(np.float)

theta = np.matrix(np.array([0,0,0,0,0,0,0,0,0,0,0,0])) #12 zeros for 11 features

alpha = 0.00001
its = 15000

th, cost = gradientDescent(x,y,theta,alpha,its)

specs = np.array([[7,0.27,0.36,20.7,0.045,45,170,1.001,3,0.45,8.8,6]]) # values for trying the model

#predictQuality(th,specs) 

Accuracy(d,th,alpha,its)
plotCostVSIterations(cost,its)

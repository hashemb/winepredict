# -*- coding: utf-8 -*-
"""

@author: Hashem
"""
import pandas as pd
import scipy.optimize as opt  
import numpy as np
import matplotlib.pyplot as plt

def mix(file1,file2):
  data1 = pd.read_csv(file1,sep=';',header= None)
  data1.insert(12,'type',0)
  data1 = data1.drop(data1.index[0:1])

  data2 = pd.read_csv(file2,sep=';',header= None)
  data2.insert(12,'type',1)
  data2 = data2.drop(data2.index[0:1])



  data = pd.concat([data1, data2], ignore_index=True)
  data.insert(0, 'Ones', 1)
  data = data.sample(frac=1)
  return data

def AccuracyWithLogisticGradient(data,th,alpha, its):
    count = 0
    for i in range(len(data)):        
     a = np.array(data.iloc[i])     
     typ = a[a.shape[0] - 1]     
     a = np.delete(a, -1)     
     a = a.reshape((1,13))     
     a = a.astype(np.float)     
     p = predictType(th,a)
     if p == typ:  
         count = count + 1
    acc = (count*100)/len(data)
    print("Accuracy when Alpha = " + str(alpha) + " and Iterations = " + str(its) + " is " + str(acc) + "%")
    return acc

def AccuracyWithOptFun(data,th):
    
    count = 0
    for i in range(len(data)):        
     a = np.array(data.iloc[i])     
     typ = a[a.shape[0] - 1]     
     a = np.delete(a, -1)     
     a = a.reshape((1,13))     
     a = a.astype(np.float)     
     p = predictType(th,a)
     if p == typ:  
         count = count + 1
    acc = (count*100)/len(data)
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
    print(x.shape)
    print((theta.T).shape)
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
      #print("J = " + str(cost[i]))

    return theta, cost


def gradientDescentLogistic(x,y,theta,alpha,its): #instead of using optimization.fmin_tnc
    temp = np.array(np.zeros(theta.shape))
    
    for i in range(its):
      error = sigmoid(x * theta.T) - y
      for j in range(13):
         term = np.multiply(error, x[:,j])
         temp[0,j] = np.sum(term) / len(x)
      theta = temp
      
    
    return theta



def LogCost(theta,x, y):
#    print(x.shape)
#    print(y.shape)
#    print(theta.shape)
    theta = np.reshape(theta,(1,13))
    #print(theta.T.shape)
    first = np.multiply(-y, np.log(sigmoid(x * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(x * theta.T)))
    return np.sum(first - second) / (len(x))

def sigmoid(x):
  #print(1 / (1 + np.exp(-x)))   Sigmoid tracking
  return 1 / (1 + np.exp(-x))


def GradientStep(theta, x, y):  
    theta = np.matrix(theta)
    x = np.matrix(x)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)

    error = sigmoid(x * theta.T) - y

    for i in range(parameters):
        term = np.multiply(error, x[:,i])
        grad[i] = np.sum(term) / len(x)

    return grad

def predictType(theta, x):  
    probability = sigmoid(np.dot(x,theta.T))
    if probability >= 0.5:
        return 1
    elif probability < 0.5:
        return 0


data = mix("red_wine.csv","white_wine.csv")
darra = np.array(data)
theta = np.matrix(np.array([0,0,0,0,0,0,0,0,0,0,0,0,0])) #13 zeros for 12 features
alpha = 0.000001
its = 15000

cols = data.shape[1]
x = data.iloc[1:,0:cols-1]
y = data.iloc[1:,cols-1:cols]

x = np.matrix(x.values)
y = np.matrix(y.values)

x = x.astype(np.float) #converting dtype to float
y = y.astype(np.float)

#test = np.array(['1','7.7','0.49','0.26','1.9','0.062','9','31','0.9966','3.39','0.64','9.6'
#,'5'])
#test1 = np.array(['1','8.6','0.37','0.65','6.4','0.08','3','8','0.99817','3.27','0.58','11'
#,'5'])
#test = test.astype(np.float)

#result = opt.fmin_tnc(func=LogCost, x0=theta, fprime=None,approx_grad=True, args=(x, y))  
h = gradientDescentLogistic(x,y,theta,alpha,its)
print(AccuracyWithLogisticGradient(data,h,alpha,its))
#print(AccuracyWithOptFun(data,h))


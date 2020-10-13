# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 14:13:54 2020

@author: MAYANK
"""

import numpy as np
import matplotlib.pyplot as plt
def init_param(prev_shape, n_h):
    
    W = np.random.randn(n_h, prev_shape)*0.01
    b = np.zeros(shape = (n_h, 1))
    return W, b

def sigmoid(z):
    return 1/(1+np.exp(-z))

def relu(z):
    return np.maximum(z, 0)

def relu_backward(z):
    z[z>0] = 1
    z[z<=0] = 0
    return z

def sigmoid_backward(z):
    return np.multiply(sigmoid(z), 1-sigmoid(z))


def forward_propagation(W, b, X):
   
    Z = np.dot(W, X) + b
    return Z


def backward_propagation( dal, grads, m, params):
    grads_desc = {}
   
    dz = np.multiply(dal, sigmoid_backward(grads['Z'+str(3)]))
    
    for i in range(3, 0, -1):
        if i<3:
            dz = np.multiply(dal, relu_backward(grads['Z'+str(i)]))
        
        
        dw = 1/m * (np.dot(dz, grads['A'+str(i-1)].T))
        db = 1/m * (np.sum(dz, axis = 1, keepdims = True))
       
        dal = np.dot(params['W'+str(i)].T, dz)
        
        grads_desc['dw'+str(i)] = dw
        grads_desc['db'+str(i)] = db
        
        
   
    return grads_desc
            




X = np.load('input_data.npy', allow_pickle = True)
X = X/255
Y = np.load('output_data.npy', allow_pickle = True)
Y = Y/255
network = [X.shape[0], 2, 3,1]
m = X.shape[1]

params = {}
for i in range(1, len(network)):
    W, b = init_param(network[i-1], network[i])
    params['W'+str(i)] = W
    params['b'+str(i)] = b


    
    
  
cost = []
iteration = []

for j in range(10000):
    #Forwrad propagation
    A = X
    grads = {}
    grads['A0'] = X
    
    for i in range(1, len(network)-1):
        prev_A = A
       
        Z = forward_propagation(params['W'+str(i)], params['b'+str(i)], prev_A)
        grads['Z'+str(i)] = Z
        A = relu(Z)
    
        grads['A'+str(i)] = A
    
    
    
    Z = forward_propagation(params['W'+str(len(network)-1)], params['b'+str(len(network)-1)], A)
    A = sigmoid(Z)
    
    cost.append((-1/m) * np.sum(np.multiply(Y, np.log(A)) + np.multiply(1-Y, np.log(1-A))))
    iteration.append(j)
    dAL = - (np.divide(Y, A) - np.divide(1-Y, 1-A))
    #dAL = A - Y
    grads['Z'+str(3)] = Z
    grads['A'+str(3)] = A
    
    grads_desc = backward_propagation(dAL, grads, m, params)
    
    
    for i in range(1, len(network)):
        params['W'+str(i)] = params['W'+str(i)] - (0.01 * grads_desc['dw'+str(i)])
        params['b'+str(i)] = params['b'+str(i)] - (0.01 * grads_desc['db'+str(i)])
    
    

    
    
plt.plot(iteration, cost)
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from numpy.core.fromnumeric import argmin

from startkit import *

#Ridge regression constants
# polyOrder = [ 2*i for i in range(1,7)] # for ridge regression
# polyOrder = [ i for i in range(1,17)] # for kernelized ridge regression
polyOrder = [ i for i in range(1,25)] # for kernelized ridge regression of circle
batch_size = int(n_train/4)
X_batches = [ X_train[k*batch_size:(k+1)*batch_size,:] for k in range(4)]
y_batches = [ y_train[k*batch_size:(k+1)*batch_size] for k in range(4)]


def plot():
    
    pos = y[:] == +1.0
    neg = y[:] == -1.0

    plt.figure()
    plt.xlabel("$x_0$")
    plt.ylabel("$x_1$")
    plt.scatter(X[pos,0].ravel(), X[pos,1].ravel(),c='red',marker="v",label="$y=+1$")
    plt.scatter(X[neg,0].ravel(), X[neg,1].ravel(),c='blue',marker="+",label="$y=-1$")
    plt.legend()
    plt.show()

def ridge_reg():
    Train_Errors = []
    Validate_Errors = []
    for d in polyOrder:
        Ridges = []
        Phi = assemble_feature(X_train,d)
        w_ridge = lstsq(Phi,y_train,LAMBDA)

        error = np.linalg.norm(Phi @ w_ridge - y_train)
        Train_Errors.append(error)
        Ridges.append(w_ridge)

        Phi = assemble_feature(X_valid,d)
        validate_error = np.linalg.norm(Phi @ w_ridge - y_valid)
        Validate_Errors.append(validate_error)

        heatmap(lambda x0,x1: assemble_feature(np.array([[x0,x1]]),d)@w_ridge,clip=1)

    Averaged_Train_Errors = [np.mean(Train_Errors[i]) for i in range(len(Train_Errors))]
    print("Averaged_Train_Errors: ",Averaged_Train_Errors)
    print("Validate_Errors: ",Validate_Errors)

    plt.figure()
    plt.plot(polyOrder,Averaged_Train_Errors,label="Training errors")
    plt.plot(polyOrder,Validate_Errors,label="Validation errors")
    plt.xlabel("Polynomial order")
    plt.ylabel("Errors")
    plt.legend()
    plt.show()


def k(x1,x2,p):
    return((1+x1@x2)**p)

def k_matrix(X_1data,X_2data,p):
    k_m = np.zeros((np.shape(X_1data)[0],np.shape(X_2data)[0]))
    for i in range(np.shape(k_m)[0]):
        for j in range(np.shape(k_m)[1]):
            x_i = X_1data[i,:]
            x_j = X_2data[j,:]
            k_m[i,j] = k(x_i,x_j,p)
    return(k_m)

def kern_alpha(X_train,X_test,y_train,p):
    K = k_matrix(X_train,X_train,p)
    A = np.linalg.inv(K+LAMBDA*np.identity(np.shape(K)[0]))
    alpha = A @ y_train
    y_predict = k_matrix(X_test,X_train,p)@ alpha
    return alpha,y_predict

def ridge_kernelized():
    Train_Errors = []
    Validate_Errors = []
    for d in polyOrder:
        Ridges = []
        K = k_matrix(X_train,X_train,d)
        alpha, y_predict = kern_alpha(X_train,X_valid,y_train,d)

        error = np.linalg.norm(K @ alpha - y_train)

        Train_Errors.append(error)
        Ridges.append(alpha)

        validate_error = np.linalg.norm(y_predict - y_valid)
        Validate_Errors.append(validate_error)

        #heatmap(lambda x0,x1: k_matrix(np.array([[x0,x1]]),d)@alpha,clip=1)

    Averaged_Train_Errors = [np.mean(Train_Errors[i]) for i in range(len(Train_Errors))]
    print("Averaged_Train_Errors: ",Averaged_Train_Errors)
    print("Validate_Errors: ",Validate_Errors)
    print("\n")

    plt.figure()
    plt.plot(polyOrder,Averaged_Train_Errors,label="Training errors")
    plt.plot(polyOrder,Validate_Errors,label="Validation errors")
    plt.xlabel("Polynomial order")
    plt.ylabel("Errors")
    plt.legend()
    plt.show()

    

if __name__ == "__main__":
    ridge_kernelized()

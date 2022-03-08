#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

# choose the data you want to load
data = np.load('circle.npz')
# data = np.load('heart.npz')
# data = np.load('asymmetric.npz')

# SPLIT = 0.8
SPLIT = 0.85 # kernel ridge of circle 
X = data["x"]
y = data["y"]
X /= np.max(X)  # normalize the data

n_train = int(X.shape[0] * SPLIT)
X_train = X[:n_train:, :]
X_valid = X[n_train:, :]
y_train = y[:n_train]
y_valid = y[n_train:]

LAMBDA = 0.001


def lstsq(A, b, lambda_=0):
    return np.linalg.solve(A.T @ A + lambda_ * np.eye(A.shape[1]), A.T @ b)


def heatmap(f, clip=5):
    # example: heatmap(lambda x, y: x * x + y * y)
    # clip: clip the function range to [-clip, clip] to generate a clean plot
    #   set it to zero to disable this function

    xx0 = xx1 = np.linspace(np.min(X), np.max(X), 72)
    x0, x1 = np.meshgrid(xx0, xx1)
    x0, x1 = x0.ravel(), x1.ravel()

    #print("x0 shape ",np.shape(x0))
    #print("x0 ",x0)
    #print("x1 shape ",np.shape(x1))
    #print("x1 ",x1)

    z0 = np.zeros((np.shape(x0)[0],1))
    for i in range(np.shape(z0)[0]):
        #print("line #",i)
        if i==0 :
            print("x0[0] shape ",x0[i])
            print("x1[0] shape ",x1[i])
        z0[i,0] = f(float(x0[i]), float(x1[i]))
    #w = w.reshape(len(w),1)
    #print(np.shape(w))
    #print(np.shape(z1))
    #z0 = w @ z1
    #print(np.shape(z0.T))

    if clip:
        z0[z0 > clip] = clip
        z0[z0 < -clip] = -clip

    plt.hexbin(x0, x1, C=z0, gridsize=50, cmap=cm.jet, bins=None)
    plt.colorbar()
    cs = plt.contour(
        xx0, xx1, z0.reshape(xx0.size, xx1.size), [-2, -1, -0.5, 0, 0.5, 1, 2], cmap=cm.jet)
    plt.clabel(cs, inline=1, fontsize=10)

    pos = y[:] == +1.0
    neg = y[:] == -1.0
    plt.scatter(X[pos, 0], X[pos, 1], c='red', marker='+')
    plt.scatter(X[neg, 0], X[neg, 1], c='blue', marker='v')
    plt.show()


def assemble_feature(x, D):
    """Create a vector of polynomial features up to order D from x"""
    ### start poly_features ###
    from scipy.special import binom
    xs = []
    for d0 in range(D + 1):
        for d1 in range(D - d0 + 1):
            # non-kernel polynomial feature
            # xs.append((x[:, 0]**d0) * (x[:, 1]**d1))
            # # kernel polynomial feature
            xs.append((x[:, 0]**d0) * (x[:, 1]**d1) * np.sqrt(binom(D, d0) * binom(D - d0, d1)))
    poly_x = np.column_stack(xs)
    ### end poly_features ###
    return poly_x

def main():
    # example usage of heatmap
    #heatmap(lambda x0, x1: x0 * x0 + x1 * x1)
    print("ok")


if __name__ == "__main__":
    main()

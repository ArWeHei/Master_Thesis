import numpy as np
import math
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import matplotlib
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

import argparse

## distance between start and end point should be >= min_dist
#brush_size = .1
#min_dist = brush_size * 2
#sp = np.array([.1, 0.7])
#ep = np.array([0.8, 0.3])
##while np.linalg.norm(sp - ep) < min_dist:
##    sp = np.random.uniform(low=low, high=high, size=2)
##    ep = np.random.uniform(low=low, high=high, size=2)
#
#theta = np.pi / 2.0
## rotates a vector by 90 degrees
#R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
## vector from start point to end point
#q1 = sp.reshape(-1, 1) - ep.reshape(-1, 1)
#q1 /= np.linalg.norm(q1)
## rotate by 90 degrees
#q2 = np.dot(R, q1)
#
#dist = np.linalg.norm(sp - ep)
#lam1 = dist / 200.0
#lam2 = dist / 25.0
#
## build covariance matrix
#Q = np.concatenate((q1, q2), axis=1)
#Lam = np.array([[lam1, 0], [0, lam2]])
#Sigma = np.dot(np.dot(Q, Lam), Q.T)
#
#mu = (sp + ep) / 2.0
#cp = np.random.multivariate_normal(mean=mu, cov=Sigma, size=1000)
#cp = np.clip(cp, 0, 1)
#
#fig = plt.figure(figsize=(10, 10))
#ax = fig.add_subplot(1, 1, 1)
#ax.scatter(cp[:,0], cp[:,1])
#ax.scatter(sp[0], sp[1], c='red')
#ax.scatter(ep[0], ep[1], c='red')
#ax.scatter(mu[0], mu[1], c='black')
#ax.set_xlim(0, 1)
#ax.set_ylim(0, 1)
#plt.savefig('scatter.pgf')
#plt.show()


#x = [
#    [0,0],
#    [1,4],
#    [2,8],
#    [0,7],
#    [7,7],
#    [7,8],
#    [10,2],
#    [0,1],
#    [1,1],
#    [5,5],
#    [6,5],
#    [3,1],
#    [10,10],
#    [6,3],
#    [6,6],
#    [8,0],
#]
#y = [
#    0,
#    0,
#    1,
#    1,
#    1,
#    1,
#    1,
#    0,
#    0,
#    1,
#    1,
#    0,
#    1,
#    1,
#    1,
#    1,
#]
#x = np.array(x)
#
#fig = plt.figure(figsize=(10, 10))
#ax = fig.add_subplot(1, 1, 1)
#ax.scatter(x[:,0], x[:,1], c=y, cmap=cm.coolwarm)
#ax.plot([-1, 0, 6, 7],[7, 6, 0, -1])
#x_coord = np.linspace(-1, 11, 100)
#y_coord = np.linspace(-1, 11, 100)
#xv, yv = np.meshgrid(x_coord, y_coord)
#decisions = (xv + yv) > 6
#ax.contourf(xv, yv, decisions, alpha=.1)
#
#ax.set_xlabel("\# ``weight loss''")
#ax.set_ylabel("\# ``invest''")
#
#plt.savefig('scatter_percep.pgf')
#plt.show()


#fig, ax = plt.subplots(1, 2, figsize=(20, 10))
#
#x = [ [0,0], [1,1], [1,0], [0,1] ]
#y = [0, 1, 1, 1]
#x = np.array(x)
#
#ax[0].scatter(x[:,0], x[:,1], c=y, cmap=cm.coolwarm, s=500)
#ax[0].set_xlabel("$x_1$")
#ax[0].set_ylabel("$x_2$")
#ax[0].set_title("OR")
#ax[0].set_yticks([1.0, 0.0])
#ax[0].set_yticklabels(["True", "False"])
#ax[0].set_xticks([1.0, 0.0])
#ax[0].set_xticklabels(["True", "False"])
##ax[0].legend()
#
#x = [ [0,0], [1,1], [1,0], [0,1] ]
#y = [0, 0, 1, 1]
#x = np.array(x)
#
#ax[1].scatter(x[:,0], x[:,1], c=y, cmap=cm.coolwarm, s=500)
#ax[1].set_xlabel("$x_1$")
#ax[1].set_ylabel("$x_2$")
#ax[1].set_title("XOR")
#ax[1].set_yticks([1.0, 0.0])
#ax[1].set_yticklabels(["True", "False"])
#ax[1].set_xticks([1.0, 0.0])
#ax[1].set_xticklabels(["True", "False"])
##ax[1].legend()
#
#
#plt.savefig('scatter_XOR.pgf')


def boltzman(x, xmid, tau):
    """
    evaluate the boltzman function with midpoint xmid and time constant tau
    over x
    """
    return 1. / (1. + np.exp(-(x-xmid)/tau))

def relu(x):
    return np.maximum(0, x)

def lrelu(x):
    return np.maximum(0.2*x, x)

x = np.arange(-6, 6, .01)
S = boltzman(x, 0, 1)
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.plot(x, S, color='red', lw=2)
plt.savefig('sigma.pgf')

S = relu(x)
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.plot(x, S, color='red', lw=2)
plt.savefig('relu.pgf')

S = lrelu(x)
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.plot(x, S, color='red', lw=2)
plt.savefig('lrelu.pgf')



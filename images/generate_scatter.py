import numpy as np
import math
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

import argparse

# distance between start and end point should be >= min_dist
brush_size = .1
min_dist = brush_size * 2
sp = np.array([.1, 0.7])
ep = np.array([0.8, 0.3])
#while np.linalg.norm(sp - ep) < min_dist:
#    sp = np.random.uniform(low=low, high=high, size=2)
#    ep = np.random.uniform(low=low, high=high, size=2)

theta = np.pi / 2.0
# rotates a vector by 90 degrees
R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
# vector from start point to end point
q1 = sp.reshape(-1, 1) - ep.reshape(-1, 1)
q1 /= np.linalg.norm(q1)
# rotate by 90 degrees
q2 = np.dot(R, q1)

dist = np.linalg.norm(sp - ep)
lam1 = dist / 200.0
lam2 = dist / 25.0

# build covariance matrix
Q = np.concatenate((q1, q2), axis=1)
Lam = np.array([[lam1, 0], [0, lam2]])
Sigma = np.dot(np.dot(Q, Lam), Q.T)

mu = (sp + ep) / 2.0
cp = np.random.multivariate_normal(mean=mu, cov=Sigma, size=1000)
cp = np.clip(cp, 0, 1)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1)
ax.scatter(cp[:,0], cp[:,1])
ax.scatter(sp[0], sp[1], c='red')
ax.scatter(ep[0], ep[1], c='red')
ax.scatter(mu[0], mu[1], c='black')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
plt.savefig('scatter.pgf')
plt.show()


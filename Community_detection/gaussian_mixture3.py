# Imports:
import os
import xarray
# for plots
import matplotlib.pyplot as plt
import matplotlib
# the usual
import numpy as np
import deepgraph as dg
import pandas as pd
import itertools
import scipy
import argparse
import sklearn
import plotting as pt
import con_sep as cp
import seaborn as sns
import math
import cppv
# gaussian mixture clustering
from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.mixture import GaussianMixture
from matplotlib.colors import ListedColormap
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib import cm

############### Argparser ################

def make_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", help="Give the path to the extreme value dataset.",
                        type=str)
    return parser

parser = make_argparser()
args = parser.parse_args()

cpv = pd.read_csv(args.data)
cpv['x_calc_centroid'] = cpv['x_calc_centroid'].astype(float)
cpv['y_calc_centroidy'] = cpv['y_calc_centroidy'].astype(float)

#cpv['time']=pd.to_datetime(cpv['time'])
X = pd.DataFrame(columns = [ 'x_centroids', 'y_centroids','ytime_mean'])
X['x_centroids'] = cpv.x_calc_centroid
X['y_centroids'] = cpv.y_calc_centroidy
X['ytime_mean'] = cpv.ytime_mean                            
range_n_clusters = [2, 3, 4, 5, 6]
for n in range_n_clusters:
    # define the model
    model = GaussianMixture(n_components=n)
    # fit the model
    model.fit(X)
    # assign a cluster to each example
    yhat = model.predict(X)
    X['cluster'] = yhat
    # retrieve unique clusters
    clusters = unique(yhat)
    # create scatter plot for samples from each cluster
    sns.set(style = "darkgrid")
    fig = plt.figure()

    ax = fig.add_subplot(111, projection = '3d')

    cmap = cm.get_cmap('viridis', n)
    ax.set_xlabel("x_centroids")
    ax.set_ylabel("y_centroids")
    ax.set_zlabel("day of year mean")

    ax.scatter(xs=X.x_centroids,ys=X.y_centroids,zs=X.ytime_mean, c=[matplotlib.cm.spectral(float(i) /10) for i in X.cluster])    
    fig.savefig("../../Results/gaussian3_%s.png" % n)

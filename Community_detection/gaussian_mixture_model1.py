# Imports:
import os
import xarray
# for plots
import matplotlib.pyplot as plt
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

############### Argparser ################

def make_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", help="Give the path to the extreme value dataset.",
                        type=str)
    return parser

parser = make_argparser()
args = parser.parse_args()

cpv = pd.read_csv(args.data)
#cpv['time']=pd.to_datetime(cpv['time'])

# define the model
model = GaussianMixture(n_components=4)
# fit the model
model.fit(cpv['ytime_mean','timespan'])
# assign a cluster to each example
yhat = model.predict(cpv['ytime_mean','timespan'])
# retrieve unique clusters
clusters = unique(yhat)
# create scatter plot for samples from each cluster
for cluster in clusters:
	# get row indexes for samples with this cluster
	row_ix = where(yhat == cluster)
	# create scatter of these samples
	plt.scatter(cpv[row_ix, 0], cpv[row_ix, 1])
# show the plot
plt.savefig("../../Results/gaussian1.png")

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
import cppv as cp
from sklearn.cluster import KMeans
import seaborn as sns
import math

############### Argparser ################

def make_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", help="Give the path to the extreme value dataset.",
                        type=str)
    return parser

parser = make_argparser()
args = parser.parse_args()

gv = pd.read_csv(args.data)
g,cpv = cp.cr_cpv(gv)
print(math.cos(cpv.loc[2].longitude_mean))
cpv['cos_lat'] = cpv['latitude_mean']*np.cos(cpv['latitude_mean'])
km = KMeans(n_clusters=5)
y_predicted = km.fit_predict(cpv[['longitude_mean','latitude_mean','cos_lat']])
cpv['F'] = y_predicted

sns.scatterplot(cpv['longitude_mean'],cpv['latitude_mean'], hue=cpv['F'])
plt.savefig("../../Results/k_means.png")


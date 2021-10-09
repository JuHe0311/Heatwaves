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

############### Argparser ################

def make_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", help="Give the path to the extreme value dataset.",
                        type=str)
    parser.add_argument("-do", "--data_original", help="Give the path to the original value dataset.",
                        type=str)
    return parser

parser = make_argparser()
args = parser.parse_args()

extr = pd.read_csv(args.data)
vt = pd.read_csv(args.data_original)
g,cpg,cpv = cp.create_cpv(extr,vt)

km = KMeans(n_clusters=5)
y_predicted = km.fit_predict(cpv[['longitude_mean','latitude_mean']])
cpv['F'] = y_predicted

sns_plot = sns.scatterplot(cpv['longitude_mean'],cpv['latitude_mean'], hue=cpv['F'])
sns_plot.savefig("../../Results/k_means.png")


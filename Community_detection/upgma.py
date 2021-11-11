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
from sklearn.model_selection import ShuffleSplit
import seaborn as sns
from scipy.cluster.hierarchy import linkage, fcluster
import cppv
import con_sep as cs
import plotting as pt

############### Argparser #################

def make_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", help="Give the path to the dataset to be worked on.",
                        type=str)
    parser.add_argument("-do", "--original_data", help="Give the path to the original dataset to be worked on.",
                        type=str)
    return parser

parser = make_argparser()
args = parser.parse_args()
extr = pd.read_csv(args.data)
vt = pd.read_csv(args.original_data)

g,cpg,cpv = cppv.create_cpv(extr)

# clustering step
    
# create condensed distance matrix
dv = 1 - cpg.e.intsec_strength.values
del cpg.e

# create linkage matrix
lm = linkage(dv, method='average', metric='euclidean')
del dv

# form flat clusters and append their labels to cpv
cpv['F'] = fcluster(lm, 15, criterion='maxclust')
del lm

# relabel families by size
f = cpv['F'].value_counts().index.values
fdic = {j: i for i, j in enumerate(f)}
cpv['F'] = cpv['F'].apply(lambda x: fdic[x])

pt.raster_plot_families(cpg,'10 biggest')

# create F col
g.v['F'] = np.ones(len(g.v), dtype=int) * -1
gcpv = cpv.groupby('F')
it = gcpv.apply(lambda x: x.index.values)

for F in range(len(it)):
    cp_index = g.v.cp.isin(it.iloc[F])
    g.v.loc[cp_index, 'F'] = F

# feature funcs
def n_cp_nodes(cp):
    return len(cp.unique())

feature_funcs = {'daily_mag': [np.sum],
                 'latitude': np.min,
                 'longitude': np.min,
                 'cp': n_cp_nodes}

# create family-g_id intersection graph
fgv = g.partition_nodes(['F', 'g_id'], feature_funcs=feature_funcs)
fgv.rename(columns={'latitude_amin': 'latitude',
                    'longitude_amin': 'longitude',
                    'cp_n_cp_nodes': 'n_cp_nodes'}, inplace=True)

pt.plot_families(15,fgv,vt,'families')



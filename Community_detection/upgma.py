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
    return parser

parser = make_argparser()
args = parser.parse_args()
gv = pd.read_csv(args.data)

g,cpv = cppv.cr_cpv(gv)

# initiate DeepGraph
cpg = dg.DeepGraph(cpv)
# create edges
cpg.create_edges(connectors=[cs.cp_node_intersection, 
                             cs.cp_intersection_strength],
                 no_transfer_rs=['intsec_card'],
                 logfile='create_cpe',
                 step_size=1e7)
print(cpg.e.intsec_strength.value_counts())
# clustering step
    
# create condensed distance matrix
dv = 1 - cpg.e.intsec_strength.values
del cpg.e

# create linkage matrix
lm = linkage(dv, method='average', metric='euclidean')
del dv
# calculate full dendrogram
plt.figure(figsize=(60, 40))
plt.title('UPGMA Heatwave Clustering')
plt.xlabel('heatwave index')
plt.ylabel('distance')
dendrogram(
    lm,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.savefig('../../Results/dendrogram.png')
# form flat clusters and append their labels to cpv
cpv['F'] = fcluster(lm, 8, criterion='maxclust')
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

pt.plot_families(8,fgv,gv,'families')

g.v.to_csv(path_or_buf = "../../Results/gv_f.csv", index=False)



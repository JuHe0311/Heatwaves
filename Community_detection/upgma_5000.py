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
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import cppv
import con_sep as cs
import plotting as pt

############### Argparser #################

def make_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", help="Give the path to the dataset to be worked on.",
                        type=str)
    parser.add_argument("-lsm", "--land_sea_mask", help="Give the path to the land sea mask to be worked on.",
                        type=str)
    return parser

parser = make_argparser()
args = parser.parse_args()
gvv = pd.read_csv(args.data)
lsc = xarray.open_dataset(args.land_sea_mask)

#create integer based (x,y) coordinates
lsc['x'] = (('longitude'), np.arange(len(lsc.longitude)))
lsc['y'] = (('latitude'), np.arange(len(lsc.latitude)))

#convert to dataframe
vt = lsc.to_dataframe()
#reset index
vt.reset_index(inplace=True)

gv = pd.merge(gvv,vt, on=['x','y'])
gv.drop(columns=['latitude_y','longitude_y','time_y'], inplace=True)
gv.rename(columns={'latitude_x': 'latitude','longitude_x':'longitude','time_x':'time'}, inplace=True)

gv['time']=pd.to_datetime(gv['time'])
gv.sort_values('time', inplace=True)
g = dg.DeepGraph(gv)
# create the edges of the graph --> based on neighboring grids in a 3D dataset
feature_funcs = {'time': [np.min, np.max],
                 'itime': [np.min, np.max],
                 't2m': [np.mean],
                   'magnitude': [np.sum],
                 'latitude': [np.mean],
                 'longitude': [np.mean], 't2m': [np.max], 'lsm':[np.mean]}
# partition the node table
cpv, cgv = g.partition_nodes('cp', feature_funcs, return_gv=True)

# append geographical id sets
cpv['g_ids'] = cgv['g_id'].apply(set)
# append cardinality of g_id sets
cpv['n_unique_g_ids'] = cpv['g_ids'].apply(len)
# append time spans
cpv['dt'] = cpv['time_amax'] - cpv['time_amin']
#rename feature name
cpv.rename(columns={'magnitude_sum': 'HWMId_magnitude'}, inplace=True)

# remove heatwaves with predominantly water coverage
cpv["keep"] = np.where(cpv['lsm_mean'] > 0.5, True, False)
cpv2 = cpv.loc[cpv['keep'] == True]
cpv2.drop(columns=['keep'], inplace=True)

# only use the largest x clusters
cpv2 = cpv2.iloc[:5000]
# initiate DeepGraph
cpg = dg.DeepGraph(cpv2)
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
plt.savefig('../../Results/dendrogram5000.png')
# form flat clusters and append their labels to cpv
cpv2['F'] = fcluster(lm, 6, criterion='maxclust')
del lm

# relabel families by size
f = cpv2['F'].value_counts().index.values
fdic = {j: i for i, j in enumerate(f)}
cpv2['F'] = cpv2['F'].apply(lambda x: fdic[x])

pt.raster_plot_families(cpg,'10 biggest5000')

# create F col
g.v['F'] = np.ones(len(g.v), dtype=int) * -1
gcpv = cpv2.groupby('F')
it = gcpv.apply(lambda x: x.index.values)

for F in range(len(it)):
    cp_index = g.v.cp.isin(it.iloc[F])
    g.v.loc[cp_index, 'F'] = F

# feature funcs
def n_cp_nodes(cp):
    return len(cp.unique())

feature_funcs = {'magnitude': [np.sum],
                 'latitude': np.min,
                 'longitude': np.min,
                 'cp': n_cp_nodes}

# create family-g_id intersection graph
fgv = g.partition_nodes(['F', 'g_id'], feature_funcs=feature_funcs)
fgv.rename(columns={'latitude_amin': 'latitude',
                    'longitude_amin': 'longitude',
                    'cp_n_cp_nodes': 'n_cp_nodes'}, inplace=True)

pt.plot_families5000(6,fgv,gv,'families')

g.v.to_csv(path_or_buf = "../../Results/gv_f5000.csv", index=False)

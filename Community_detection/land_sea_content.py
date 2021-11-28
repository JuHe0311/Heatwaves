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
vt['lsm_new'] = np.where(vt["lsm"] <= 0.5, 1, 0)


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
                 'longitude': [np.mean], 't2m': [np.max], 'lsm_new':[np.mean]}
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

plt.hist(cpv.lsm_new_mean, bins=10)
plt.title("Percentage of land content in heatwaves")
plt.xlabel('Land content')
plt.ylabel('Number heatwaves')
plt.savefig('../../Results/land_content.png')

cpv["keep"] = np.where(cpv['lsm_new_mean'] > 0.2, True, False)
cpv2 = cpv.loc[cpv['keep'] == True]
cpv2.drop(columns=['keep'], inplace=True)


cpv.v.to_csv(path_or_buf = "../../Results/cpv_lsc.csv", index=False)

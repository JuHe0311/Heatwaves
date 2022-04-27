# takes net_cdf file and converts it into a pandas dataframe with xarray
# creates integer based coordinates
# saves pandas dataframe under Results

# data i/o
import xarray
import argparse
# the usual
import numpy as np
import pandas as pd
import deepgraph as dg
import plotting as plot
import con_sep as cs
import matplotlib.pyplot as plt
import seaborn as sns
### Argparser ###

def make_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", help="Give the path to the original dataset to be worked on.",
                        type=str)
    return parser

parser = make_argparser()
args = parser.parse_args()
gv = pd.read_csv(args.data)
gv['time']=pd.to_datetime(gv['time'])
g = dg.DeepGraph(gv)

# create supernodes from deep graph by partitioning the nodes by cp
# feature functions applied to the supernodes
feature_funcs = {'time': [np.min, np.max],
                 't2m': [np.mean],
                 'magnitude': [np.sum],
                 'latitude': [np.mean],
                 'longitude': [np.mean], 
                 't2m': [np.max],'ytime':[np.mean],}
# partition graph
cpv, ggv = g.partition_nodes('cp', feature_funcs, return_gv=True)
# append neccessary stuff
# append geographical id sets
cpv['g_ids'] = ggv['g_id'].apply(set)
# append cardinality of g_id sets
cpv['n_unique_g_ids'] = cpv['g_ids'].apply(len)
# append time spans
cpv['dt'] = cpv['time_amax'] - cpv['time_amin']
# append time spans
cpv['timespan'] = cpv.dt.dt.days+1
# not neccessary for precipitation
cpv.rename(columns={'magnitude_sum': 'HWMId_magnitude'}, inplace=True)
print(cpv)

# plot seaborn pairplot

sns.pairplot(g.v, x_vars=['n_nodes','magnitude', 'days', 'ytime'], y_vars=['n_nodes','magnitude', 'days', 'ytime'], kind='reg');

sns.pairplot(cpv, x_vars=['n_nodes','HWMId_magnitude', 'timespan', 'ytime_mean'], y_vars=['n_nodes','HWMId_magnitude', 'timespan', 'ytime_mean'], kind='reg');


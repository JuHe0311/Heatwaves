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
    parser.add_argument("-n", "--ndvi", help="Give the path to the ndvi dataset to be worked on.",
                        type=str)

    return parser

parser = make_argparser()
args = parser.parse_args()
extr = pd.read_csv(args.data)
d = pxarray.open_dataset(args.ndvi)

#create integer based (x,y) coordinates
d['x'] = (('X'), np.arange(len(d.X)))
d['y'] = (('Y'), np.arange(len(d.Y)))

#convert to dataframe
dt = d.to_dataframe()

#reset index
dt.reset_index(inplace=True)

dt.mask(dt.ndvi < 0, inplace=True)

dt.dropna(inplace=True)

# add correct times
datetimes = pd.to_datetime(dt['T'])
# assign your new columns
dt['day'] = datetimes.dt.day
dt['month'] = datetimes.dt.month
dt['year'] = datetimes.dt.year


dt.X = dt.X.apply(d_round)
dt.Y = dt.Y.apply(d_round)

# code sniplet calculates the ndvi anomaly at day 8 of september for every year, for every location
# n is the final pandas dataframe with all anomalies for the 8th of september
ndvi = dg.DeepGraph(dt)
ndvi.filter_by_values_v('month',9)
ndvi.filter_by_values_v('day',8)
feature_funcs = {'ndvi': [np.mean]}
ndvi_cpv,ndvi_gv = ndvi.partition_nodes(['x','y'],feature_funcs=feature_funcs,return_gv=True)
ndvi_cpv.reset_index(inplace=True)
n = pd.merge(ndvi.v,ndvi_cpv, on=["x", 'y'])
n.drop(columns=['n_nodes','day','month'], inplace=True)
n['anomaly'] = n['ndvi']-n['ndvi_mean']
n.rename(columns={'X': 'longitude', 'Y':'latitude', 'T':'time'}, inplace=True)

data = pd.merge(extr,n, how='left',on=['x', 'y', 'year'])
data.drop(columns=['time_y','longitude_y','latitude_y'], inplace=True)
data.rename(columns={'latitude_x': 'latitude', 'longitude_x':'longitude', 'time_x':'time'}, inplace=True)
data['anomaly'] = data['anomaly'].fillna(0)

# sort by time
data.sort_values('time', inplace=True)
g = dg.DeepGraph(data)
# create the edges of the graph --> based on neighboring grids in a 3D dataset
g.create_edges_ft(ft_feature=('itime', 1), 
                  connectors=[grid_2d_dx, grid_2d_dy], 
                  selectors=[s_grid_2d_dx, s_grid_2d_dy],
                  r_dtype_dic={'ft_r': np.bool,
                               'dx': np.int8,
                               'dy': np.int8}, 
                  max_pairs=1e7)

# rename fast track relation
g.e.rename(columns={'ft_r': 'dt'}, inplace=True)
# all singular components (components comprised of one node only)
# are consolidated under the label 0
g.append_cp(consolidate_singles=True)
# we don't need the edges any more
del g.e

# create supernode table of connected nodes --> partitioning of the graph by the component membership of the nodes
# feature functions, will be applied to each component of g
feature_funcs = {'time': [np.min, np.max],
                 'itime': [np.min, np.max],
                 't2m': [np.mean],
                 'magnitude': [np.sum],
                 'latitude': [np.mean],
                 'longitude': [np.mean], 't2m': [np.max], 'anomaly': [np.sum,np.mean]
                }

# partition the node table
cpv, gv = g.partition_nodes('cp', feature_funcs, return_gv=True)

# append geographical id sets
cpv['g_ids'] = gv['g_id'].apply(set)

# append cardinality of g_id sets
cpv['n_unique_g_ids'] = cpv['g_ids'].apply(len)

# append time spans
cpv['dt'] = cpv['time_amax'] - cpv['time_amin']
# add correct times
datetimes = pd.to_datetime(cpv['time_amin'])
cpv['year'] = datetimes.dt.year


#somehow change spatial coverage aka find a good synonmy for this
# append spatial coverage
def area(group):
    return group.drop_duplicates('g_id').t2m.sum()
cpv['area'] = gv.apply(area)

cpv.rename(columns={'magnitude_sum': 'HWMId_magnitude'}, inplace=True)
# discard singular components from cpv
#cpv.drop(0, inplace=True)
#cpv['dt']=pd.to_timedelta(cpv['dt'])
#filter out small heatwaves which are shorter than 3 days (in the dt 2 days)
a = pd.Timedelta(days=1)
cpv["keep"] = np.where(cpv.dt > a, True, False)
cpv = cpv[cpv.keep != False]
cpv.drop(columns=['keep'], inplace=True)
# filter out small events from g by only keeping the cps that are in cpv
cpv.reset_index(inplace=True)
cps = set(cpv.cp)
g.filter_by_values_v('cp', cps)
cpv.set_index('cp', inplace=True)

# accumulate all heatwaves of the same year and calculate features

gpv = dg.DeepGraph(cpv)

# create supernode table of connected nodes --> partitioning of the graph by the component membership of the nodes
# feature functions, will be applied to each component of g
feature_funcs = {'HWMId_magnitude': [np.sum],
                 'latitude_mean': [np.mean],
                 'longitude_mean': [np.mean],
                'anomaly_mean':[np.mean]}

# partition the node table
y_heat, y_heat_g = gpv.partition_nodes('year', feature_funcs, return_gv=True)
# append geographical id sets
y_heat.rename(columns={'n_nodes': 'number of heatwaves'}, inplace=True)


# pplot anomalies correlated to number of heatwaves in a year
plt.figure(figsize=(10,8), dpi=80)
graph = sns.lineplot(x="number of heatwaves", y="anomaly_mean_mean",
             data=y_heat)
plt.savefig('../../Results/no_heatwaves.png',
                       dpi=300, bbox_inches='tight')

# plot anomalies correlated to the hwmid sum
plt.figure(figsize=(10,8), dpi=80)
graph = sns.lineplot(x="HWMId_magnitude_sum", y="anomaly_mean_mean",
             data=y_heat)
plt.savefig('../../Results/hwmid.png',
                       dpi=300, bbox_inches='tight')

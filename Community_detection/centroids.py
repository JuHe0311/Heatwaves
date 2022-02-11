#imports
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
import con_sep as cs
import argparse
from collections import Counter

# Functions

def calc_centroid(x):
    xes = list(x)
    d_x = Counter(xes)
    x_bar = 0
    for key, value in d_x.items():
        x_bar = x_bar + key*value
    x_bar = 1/len(xes)*x_bar
    return x_bar

def calc_centroidy(y):
    yes = list(y)
    d_y = Counter(yes)
    y_bar = 0
    for key, value in d_y.items():
        y_bar = y_bar + key*value
    y_bar = 1/len(yes)*y_bar
    return y_bar
  
#argparser

def make_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--nodes", help="Give the path to the original dataset to be worked on.",
                        type=str)
    return parser

parser = make_argparser()
args = parser.parse_args()
d = pd.read_csv(args.nodes)
d['time']=pd.to_datetime(d['time'])
d.sort_values('time', inplace=True)
g = dg.DeepGraph(d)
# create the edges of the graph --> based on neighboring grids in a 3D dataset
g.create_edges_ft(ft_feature=('itime', 1), 
                  connectors=[cs.grid_2d_dx, cs.grid_2d_dy], 
                  selectors=[cs.s_grid_2d_dx, cs.s_grid_2d_dy],
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
                 'longitude': [np.mean], 
                 't2m': [np.max], 
                 'ytime':[np.min, np.max, np.mean], 'x':[calc_centroid],'y':[calc_centroidy]}

# partition the node table
cpv, gv = g.partition_nodes('cp', feature_funcs, return_gv=True)

# append geographical id sets
cpv['g_ids'] = gv['g_id'].apply(set)

# append geographical id sets
cpv['ytimes'] = gv['ytime'].apply(set)

# append cardinality of g_id sets
cpv['n_unique_g_ids'] = cpv['g_ids'].apply(len)

# append time spans
cpv['dt'] = cpv['time_amax'] - cpv['time_amin']

# append time spans
cpv['timespan'] = cpv.dt.dt.days+1

cpv.rename(columns={'magnitude_sum': 'HWMId_magnitude'}, inplace=True)
# discard singular components
#cpv.drop(0, inplace=True)
#
a = pd.Timedelta(days=1)
cpv['dt']=pd.to_timedelta(cpv['dt'])
cpv["keep"] = np.where(((cpv.dt > a)&(cpv.n_unique_g_ids > 1000)), True, False)
cpv = cpv[cpv.keep != False]
cpv.drop(columns=['keep'], inplace=True)
# filter out small events from g by only keeping the cps that are in cpv
cpv.reset_index(inplace=True)
print(cpv)
cps = set(cpv.cp)
g.filter_by_values_v('cp', cps)
cpv.set_index('cp', inplace=True)
gvg = g.v
gvg.to_csv(path_or_buf = "../../Results/gv_95_nosmall_centr.csv", index=False)
cpv.to_csv(path_or_buf = "../../Results/cpv_95_nosmall_centr.csv", index=False)

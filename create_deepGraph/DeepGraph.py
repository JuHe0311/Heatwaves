# Imports:
import os
import xarray
# for plots
import matplotlib.pyplot as plt
# the usual
import numpy as np
import pandas as pd
import itertools
import scipy
import argparse
############### Argparser #############


def make_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", help="Give the path to the dataset to be worked on.",
                        type=str)
    return parser


parser = make_argparser()
args = parser.parse_args()

data_path = args.data

extr = pd.DataFrame(data_path)
print(extr)
# connectors calculate the distance between every pair of nodes based on their 3D location
# connectors

# distance between x coordinates of two nodes
def grid_2d_dx(x_s, x_t):
    dx = x_t - x_s
    return dx

# distance between y coordinates of two nodes
def grid_2d_dy(y_s, y_t):
    dy = y_t - y_s
    return dy


# selectors
def s_grid_2d_dx(dx, sources, targets):
    dxa = np.abs(dx)
    sources = sources[dxa <= 1]
    targets = targets[dxa <= 1]
    return sources, targets

def s_grid_2d_dy(dy, sources, targets):
    dya = np.abs(dy)
    sources = sources[dya <= 1]
    targets = targets[dya <= 1]
    return sources, targets


  
# create the graph
g = dg.DeepGraph(extr)

# create the edges of the graph --> based on neighboring grids in a 3D dataset
g.create_edges_ft(ft_feature=('itime', 1), 
                  connectors=[grid_2d_dx, grid_2d_dy], 
                  selectors=[s_grid_2d_dx, s_grid_2d_dy],
                  r_dtype_dic={'ft_r': np.bool,
                               'dx': np.int8,
                               'dy': np.int8}, 
                  max_pairs=1e7)

# all singular components (components comprised of one node only)
# are consolidated under the label 0
g.append_cp(consolidate_singles=True)
# we don't need the edges any more
del g.e

# rename fast track relation
g.e.rename(columns={'ft_r': 'dt'}, inplace=True)

# create supernode table of connected nodes --> partitioning of the graph by the component membership of the nodes
# feature functions, will be applied to each component of g
feature_funcs = {'time': [np.min, np.max],
                 'itime': [np.min, np.max],
                 't2m': [np.mean],
                 'daily_mag': [np.sum],
                 'latitude': [np.mean],
                 'longitude': [np.mean], 't2m': [np.max]}

# partition the node table
cpv, gv = g.partition_nodes('cp', feature_funcs, return_gv=True)

# append geographical id sets
cpv['g_ids'] = gv['g_id'].apply(set)

# append cardinality of g_id sets
cpv['n_unique_g_ids'] = cpv['g_ids'].apply(len)

# append time spans
cpv['dt'] = cpv['time_amax'] - cpv['time_amin']

cpv.rename(columns={'daily_mag_sum': 'HWMId_magnitude'}, inplace=True)

# discard singular components
cpv.drop(0, inplace=True)
# save cpv
cpv.to_csv(path_or_buf = "../../Results/cpv.csv", index=False)

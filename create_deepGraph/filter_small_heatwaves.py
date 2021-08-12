# code filters out small heatwaves (based on time)
# heatwaves that are shorter than 3 days are removed from the dataset

######## Imports ########

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
    parser.add_argument("-do", "--data_original", help="Give the path to the original dataset.",
                        type=str)
    parser.add_argument("-de", "--data_extreme", help="Give the path to the extreme dataset.",
                        type=str)
    parser.add_argument("-dcpv", "--cpv", help="Give the path to the partitioned dataset.",
                        type=str)
    return parser


parser = make_argparser()
args = parser.parse_args()

# create variables
vt_path = pd.DataFrame(args.data_original)
cpv_path = pd.DataFrame(args.cpv)
#extr_path = pd.DataFrame(args.data_extreme)

vt = pd.read_csv(vt_path)
cpv = pd.read_csv(cpv_path)
extr = pd.read_csv(args.data_extreme)

# create g
g = dg.DeepGraph(extr)

# connectors and selectors
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


  
# filter out small heatwave events from the initial deep graph g
vt['small'] = np.ones(len(vt), dtype=int) * -1
gcpv = cpv.groupby('small')
it = gcpv.apply(lambda x: x.index.values)

for small in range(len(it)):
    cp_index = g.v.cp.isin(it.iloc[small])
    g.v.loc[cp_index, 'small'] = small
g.filter_by_values_v('small', 1)

# redo cpv and cpg as they did not take over the changes made to g

# create supernode table of connected nodes --> partitioning of the graph by the component membership of the nodes
# feature functions, will be applied to each component of g
feature_funcs = {'time': [np.min, np.max],
                 'itime': [np.min, np.max],
                 't2m': [np.mean],
                 'daily_mag': [np.sum],
                 'latitude': [np.mean],
                 'longitude': [np.mean], 't2m': [np.max]}

# partition the node table
cpv_small, gv = g.partition_nodes('cp', feature_funcs, return_gv=True)

# append geographical id sets
cpv_small['g_ids'] = gv['g_id'].apply(set)

# append cardinality of g_id sets
cpv_small['n_unique_g_ids'] = cpv_small['g_ids'].apply(len)

# append time spans
cpv_small['dt'] = cpv_small['time_amax'] - cpv_small['time_amin']

cpv_small.rename(columns={'daily_mag_sum': 'HWMId_magnitude'}, inplace=True)

# create superedges between the supernodes to find heatwave clusters with strong regional overlaps

# compute intersection of geographical locations
def cp_node_intersection(g_ids_s, g_ids_t):
    intsec = np.zeros(len(g_ids_s), dtype=object)
    intsec_card = np.zeros(len(g_ids_s), dtype=np.int)
    for i in range(len(g_ids_s)):
        intsec[i] = g_ids_s[i].intersection(g_ids_t[i])
        intsec_card[i] = len(intsec[i])
    return intsec_card

# compute a spatial overlap measure between clusters
def cp_intersection_strength(n_unique_g_ids_s, n_unique_g_ids_t, intsec_card):
    min_card = np.array(np.vstack((n_unique_g_ids_s, n_unique_g_ids_t)).min(axis=0), 
                        dtype=np.float64)
    intsec_strength = intsec_card / min_card
    return intsec_strength

# compute temporal distance between clusters
def time_dist(dtime_amin_s, dtime_amin_t):
    dt = dtime_amin_t - dtime_amin_s
    return dt

# initiate DeepGraph
cpg = dg.DeepGraph(cpv_small)

cpg.create_edges(connectors=[cp_node_intersection, 
                             cp_intersection_strength],
                 no_transfer_rs=['intsec_card'],
                 logfile='create_cpe',
                 step_size=1e7)

cpv_small.to_csv(path_or_buf = "../../Results/cpv_small.csv", index=False)


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

#### connectors and selectors needed for the deep graphs #####

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

  
#### create cpv dataset --> partition nodes based on their spatio-temporal neigborhood ####  
# create the graph
def create_cpv(extr_data):
  extr_data['time']=pd.to_datetime(extr_data['time'])
  g = dg.DeepGraph(extr_data)

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
  #rename feature name
  cpv.rename(columns={'daily_mag_sum': 'HWMId_magnitude'}, inplace=True)

  # discard singular components
  cpv.drop(0, inplace=True)
  cpv['dt']=pd.to_timedelta(cpv['dt'])

  ###### filter out small heatwaves that are shorter than 2 days #####
  # initiate DeepGraph
  cpg = dg.DeepGraph(cpv)

  cpg.v['small'] = 0
  for i in range(1,len(cpg.v)):
      if cpg.v.dt.loc[i].days > 1:
          cpg.v.small.loc[i] = 1        

  # create edges
  cpg.create_edges(connectors=[cp_node_intersection, 
                             cp_intersection_strength],
                 no_transfer_rs=['intsec_card'],
                 logfile='create_cpe',
                 step_size=1e7)
  
  # filter out small heatwave events from the initial deep graph g
  vt['small'] = np.ones(len(vt), dtype=int) * -1
  gcpv = cpv.groupby('small')
  it = gcpv.apply(lambda x: x.index.values)

  for small in range(len(it)):
      cp_index = g.v.cp.isin(it.iloc[small])
      g.v.loc[cp_index, 'small'] = small
  g.filter_by_values_v('small', 1)

  # redo cpv and cpg as they did not take over the changes made to g
  feature_funcs = {'time': [np.min, np.max],
                 'itime': [np.min, np.max],
                 't2m': [np.mean],
                 'daily_mag': [np.sum],
                 'latitude': [np.mean],
                 'longitude': [np.mean], 't2m': [np.max]}
  cpv_small, gv = g.partition_nodes('cp', feature_funcs, return_gv=True)
  cpv_small['g_ids'] = gv['g_id'].apply(set)
  cpv_small['n_unique_g_ids'] = cpv_small['g_ids'].apply(len)
  cpv_small['dt'] = cpv_small['time_amax'] - cpv_small['time_amin']
  cpv_small.rename(columns={'daily_mag_sum': 'HWMId_magnitude'}, inplace=True)

  # initiate DeepGraph
  cpg = dg.DeepGraph(cpv_small)

  cpg.create_edges(connectors=[cp_node_intersection, 
                             cp_intersection_strength],
                 no_transfer_rs=['intsec_card'],
                 logfile='create_cpe',
                 step_size=1e7)
  cpv_small.to_csv(path_or_buf = "../../Results/cpv_small.csv", index=False)
  return cpv_small

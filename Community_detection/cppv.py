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
import con_sep as cs


#### create cpv dataset --> partition nodes based on their spatio-temporal neigborhood ####  
# create the graph
def create_cpv(extr_data, vt):
  extr_data['time']=pd.to_datetime(extr_data['time'])
  g = dg.DeepGraph(extr_data)

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
 # del g.e

  # create supernode table of connected nodes --> partitioning of the graph by the component membership of the nodes
  # feature functions, will be applied to each component of g
  feature_funcs = {'time': [np.min, np.max],
                 'itime': [np.min, np.max],
                 't2m': [np.mean],
                 'latitude': [np.mean],
                 'longitude': [np.mean], 't2m': [np.max]}
  # include: 'daily_mag': [np.sum],
  # partition the node table
  cpv, cpg = g.partition_nodes('cp', feature_funcs, return_gv=True)

  # append geographical id sets
  cpv['g_ids'] = cpg['g_id'].apply(set)
  # append cardinality of g_id sets
  cpv['n_unique_g_ids'] = cpv['g_ids'].apply(len)
  # append time spans
  cpv['dt'] = cpv['time_amax'] - cpv['time_amin']
  #rename feature name
  #cpv.rename(columns={'daily_mag_sum': 'HWMId_magnitude'}, inplace=True)

  # discard singular components
  cpv.drop(0, inplace=True)
  cpv['dt']=pd.to_timedelta(cpv['dt'])
  print(cpv)
  ###### filter out small heatwaves that are shorter than 2 days #####
  a = pd.Timedelta(days=1)
  cpv["keep"] = np.where(cpv.dt > a, True, False)
  cpv = cpv[cpv.keep != False]
  cpv.drop(columns=['keep'], inplace=True)
  print(cpv)
  # filter out small events from g by only keeping the cps that are in cpv
  cpv.reset_index(inplace=True)
  cps = set(cpv.cp)
  g.filter_by_values_v('cp', cps)
  cpv.set_index('cp', inplace=True)
  cpv.to_csv(path_or_buf = "../../Results/cpv_small.csv", index=False)
  gv = g.v
  gv.to_csv(path_or_buf = "../../Results/gv.csv", index=False)
  return g,cpg,cpv




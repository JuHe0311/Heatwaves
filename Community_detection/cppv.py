### Imports ###
import numpy as np
import deepgraph as dg
import pandas as pd
import con_sep as cs


#### create cpv dataset --> partition nodes based on their spatio-temporal neigborhood ####  
# input: extreme dataset on which a deep graph should be built, b is the spatial threshold: how many unique grid cells does a heat wave need to be 
# considered a heat wave
# returns: g --> nodes table, cpv --> supernodes table, gv --> pandas groupby object
def create_cpv(extr_data, b):
  extr_data['time']=pd.to_datetime(extr_data['time'])
  extr_data.sort_values('time', inplace=True)
  
  # create the deep graph
  g = dg.DeepGraph(extr_data)
  
  # create the edges of the graph --> based on neighboring grids in a 3D dataset
  g.create_edges_ft(ft_feature=('itime', 1), 
                  connectors=[cs.grid_2d_dx, cs.grid_2d_dy], 
                  selectors=[cs.s_grid_2d_dx, cs.s_grid_2d_dy],
                  r_dtype_dic={'ft_r': np.bool,
                               'dx': np.int8,
                               'dy': np.int8}, 
                  max_pairs=1e7)
  
  # rename the edges
  g.e.rename(columns={'ft_r': 'dt'}, inplace=True)
  
  # all singular components are consolidated under the label 0
  g.append_cp(consolidate_singles=True)
  
  # delete the edges
  del g.e
  
  # create supernode table of connected nodes
  # feature functions, will be applied to each component of g
  # supernodes are heat waves
  feature_funcs = {'time': [np.min, np.max],
                   'itime': [np.min, np.max],
                   't2m': [np.mean],
                   'magnitude': [np.sum],
                   'latitude': [np.mean],
                   'longitude': [np.mean], 
                   't2m': [np.max],
                   'ytime':[np.mean]}

  g.v['ytime'] = g.v.ytime.astype(int)
  g.v['x'] = g.v.x.astype(int)
  g.v['y'] = g.v.y.astype(int)

  # partition the nodes table based on their component membership
  cpv, gv = g.partition_nodes('cp', feature_funcs, return_gv=True)

  # append geographical id sets
  cpv['g_ids'] = gv['g_id'].apply(set)
  # append cardinality of g_id sets
  cpv['n_unique_g_ids'] = cpv['g_ids'].apply(len)
  # append day of year sets
  cpv['ytimes'] = gv['ytime'].apply(set)
  # append time spans
  cpv['dt'] = cpv['time_amax'] - cpv['time_amin']
  cpv['timespan'] = cpv.dt.dt.days+1
  #rename feature name
  cpv.rename(columns={'magnitude_sum': 'HWMId_magnitude'}, inplace=True)

  cpv['dt']=pd.to_timedelta(cpv['dt']) 
  
  #filter out small heatwaves that are shorter than 3 days and that have less than b different grid ids
  a = pd.Timedelta(days=1)
  cpv["keep"] = np.where(((cpv.dt > a)&(cpv.n_unique_g_ids > b)), True, False)
  cpv = cpv[cpv.keep != False]
  cpv.drop(columns=['keep'], inplace=True)
  # filter out small events from g by only keeping the cps that are in cpv
  cpv.reset_index(inplace=True)
  cps = set(cpv.cp)
  g.filter_by_values_v('cp', cps)
  
  # append cp column to cpv
  cpv['cpp'] = cpv['cp'] 
  cpv.set_index('cpp', inplace=True)
  return g,gv,cpv



  

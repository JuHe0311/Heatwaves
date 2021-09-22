# this module contains all functions for different plotting styles of the results

#### Imports ####
# for plots
import matplotlib.pyplot as plt
# the usual
import numpy as np
import deepgraph as dg
import pandas as pd
import itertools
import scipy


def plot_clusters(nodes, plot_title):
  # idea: create partition by g_ids: count how many heatwaves contain the g_id, plot this then
    # feature functions, will be applied to each g_id
    # I need to adapt the feature functions somehow!

    # first: create intersection partition of grid ids and cp, feature functions sum up how many nodes there are 
    # at one grid id for one heatwave (one cp)
    feature_funcs = {'g_id': [np.sum],
                     'latitude': np.min,
                     'longitude': np.min,}
    ggpv_wt = dg.DeepGraph(nodes)
    sv = ggpv_wt.partition_nodes(['g_id', 'cp'], feature_funcs)


    # second: partition again by g_id and apply no specific feature functions
    feature_funcs = {'latitude_amin': np.min,
                     'longitude_amin': np.min,
                     }
    ggpv_wt = dg.DeepGraph(sv)
    sv = ggpv_wt.partition_nodes(['g_id'], feature_funcs)
    # reset the g_id index to make it to a column
    sv.reset_index(level=0, inplace=True)
    # rename columns: n_heatwaves: number of how many heatwaves have the grid id included in their set of grid ids
    sv.rename(columns={'latitude_amin': 'lat',
                        'longitude_amin': 'lon',
                        'n_nodes': 'n_heatwaves'}, inplace=True)
    # for all heatwaves of the first, second or third cluster from walktrap community plot the geographical distribution
    # how many heatwaves hit one geographical location?
    # feature functions, will be applied to each g_id
    feature_funcs = {'latitude_amin': np.min,
                     'longitude_amin': np.min}

    gv = dg.DeepGraph(sv)
    gv=sv
    #gv = ggpv_wt1.partition_nodes('g_id', feature_funcs)
    gv.rename(columns={'latitude_amin': 'lat',
                       'longitude_amin': 'lon'}, inplace=True)


    cols = {plot_title: gv.n_heatwaves}

    for name, col in cols.items():

        # for easy filtering, we create a new DeepGraph instance for 
        # each component
        gt = dg.DeepGraph(gv)

        # configure map projection
        kwds_basemap = {'llcrnrlon': vt.longitude.min() - 1,
                        'urcrnrlon': vt.longitude.max() + 1,
                        'llcrnrlat': vt.latitude.min() - 1,
                        'urcrnrlat': vt.latitude.max() + 1}
    
        # configure scatter plots
        kwds_scatter = {'s': 3400,
                        'marker': 's',
                        'c': col.values,
                        'cmap': 'viridis_r',
                        'alpha': .5,
                        'edgecolors': 'none'}

        # create scatter plot on map
        obj = gt.plot_map(lon='lon', lat='lat',
                          kwds_basemap=kwds_basemap,
                          kwds_scatter=kwds_scatter)

        # configure plots
        obj['m'].drawcoastlines(linewidth=.8)
        obj['m'].drawparallels(range(-90, 90, 2), linewidth=.2)
        obj['m'].drawmeridians(range(0, 360, 2), linewidth=.2)
        obj['m'].drawcountries()
        obj['ax'].set_title(name)
    
        # colorbar
        cb = obj['fig'].colorbar(obj['pc'], fraction=.022, pad=.02)
        cb.set_label('{}'.format(name), fontsize=15) 
        obj['fig'].savefig('../../Results/clust_{:03d}.png'.format(i),
                       dpi=300, bbox_inches='tight')
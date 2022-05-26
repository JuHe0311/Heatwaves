# This module contains all functions for different plotting styles of the results

### Imports ###
import matplotlib.pyplot as plt
import numpy as np
import deepgraph as dg
import seaborn as sns
# plots heat wave families or clusters on a map        
def plot_hits(number_families,fgv,v,plot_title):
  families = np.arange(number_families)
  for F in families:

    # for easy filtering, we create a new DeepGraph instance for 
    # each component
    gt = dg.DeepGraph(fgv.loc[F])

    # configure map projection
    kwds_basemap = {'llcrnrlon': v.longitude.min() - 1,
                    'urcrnrlon': v.longitude.max() + 1,
                    'llcrnrlat': v.latitude.min() - 1,
                    'urcrnrlat': v.latitude.max() + 1}

    # configure scatter plots
    kwds_scatter = {'s': 1,
                        'c': gt.v.n_nodes.values,
                        'cmap': 'viridis_r',
                        'edgecolors': 'none'}

    # create scatter plot on map
    obj = gt.plot_map(lat='latitude', lon='longitude',kwds_basemap=kwds_basemap, kwds_scatter=kwds_scatter)

    # configure plots
    obj['m'].drawcoastlines(linewidth=.8)
    obj['m'].drawparallels(range(-50, 50, 20), linewidth=.2)
    obj['m'].drawmeridians(range(0, 360, 20), linewidth=.2)
    cb = obj['fig'].colorbar(obj['pc'], fraction=.022, pad=.02)
    cb.set_label('Number of Heatwave Days', fontsize=15) 
    obj['ax'].set_title('%s Cluster %s' % (plot_title,F))
    
    obj['fig'].savefig('../../Results/Heatwavedays_%s_Cluster_%s.png' % (plot_title,F),
                       dpi=300, bbox_inches='tight')

    
# plots heat wave families or clusters on a map        
def plot_families(number_families,fgv,v,plot_title):
  families = np.arange(number_families)
  for F in families:

    # for easy filtering, we create a new DeepGraph instance for 
    # each component
    gt = dg.DeepGraph(fgv.loc[F])

    # configure map projection
    kwds_basemap = {'llcrnrlon': v.longitude.min() - 1,
                    'urcrnrlon': v.longitude.max() + 1,
                    'llcrnrlat': v.latitude.min() - 1,
                    'urcrnrlat': v.latitude.max() + 1}

    # configure scatter plots
    kwds_scatter = {'s': 1,
                        'c': gt.v.n_cp_nodes.values,
                        'cmap': 'viridis_r',
                        'edgecolors': 'none'}

    # create scatter plot on map
    obj = gt.plot_map(lat='latitude', lon='longitude',kwds_basemap=kwds_basemap, kwds_scatter=kwds_scatter)

    # configure plots
    obj['m'].drawcoastlines(linewidth=.8)
    obj['m'].drawparallels(range(-50, 50, 20), linewidth=.2)
    obj['m'].drawmeridians(range(0, 360, 20), linewidth=.2)
    cb = obj['fig'].colorbar(obj['pc'], fraction=.022, pad=.02)
    cb.set_label('Number of Heatwaves', fontsize=15) 
    obj['ax'].set_title('%s Cluster %s' % (plot_title,F))
    
    obj['fig'].savefig('../../Results/%s_Cluster_%s.png' % (plot_title,F),
                       dpi=300, bbox_inches='tight')

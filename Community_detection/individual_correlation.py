# data i/o
import xarray
import argparse
# the usual
import numpy as np
import pandas as pd
import deepgraph as dg
#import plotting as plot
import con_sep as cs
import matplotlib.pyplot as plt
import seaborn as sns

### Functions ###

def plot_hits(fgv,v,plot_title):

    # for easy filtering, we create a new DeepGraph instance for 
    # each component
    gt = dg.DeepGraph(fgv)

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
    obj['ax'].set_title('%s' % (plot_title))
    
    obj['fig'].savefig('../../Results/Heatwavedays_%s_Cluster %s.png' % (plot_title,F),
                       dpi=300, bbox_inches='tight')

    
# plots heat wave families or clusters on a map        
def plot_families(fgv,v,plot_title):
    # for easy filtering, we create a new DeepGraph instance for 
    # each component
    gt = dg.DeepGraph(fgv)

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
    obj['ax'].set_title('%s' % (plot_title))
    
    obj['fig'].savefig('../../Results/%s_Cluster %s.png' % (plot_title,F),
                       dpi=300, bbox_inches='tight')


### Argparser ###

def make_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n_nodes", help="Give the path to the original dataset to be worked on.",
                        type=str)
    parser.add_argument("-hwmid", "--hwmid", help="Give the path to the original dataset to be worked on.",
                        type=str)
    parser.add_argument("-ndvi", "--ndvi", help="Give the path to the original dataset to be worked on.",
                        type=str)
    parser.add_argument("-d", "--data", help="Give the path to the original dataset to be worked on.",
                        type=str)
    return parser

parser = make_argparser()
args = parser.parse_args()
gv = pd.read_csv(args.data)
n_nodes = pd.read_csv(args.n_nodes)
hwmid = pd.read_csv(args.hwmid)
ndvi = pd.read_csv(args.hwmid)
gv['time']=pd.to_datetime(gv['time'])


#########################################################
# Plot the 10 most highly correlated heat waves

# add absolute value column
n_nodes['absolute_values'] = np.absolute(n_nodes['corr'])
hwmid['absolute_values'] = np.absolute(hwmid['corr'])

n_nodes.sort_values(by='absolute_values',axis=0,inplace=True,ascending=False)
hwmid.sort_values(by='absolute_values',axis=0,inplace=True,ascending=False)

# plot the heat waves with the 10 highest correlation values for heat wave days
for i in range(10):
    year_nodes = n_nodes.year.iloc[i]
    clust_nodes = n_nodes.cluster.iloc[i]
    gv_tmp = gv[gv.time.dt.year == year_nodes]
    gv_tmp = gv_tmp[gv_tmp.F_upgma == clust_nodes]
    # feature funcs
    def n_cp_nodes(cp):
        return len(cp.unique())
    feature_funcs = {'magnitude': [np.sum],
                     'latitude': np.min,
                     'longitude': np.min,
                     'cp': n_cp_nodes}
    k_means_dg = dg.DeepGraph(gv_tmp)

    # create family-g_id intersection graph
    fgv = k_means_dg.partition_nodes(['F_upgma', 'g_id'], feature_funcs=feature_funcs)
    fgv.rename(columns={'cp_n_cp_nodes': 'n_cp_nodes', 'longitude_amin':'longitude','latitude_amin':'latitude'}, inplace=True)
    plot_families(fgv,gv,'Number of Heatwaves, Correlation: %s' % n_nodes['corr'].iloc[i])
    plot_hits(fgv,gv,'Number of Hits, Correlation: %s' % n_nodes['corr'].iloc[i])

# plot the heat waves with the 10 highest correlation values for heat wave magnitude
for i in range(10):
    year_nodes = hwmid.year.iloc[i]
    clust_nodes = hwmid.cluster.iloc[i]
    gv_tmp = gv[gv.time.dt.year == year_nodes]
    gv_tmp = gv_tmp[gv_tmp.F_upgma == clust_nodes]
    # feature funcs
    def n_cp_nodes(cp):
        return len(cp.unique())
    feature_funcs = {'magnitude': [np.sum],
                     'latitude': np.min,
                     'longitude': np.min,
                     'cp': n_cp_nodes}
    k_means_dg = dg.DeepGraph(gv_tmp)

    # create family-g_id intersection graph
    fgv = k_means_dg.partition_nodes(['F_upgma', 'g_id'], feature_funcs=feature_funcs)
    fgv.rename(columns={'cp_n_cp_nodes': 'n_cp_nodes', 'longitude_amin':'longitude','latitude_amin':'latitude'}, inplace=True)
    plot_families(fgv,gv,'Number of Heatwaves, Correlation: %s' % i)
    plot_hits(fgv,gv,'Number of Hits, Correlation: %s' % i)


#########################################################
# Split the dataset into three and compare correlation



# data i/o
import argparse
# the usual
import numpy as np
import pandas as pd
import deepgraph as dg
import plotting as plot
import con_sep as cs
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.cluster import KMeans

### Argparser ###

def make_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", help="Give the path to the nodes dataset to be worked on.",
                        type=str)
    parser.add_argument("-cpv", "--cpv", help="Give the path to the supernodes tables",
                        type=str)
    parser.add_argument("-n", "--number", help="Give the number of heatwaves to be plotted",
                        type=int)
    parser.add_argument("-b", "--by", help="Give the column name by which the heatwaves should be sorted",
                        type=str)
    return parser

parser = make_argparser()
args = parser.parse_args()
cpv = pd.read_csv(args.cpv)
gv = pd.read_csv(args.data)
n = args.number
b = args.by

cpv.sort_values(by=b,inplace=True,ascending=False)

for i in range(1,n):
    cp = cpv.cp.iloc[i-1]
    ggg = dg.DeepGraph(gv[gv.cp==cp])
    start = cpv.time_amin.iloc[i-1]
    end = cpv.time_amax.iloc[i-1]
    # feature funcs
    def n_cp_nodes(cp):
        return len(cp.unique())

    feature_funcs = {'magnitude': [np.sum],
                     'latitude': np.min,
                     'longitude': np.min,
                     'cp': n_cp_nodes}
    # create family-g_id intersection graph
    fgv = ggg.partition_nodes(['g_id'], feature_funcs=feature_funcs)
    fgv.rename(columns={'cp_n_cp_nodes': 'n_cp_nodes', 'longitude_amin':'longitude','latitude_amin':'latitude'}, inplace=True)


    gt = dg.DeepGraph(fgv)
    # configure map projection
    kwds_basemap = {'llcrnrlon': gv.longitude.min() - 1,
                    'urcrnrlon': gv.longitude.max() + 1,
                    'llcrnrlat': gv.latitude.min() - 1,
                    'urcrnrlat': gv.latitude.max() + 1}
    
    

    
    # configure scatter plots
    kwds_scatter = {'s': 2.5,
                    'c': gt.v.magnitude_sum.values,
                    'cmap': 'viridis_r',
                    'alpha': .5,
                    'edgecolors': 'none'}

    # create scatter plot on map
    obj = gt.plot_map(lon='longitude', lat='latitude',
                      kwds_basemap=kwds_basemap,
                      kwds_scatter=kwds_scatter)

    # configure plots
    obj['m'].drawcoastlines(linewidth=.8,zorder=10)
    obj['m'].drawparallels(range(-50, 50, 20), linewidth=.2)
    obj['m'].drawmeridians(range(0, 360, 20), linewidth=.2)
    obj['ax'].set_title('Most Intense Heat Wave %s, %s - %s' % (i,start,end))
    
    # colorbar
    cb = obj['fig'].colorbar(obj['pc'], fraction=.022, pad=.02)
    cb.set_label('Number of Heat Wave Days', fontsize=15) 
    
    obj['fig'].savefig('../../Results/HWMID_intensity2_%s.png' % i,dpi=300, bbox_inches='tight')

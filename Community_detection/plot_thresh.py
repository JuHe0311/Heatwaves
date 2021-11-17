# data i/o
import xarray
import argparse
# the usual
import numpy as np
import pandas as pd
import extr as ex
import matplotlib
import deepgraph as dg
import cppv


### Argparser ###

def make_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--thresh", help="Give the path to the original dataset to be worked on.",
                        type=str)
    return parser

parser = make_argparser()
args = parser.parse_args()
thresh = pd.read_csv(args.thresh)

g = dg.DeepGraph(thresh)
g.filter_by_values_v('year',2010)
g.v

cols = {'thresh': g.v.thresh,
        'thresh0': g.v.thresh0}

for name, col in cols.items():

    # for easy filtering, we create a new DeepGraph instance for 
    # each component
    gt = dg.DeepGraph(g.v)

    # configure map projection
    kwds_basemap = {'llcrnrlon': g.v.longitude.min() - 1,
                    'urcrnrlon': g.v.longitude.max() + 1,
                    'llcrnrlat': g.v.latitude.min() - 1,
                    'urcrnrlat': g.v.latitude.max() + 1}
    
    # configure scatter plots
    kwds_scatter = {'s': 1,
                    'c': col.values,
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
    obj['ax'].set_title(name)
    
    # colorbar
    cb = obj['fig'].colorbar(obj['pc'], fraction=.022, pad=.02)
    cb.set_label('{}'.format(name), fontsize=15) 
    
    obj['fig'].savefig('../../Results/%s.png' % (name),
                       dpi=300, bbox_inches='tight')

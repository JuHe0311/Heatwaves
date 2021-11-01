# data i/o
import os
import xarray
import argparse
# for plots
import matplotlib.pyplot as plt

# the usual
import numpy as np
import pandas as pd

import deepgraph as dg

# notebook display
from IPython.display import HTML

### Argparser ###

def make_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", help="Give the path to the original dataset to be worked on.",
                        type=str)
    return parser

parser = make_argparser()
args = parser.parse_args()
v = pd.read_csv(args.data)

g = dg.DeepGraph(v)
# configure map projection
kwds_basemap = {'llcrnrlon': v.longitude.min() - 1,
                'urcrnrlon': v.longitude.max() + 1,
                'llcrnrlat': v.latitude.min() - 1,
                'urcrnrlat': v.latitude.max() + 1,
                'resolution': 'i'}

# configure scatter plots
kwds_scatter = {'s': 1.5,
                'c': g.v.t2m.values,
                'edgecolors': 'none',
                'cmap': 'viridis_r'}

# create generator of scatter plots on map
objs = g.plot_map_generator('longitude', 'latitude', 'time',
                            kwds_basemap=kwds_basemap,
                            kwds_scatter=kwds_scatter)

# plot and store frames
for i, obj in enumerate(objs):

    # configure plots
    cb = obj['fig'].colorbar(obj['pc'], fraction=0.025, pad=0.01)
    cb.set_label('[mm/h]')
    obj['m'].fillcontinents(color='0.2', zorder=0, alpha=.4)
    obj['ax'].set_title('{}'.format(obj['group']))

    # store and close
    obj['fig'].savefig('../../Results/tmp/pcp_{:03d}.png'.format(i),
                       dpi=300, bbox_inches='tight')
    plt.close(obj['fig'])
# create video with ffmpeg
cmd = "ffmpeg -y -r 5 -i pcp_%03d.png -c:v libx264 -r 20 -vf scale=2052:1004 {}.mp4"
os.system(cmd.format('../../Results/tmp/pcp'))

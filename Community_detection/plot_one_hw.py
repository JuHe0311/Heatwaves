# data i/o
import os
import xarray
import argparse
# for plots
import matplotlib.pyplot as plt
# the usual
import numpy as np
import pandas as pd
import sklearn
import deepgraph as dg
from IPython.display import HTML

### Argparser ###

def make_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", help="Give the path to the original dataset to be worked on.",
                        type=str)
    parser.add_argument("-cpv", "--cpv", help="Give the path to the original dataset to be worked on.",
                        type=str)
    return parser

parser = make_argparser()
args = parser.parse_args()
gv = pd.read_csv(args.data)
cpv = pd.read_csv(args.cpv)
g = dg.DeepGraph(gv)

# temporary DeepGraph instance containing 
# only the largest component
gt = dg.DeepGraph(g.v)
gt.filter_by_values_v('cp', 1)

# configure map projection
from mpl_toolkits.basemap import Basemap
m1 = Basemap(projection='ortho',
             lon_0=cpv.loc[1].longitude_mean + 12,
             lat_0=cpv.loc[1].latitude_mean + 8,
             resolution=None)
width = (m1.urcrnrx - m1.llcrnrx) * .65
height = (m1.urcrnry - m1.llcrnry) * .45

kwds_basemap = {'projection': 'ortho',
                'lon_0': cpv.loc[1].longitude_mean + 12,
                'lat_0': cpv.loc[1].latitude_mean + 8,
                'llcrnrx': -0.5 * width,
                'llcrnry': -0.5 * height,
                'urcrnrx': 0.5 * width,
                'urcrnry': 0.5 * height,
                'resolution': 'i'}

# configure scatter plots
kwds_scatter = {'s': 2,
                'c': np.log(gt.v.t2m.values / 100.),
                'edgecolors': 'none',
                'cmap': 'viridis_r'}

# create generator of scatter plots on map
objs = gt.plot_map_generator('longitude', 'latitude', 'time',
                              kwds_basemap=kwds_basemap,
                              kwds_scatter=kwds_scatter)

# plot and store frames
for i, obj in enumerate(objs):

    # configure plots
    obj['m'].fillcontinents(color='0.2', zorder=0, alpha=.4)
    #obj['m'].drawparallels(range(-50, 50, 20), linewidth=.2)
    #obj['m'].drawmeridians(range(0, 360, 20), linewidth=.2)
    obj['ax'].set_title('{}'.format(obj['group']))

    # store and close
    obj['fig'].savefig('../Results/cp1_ortho_{:03d}.png'.format(i), 
                       dpi=300, bbox_inches='tight')
    plt.close(obj['fig'])
# create video with ffmpeg
cmd = "ffmpeg -y -r 5 -i ../Results/cp1_ortho_%03d.png -c:v libx264 -r 20 -vf scale=1919:1406 {}.mp4"
os.system(cmd.format('precipitation_files/cp1_ortho'))

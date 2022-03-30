# Imports:
import xarray
# the usual
import numpy as np
import deepgraph as dg
import pandas as pd
import argparse

############### Functions #################


############### Argparser #################

def make_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ndvi', "--ndvi_data", help="Give the path to the ndvi dataset to be worked on.",
                       type=str)
    parser.add_argument('-c', "--coarsen_factor", help="Give the factor by which to downsample the grid of the ndvi",
                       type=int)

    return parser

parser = make_argparser()
args = parser.parse_args()
d = xarray.open_dataset(args.ndvi_data)
c = args.coarsen_factor

# downsample the grid, so the grid fits the grid of the temperature dataset
# always take the maximum value of the grids that are combined (take into account errors that reduce NDVI)
d=d.coarsen(X=c,Y=c,boundary='exact').max()

# calculate monthly anomalies
clm = d.sel(T=slice('1981-07-08 12:00:00','2015-12-24 00:00:00')).groupby('T.month').max(dim='T')
anm = (d.groupby('T.month') - clm)

#create integer based (x,y) coordinates
anm['x'] = (('X'), np.arange(len(anm.X)))
anm['y'] = (('X'), np.arange(len(anm.X)))

dt = anm.to_dataframe()
dt.reset_index(inplace=True)

# add neccessary columns
datetimes = pd.to_datetime(dt['T'])
dt['day'] = datetimes.dt.day
dt['month'] = datetimes.dt.month
dt['year'] = datetimes.dt.year
# append a column indicating geographical locations (i.e., supernode labels)
dt['g_id'] = dt.groupby(['X', 'Y']).grouper.group_info[0]
dt['g_id'] = dt['g_id'].astype(np.uint32)

dt.to_csv(path_or_buf = "../../Results/ndvi_data_prepared.csv", index=False)

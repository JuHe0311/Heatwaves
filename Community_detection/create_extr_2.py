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
import extr as ex

### Argparser ###

def make_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-do", "--original_data", help="Give the path to the original dataset to be worked on.",
                        type=str)
    parser.add_argument("-t", "--thresholds", help="Give the path to the thresholds",
                        type=str)
    return parser

parser = make_argparser()
args = parser.parse_args()
d = xarray.open_dataset(args.original_data)
thresholds = pd.read_csv(args.thresholds)

#create integer based (x,y) coordinates
d['x'] = (('longitude'), np.arange(len(d.longitude)))
d['y'] = (('latitude'), np.arange(len(d.latitude)))
#convert to dataframe
vt = d.to_dataframe()
#reset index
vt.reset_index(inplace=True)
# add correct times
datetimes = pd.to_datetime(vt['time'])
# assign your new columns
vt['day'] = datetimes.dt.day
vt['month'] = datetimes.dt.month
vt['year'] = datetimes.dt.year
# append dayofyear 
vt['ytime'] = vt.time.apply(lambda x: x.dayofyear)
# append integer-based time
times = pd.date_range(vt.time.min(), vt.time.max(), freq='D')
tdic = {time: itime for itime, time in enumerate(times)}
vt['itime'] = vt.time.apply(lambda x: tdic[x])
vt['itime'] = vt['itime'].astype(np.uint16)
# remove 366th day (29th of february, every 4 years)
vt = ex.cut_366(vt)
# convert temperature from kelvin to degrees celcius
ex.conv_to_degreescelcius(vt)

# calculate extreme dataset
extr = ex.extr_events(vt,thresholds)

# calculate the daily magnitudes of the extr dataset
ex.daily_magnitude(vt, extr)

# adapt extreme dataset
# append some neccessary stuff to the extr dataset
# append a column indicating geographical locations (i.e., supernode labels)
extr['g_id'] = extr.groupby(['longitude', 'latitude']).grouper.group_info[0]
extr['g_id'] = extr['g_id'].astype(np.uint32)    

# sort by time
extr.sort_values('time', inplace=True)
#remove columns 75 and 25 percentile
extr.drop(['seventyfive_percentile', 'twentyfive_percentile', 'day', 'month','year'], axis=1, inplace=True)

# save extreme dataset
extr.to_csv(path_or_buf = "../../Results/extreme_dataset.csv", index=False)

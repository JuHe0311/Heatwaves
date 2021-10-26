# takes net_cdf file and converts it into a pandas dataframe with xarray
# creates integer based coordinates
# saves pandas dataframe under Results

# data i/o
import xarray
import argparse
# the usual
import numpy as np
import pandas as pd
import extr as ex
import matplotlib
import deepgraph as dg


### functions ###

### Argparser ###

def make_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-do", "--original_data", help="Give the path to the original dataset to be worked on.",
                        type=str)
    return parser

parser = make_argparser()
args = parser.parse_args()
d = xarray.open_dataset(args.original_data)

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

ex.conv_to_degreescelcius(vt)
vt.to_csv(path_or_buf = "../../Results/degreecs_vt.csv", index=False)

first = np.arange(350,366)
second = np.arange(1,366)
third = np.arange(1,16)
time = np.concatenate((first, second, third), axis=None)

g_t = dg.DeepGraph(vt)
#remove 366th day
ytime = np.arange(366)
g_t.filter_by_values_v('ytime',ytime)
vt.to_csv(path_or_buf = "../../Results/366cut.csv", index=False)

### calculate threshold
# partition the node table
cpv_t, gv_t = g_t.partition_nodes(['x','y','ytime'],return_gv=True)
cpv_t['t2m'] = gv_t['t2m'].apply(list)
cpv_t.reset_index(inplace=True)
vt['thresh'] = np.ones(len(vt), dtype=int) * -1
tmp2 = pd.DataFrame(columns=['x','y','ytime','thresh'])
for i in range(366):
    g = dg.DeepGraph(cpv_t)
    k = time[i:i+31]
    g.filter_by_values_v('ytime', k)
    tmp, tmp_p = g.partition_nodes(['x','y'],return_gv=True)
    tmp['t2m'] = tmp_p['t2m'].apply(list)
    tmp.reset_index(inplace=True)
    tmp['thresh'] = tmp['t2m'].apply(ex.calc_perc)
    tmp.drop(['t2m'],axis=1,inplace=True)
    tmp['ytime'] = i+1
    tmp2 = pd.concat([tmp2,tmp])
result = pd.merge(vt,tmp2, on=["ytime", "x", 'y'])
result.drop(columns=['n_nodes', 'ytime'], inplace=True)
result.to_csv(path_or_buf = "../../Results/thresh.csv", index=False)
print(result)
# calculate extreme dataset

result["keep"] = np.where(result["t2m"] >= result["thresh_x"], True, False)
extr = result.loc[result['keep'] == True]
extr.drop(columns=['keep'], inplace=True)

extr.to_csv(path_or_buf = "../../Results/extr.csv", index=False)

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
import gc

### functions ###
def perc25(a_list):
    threshold = np.percentile(a_list,25)
    return threshold

def perc75(a_list):
    threshold = np.percentile(a_list,75)
    return threshold

def calc_mag(data):
    if data.t2m > data.t2m_amax_perc25:
        mag = (data.t2m-data.t2m_amax_perc25)/(data.t2m_amax_perc75-data.t2m_amax_perc25)
    else:
        mag = 0
    return mag


### Argparser ###

def make_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--threshold_dataset", help="Give the path to the original dataset to be worked on.",
                        type=str)
    parser.add_argument("-d", "--data", help="Give the path to the original dataset to be worked on.",
                        type=str)
    return parser

parser = make_argparser()
args = parser.parse_args()
thresh = pd.read_csv(args.threshold_dataset)


# append some neccessary stuff to the extr dataset
# append a column indicating geographical locations (i.e., supernode labels)
thresh['g_id'] = thresh.groupby(['longitude', 'latitude']).grouper.group_info[0]
thresh['g_id'] = thresh['g_id'].astype(np.uint32)  

# append integer-based time
times = pd.date_range(thresh.time.min(), thresh.time.max(), freq='D')
tdic = {time: itime for itime, time in enumerate(times)}
thresh['itime'] = thresh.time.apply(lambda x: tdic[x])
thresh['itime'] = thresh['itime'].astype(np.uint16)

# calculate extreme dataset
thresh["keep"] = np.where(thresh["t2m"] >= thresh["thresh"], True, False)
extr = thresh.loc[thresh['keep'] == True]
extr.drop(columns=['keep'], inplace=True)

# sort by time
extr.sort_values('time', inplace=True)

# assign your new columns
datetimes = pd.to_datetime(thresh['time'])
thresh['day'] = datetimes.dt.day
thresh['month'] = datetimes.dt.month
thresh['year'] = datetimes.dt.year
# calculate daily magnitude of extreme events
f_funcs = {'t2m': [np.max]}
gg = dg.DeepGraph(thresh)

del thresh
gc.collect()

gg_t = gg.partition_nodes(['x','y','year'],f_funcs)
gg_t.reset_index(inplace=True)
feature_funcs = {'t2m_amax': [perc75,perc25]}
ggt = dg.DeepGraph(gg_t)
ggg = ggt.partition_nodes(['x','y'], feature_funcs)
rex = pd.merge(extr,ggg, on=['x', 'y'])

del extr,gg_t,gg,ggg
gc.collect()

rex.drop(columns=['n_nodes'], inplace=True)
rex['magnitude']=rex.apply(calc_mag, axis=1)
rex.drop(columns=['t2m_amax_perc25','t2m_amax_perc75','thresh'], inplace=True)

# save the extreme dataset
rex.to_csv(path_or_buf = "../../Results/extr_new.csv", index=False)

# create heatwaves from the extreme dataset
rex.sort_values('time', inplace=True)
g,cpg,cpv = cppv.create_cpv(rex)

# save heatwaves
cpv.to_csv(path_or_buf = "../../Results/cpv_new.csv", index=False)
g.v.to_csv(path_or_buf = "../../Results/gv_new.csv", index=False)

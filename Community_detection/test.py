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
import cppv

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
    parser.add_argument("-t", "--threshold", help="Give the path to the original dataset to be worked on.",
                        type=str)
    return parser

parser = make_argparser()
args = parser.parse_args()
result = pd.read_csv(args.threshold)
result['time']=pd.to_datetime(result['time'])
# calculate extreme dataset

result["keep"] = np.where(result["t2m"] >= result["thresh"], True, False)
result["keep0"] = np.where(result["t2m"] >= result["thresh0"], True, False)
extr = result.loc[result['keep'] == True]
extr0 = result.loc[result['keep0'] == True]
extr.drop(columns=['keep'], inplace=True)
extr0.drop(columns=['keep0'], inplace=True)

# append some neccessary stuff to the extr dataset
# append a column indicating geographical locations (i.e., supernode labels)
extr['g_id'] = extr.groupby(['longitude', 'latitude']).grouper.group_info[0]
extr['g_id'] = extr['g_id'].astype(np.uint32)    
extr0['g_id'] = extr0.groupby(['longitude', 'latitude']).grouper.group_info[0]
extr0['g_id'] = extr0['g_id'].astype(np.uint32)    
# append integer-based time
times = pd.date_range(extr.time.min(), extr.time.max(), freq='D')
tdic = {time: itime for itime, time in enumerate(times)}
extr['itime'] = extr.time.apply(lambda x: tdic[x])
extr['itime'] = extr['itime'].astype(np.uint16)

times = pd.date_range(extr0.time.min(), extr0.time.max(), freq='D')
tdic = {time: itime for itime, time in enumerate(times)}
extr0['itime'] = extr0.time.apply(lambda x: tdic[x])
extr0['itime'] = extr0['itime'].astype(np.uint16)
# sort by time
extr.sort_values('time', inplace=True)
extr0.sort_values('time', inplace=True)

# calculate daily magnitude of extreme events
f_funcs = {'t2m': [np.max]}
gg = dg.DeepGraph(vt)
gg_t = gg.partition_nodes(['x','y','year'],f_funcs)
gg_t.reset_index(inplace=True)
feature_funcs = {'t2m_amax': [perc75,perc25]}
ggt = dg.DeepGraph(gg_t)
ggg = ggt.partition_nodes(['x','y'], feature_funcs)
rex = pd.merge(extr,ggg, on=['x', 'y'])
rex.drop(columns=['n_nodes'], inplace=True)
rex['magnitude']=rex.apply(calc_mag, axis=1)
rex.drop(columns=['t2m_amax_perc25','t2m_amax_perc75','thresh'], inplace=True)

rex0 = pd.merge(extr0,ggg, on=['x', 'y'])
rex0.drop(columns=['n_nodes'], inplace=True)
rex0['magnitude']=rex0.apply(calc_mag, axis=1)
rex0.drop(columns=['t2m_amax_perc25','t2m_amax_perc75','thresh'], inplace=True)

rex.to_csv(path_or_buf = "../../Results/extr.csv", index=False)
rex0.to_csv(path_or_buf = "../../Results/extr95.csv", index=False)

rex.sort_values('time', inplace=True)
g,cpg,cpv = cppv.create_cpv(rex)
g0,cpg0,cpv0 = cppv.create_cpv(rex0)

cpv.to_csv(path_or_buf = "../../Results/cpv.csv", index=False)
g.v.to_csv(path_or_buf = "../../Results/gv.csv", index=False)
g0.v.to_csv(path_or_buf = "../../Results/gv0.csv", index=False)
cpv0.to_csv(path_or_buf = "../../Results/cpv95.csv", index=False)

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
    parser.add_argument("-d", "--data", help="Give the path to the original dataset to be worked on.",
                        type=str)
    parser.add_argument("-t", "--thresholds", help="Give the path to the thresholds",
                        type=str)
    parser.add_argument("-m", "--minmax", help="Give the path to the minmax table",
                        type=str)
    return parser

parser = make_argparser()
args = parser.parse_args()
vt = pd.read_csv(args.data)
thresholds = pd.read_csv(args.thresholds)
minmax = pd.read_csv(args.minmax)

# calculate extreme dataset
extr = ex.extr_events(vt,thresholds,minmax)

# adapt extreme dataset
# append some neccessary stuff to the extr dataset
# append a column indicating geographical locations (i.e., supernode labels)
extr['g_id'] = extr.groupby(['longitude', 'latitude']).grouper.group_info[0]
extr['g_id'] = extr['g_id'].astype(np.uint32)    
# sort by time
extr.sort_values('time', inplace=True)
extr.drop(['day', 'month','year'], axis=1, inplace=True)

# save extreme dataset
extr.to_csv(path_or_buf = "../../Results/extreme_dataset.csv", index=False)

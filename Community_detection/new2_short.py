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
    parser.add_argument("-e", "--extreme_dataset", help="Give the path to the original dataset to be worked on.",
                        type=str)
    parser.add_argument("-g", "--g_ids", help="Give the number how many unique g_ids a heatwave needs to be considered a heatwave",
                        type=int)
    return parser

parser = make_argparser()
args = parser.parse_args()
rex = pd.read_csv(args.extreme_dataset)
b = args.g_ids

# create heatwaves from the extreme dataset
rex.sort_values('time', inplace=True)
g,cpg,cpv = cppv.create_cpv(rex,b)

# save heatwaves
cpv.to_csv(path_or_buf = "../../Results/cpv_new.csv", index=False)
g.v.to_csv(path_or_buf = "../../Results/gv_new.csv", index=False)

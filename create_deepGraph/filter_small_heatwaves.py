# code filters out small heatwaves (based on time)
# heatwaves that are shorter than 3 days are removed from the dataset

######## Imports ########

import os
import xarray
# for plots
import matplotlib.pyplot as plt
# the usual
import numpy as np
import pandas as pd
import itertools
import scipy
import argparse

############### Argparser #############


def make_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-do", "--data_original", help="Give the path to the original dataset.",
                        type=str)
    parser.add_argument("-de", "--data_extreme", help="Give the path to the extreme dataset.",
                        type=str)
    parser.add_argument("-dcpv", "--cpv", help="Give the path to the partitioned dataset.",
                        type=str)
    parser.add_argument("-s", "--save_path",  help="Give a path where to save the extreme output dataset",
                        type=str)
    return parser


parser = make_argparser()
args = parser.parse_args()

data_path = args.data_original
extr_path = args.data_extreme
save_path = args.save_path
cpv_path = args.cpv

# create variables
vt = pd.DataFrame(args.data_original)
cpv = pd.DataFrame(args.cpv)
extr = pd.DataFrame(args.data_extreme)

# create g
g = dg.DeepGraph(extr)

# connectors and selectors
# distance between x coordinates of two nodes
def grid_2d_dx(x_s, x_t):
    dx = x_t - x_s
    return dx
# distance between y coordinates of two nodes
def grid_2d_dy(y_s, y_t):
    dy = y_t - y_s
    return dy
# selectors
def s_grid_2d_dx(dx, sources, targets):
    dxa = np.abs(dx)
    sources = sources[dxa <= 1]
    targets = targets[dxa <= 1]
    return sources, targets

def s_grid_2d_dy(dy, sources, targets):
    dya = np.abs(dy)
    sources = sources[dya <= 1]
    targets = targets[dya <= 1]
    return sources, targets

 
  
# filter out small heatwave events from the initial deep graph g
vt['small'] = np.ones(len(vt), dtype=int) * -1
gcpv = cpv.groupby('small')
it = gcpv.apply(lambda x: x.index.values)

for small in range(len(it)):
    cp_index = g.v.cp.isin(it.iloc[small])
    g.v.loc[cp_index, 'small'] = small
g.filter_by_values_v('small', 1)

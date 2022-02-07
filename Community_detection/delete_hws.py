#imports
import os
import xarray
# for plots
import matplotlib.pyplot as plt
# the usual
import numpy as np
import deepgraph as dg
import pandas as pd
import itertools
import scipy
import argparse
import con_sep as cs
import argparse

#argparser

def make_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--supernodes", help="Give the path to the original dataset to be worked on.",
                        type=str)
    parser.add_argument("-n", "--nodes", help="Give the path to the original dataset to be worked on.",
                        type=str)
    return parser

parser = make_argparser()
args = parser.parse_args()
cpv = pd.read_csv(args.supernodes)
g = pd.read_csv(args.nodes)


#
a = pd.Timedelta(days=1)
cpv['dt']=pd.to_timedelta(cpv['dt'])
cpv["keep"] = np.where(((cpv.dt > a)&(cpv.n_unique_g_ids > 2)), True, False)
cpv = cpv[cpv.keep != False]
cpv.drop(columns=['keep'], inplace=True)
# filter out small events from g by only keeping the cps that are in cpv
cpv.reset_index(inplace=True)
print(cpv)
cps = set(cpv.cp)
gg = dg.DeepGraph(g)
gg.filter_by_values_v('cp', cps)
cpv.set_index('cp', inplace=True)
gvg = gg.v
gvg.to_csv(path_or_buf = "../../Results/gv_95_nosmall.csv", index=False)
cpv.to_csv(path_or_buf = "../../Results/cpv_95_nosmall.csv", index=False)

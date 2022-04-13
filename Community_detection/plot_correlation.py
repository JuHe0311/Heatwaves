# Imports:
import xarray
# the usual
import numpy as np
import deepgraph as dg
import pandas as pd
import argparse
import scipy
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import plotting as plot

############### Functions #################

# calculate mean correlation over the years for every family in every cluster
def mean_correlation(data):
    feature_funcs = {'corr':[np.mean]}
    g = dg.DeepGraph(data)
    fgv = g.partition_nodes(['cluster'], feature_funcs=feature_funcs)
    return fgv

############### Argparser #################

def make_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', "--nnodes_corr", help="Give the path to the n_nodes correlation table.",
                       type=str)
    parser.add_argument('-hwmid', "--hwmid_corr", help="Give the path to the hwmid correlation table",
                       type=str)
    return parser

parser = make_argparser()
args = parser.parse_args()
n_nodes_corr = pd.read_csv(args.nnodes_corr)
hwmid_corr = pd.read_csv(args.hwmid_corr)

# add a colum stating whether a correlation is significant
# threshold of significance = 5%
n_nodes_corr['significant'] = np.where(n_nodes_corr.p_value < 0.05, 1,0)
hwmid_corr['significant'] = np.where(hwmid_corr.p_value < 0.05, 1,0)
n_nodes_corr.reset_index(inplace=True)
hwmid_corr.reset_index(inplace=True)
# plot timeseries for every cluster
plot.corr_time_series(n_nodes_corr,'n_nodes')
plot.corr_time_series(hwmid_corr,'hwmid')

# plot boxplots to compare all clusters in 1 family
plot.corr_violinplot(n_nodes_corr,'n_nodes_unfiltered')
plot.corr_violinplot(hwmid_corr,'hwmid_unfiltered')


# remove non-significant values
n_nodes_corr.drop(n_nodes_corr.loc[n_nodes_corr['significant']==0].index,inplace=True)
hwmid_corr.drop(hwmid_corr.loc[hwmid_corr['significant']==0].index,inplace=True)


mean_corr_nodes = mean_correlation(n_nodes_corr)
mean_corr_hwmid = mean_correlation(hwmid_corr)
print('mean correlation n_nodes')
print(mean_corr_nodes)
print('mean correlation hwmid')
print(mean_corr_hwmid)
       
# plot boxplots to compare all clusters in 1 family
plot.corr_violinplot(n_nodes_corr,'n_nodes')
plot.corr_violinplot(hwmid_corr,'hwmid')

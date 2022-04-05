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
n_nodes_corr['significant'] = np.where(n_nodes_corr.p_value > 0.9, 1,0)
hwmid_corr['significant'] = np.where(hwmid_corr.p_value > 0.9, 1,0)
n_nodes_corr.reset_index(inplace=True)
hwmid_corr.reset_index(inplace=True)
# plot timeseries for every cluster
plot.corr_time_series(n_nodes_corr)
plot.corr_time_series(hwmid_corr)
# remove non-significant values
#n_nodes_corr.drop(n_nodes_corr.loc[n_nodes_corr['significant']==0].index,inplace=True)
#hwmid_corr.drop(hwmid_corr.loc[hwmid_corr['significant']==0].index,inplace=True)


# plot boxplots to compare all clusters in 1 family

# somehow print a table?



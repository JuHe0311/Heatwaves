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


############### Functions #################


############### Argparser #################

def make_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', "--nnodes_corr", help="Give the path to the n_nodes correlation table.",
                       type=str)
    parser.add_argument('-h', "--hwmid_corr", help="Give the path to the hwmid correlation table",
                       type=str)
    return parser

parser = make_argparser()
args = parser.parse_args()
n_nodes_corr = pd.read_csv(args.nnodes_corr)
hwmid_corr = pd.read_csv(args.hwmid_corr)

# add a colum stating whether a correlation is significant
n_nodes_corr['significant'] = np.where(n_nodes_corr.p_value > 0.9, 1,0)
hwmid_corr['significant'] = np.where(hwmid_corr.p_value > 0.9, 1,0)

# plot timeseries for every cluster


# plot boxplots to compare all clusters in 1 family

# somehow print a table?



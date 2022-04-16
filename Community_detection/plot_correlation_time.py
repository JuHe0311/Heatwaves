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
import pymannkendall as mk
from sklearn.metrics import r2_score 
############### Functions #################

# calculate mean correlation over the years for every family in every cluster
def mean_correlation(data):
    feature_funcs = {'corr':[np.mean]}
    g = dg.DeepGraph(data)
    fgv = g.partition_nodes(['cluster'], feature_funcs=feature_funcs)
    return fgv


def scatter(data, name):
    res = stats.linregress(data['year'], data['corr'])
    plt.scatter(data['year'],data['corr'])
    print(f"R-squared: {res.rvalue**2:.6f}")
    plt.plot(data['year'],data['corr'], 'o', label='original data')
    plt.plot(data['year'], res.intercept + res.slope*data['year'], 'r', label='y = %s + %s*x' % (res.intercept,res.slope))
    plt.legend()
    # the line equation:
    #print('y=%.6fx+(%.6f)' %(z[0],z[1]))
    plt.ylabel('spearman correlation coefficient')
    plt.xlabel('years')
    plt.title('%s, R_square: %s' % (name,res.rvalue**2))
    plt.savefig('../../Results/scatter_%s.png' % name)
    plt.clf()
    
############### Argparser #################

def make_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', "--nnodes_corr", help="Give the path to the n_nodes correlation table.",
                       type=str)
    parser.add_argument('-hwmid', "--hwmid_corr", help="Give the path to the hwmid correlation table",
                       type=str)
    parser.add_argument('-hc', "--hwmid_clust",nargs='*', help="Give the path to the hwmid correlation table",
                       type=int)
    parser.add_argument('-nc', "--nnodes_clust",nargs='*', help="Give the path to the hwmid correlation table",
                       type=int)
    return parser

parser = make_argparser()
args = parser.parse_args()
n_nodes_corr = pd.read_csv(args.nnodes_corr)
hwmid_corr = pd.read_csv(args.hwmid_corr)
hwmid_clust = args.hwmid_clust
n_nodes_clust = args.nnodes_clust

# add a colum stating whether a correlation is significant
# threshold of significance = 5%
n_nodes_corr['significant'] = np.where(n_nodes_corr.p_value < 0.05, 1,0)
hwmid_corr['significant'] = np.where(hwmid_corr.p_value < 0.05, 1,0)
n_nodes_corr.reset_index(inplace=True)
hwmid_corr.reset_index(inplace=True)

# remove non-significant values
n_nodes_corr.drop(n_nodes_corr.loc[n_nodes_corr['significant']==0].index,inplace=True)
hwmid_corr.drop(hwmid_corr.loc[hwmid_corr['significant']==0].index,inplace=True)

for val in hwmid_clust:
    hwmid_filt = hwmid_corr[hwmid_corr.cluster == val]
    print(mk.original_test(hwmid_filt['corr']))
    scatter(hwmid_filt,'hwmid_%s' % val)

for val in n_nodes_clust:
    n_nodes_filt = n_nodes_corr[n_nodes_corr.cluster == val]
    print(mk.original_test(n_nodes_filt['corr']))
    scatter(n_nodes_filt, 'n_nodes_%s' % val)

  


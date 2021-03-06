### Imports ###
import numpy as np
import deepgraph as dg
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pymannkendall as mk
from sklearn.metrics import r2_score 

### Functions ###
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
    plt.plot(data['year'],data['corr'], 'o')
    plt.plot(data['year'], res.intercept + res.slope*data['year'], 'r', label=f'y = {res.intercept:.2f} + {res.slope:.4f}*x')
    plt.legend()
    plt.ylabel('spearman correlation coefficient')
    plt.xlabel('years')
    plt.title(f'{name}, R_square: {res.rvalue**2:.5f}')
    plt.savefig('../../Results/scatter_%s.png' % name)
    plt.clf()
 
### Argparser ###
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
# threshold of significance = 1%
n_nodes_corr['significant'] = np.where(n_nodes_corr.p_value < 0.01, 1,0)
hwmid_corr['significant'] = np.where(hwmid_corr.p_value < 0.01, 1,0)
n_nodes_corr.reset_index(inplace=True)
hwmid_corr.reset_index(inplace=True)


# remove non-significant values
n_nodes_corr.drop(n_nodes_corr.loc[n_nodes_corr['significant']==0].index,inplace=True)
hwmid_corr.drop(hwmid_corr.loc[hwmid_corr['significant']==0].index,inplace=True)
# save significant correlations
n_nodes_corr.to_csv(path_or_buf = "../../Results/n_nodes_corr_sig.csv", index=False)
hwmid_corr.to_csv(path_or_buf = "../../Results/hwmid_corr_sig.csv", index=False)

mean_corr_nodes = mean_correlation(n_nodes_corr)
mean_corr_hwmid = mean_correlation(hwmid_corr)
print('mean correlation n_nodes')
print(mean_corr_nodes)
print('mean correlation hwmid')
print(mean_corr_hwmid)
hwmid_clust = list(hwmid_corr.cluster.unique())
n_nodes_clust = list(n_nodes_corr.cluster.unique())
       
# create time series plots
for val in hwmid_clust:
    hwmid_filt = hwmid_corr[hwmid_corr.cluster == val]
    if len(hwmid_filt) >= 30:
        print('hwmid')
        print(val)
        print(mk.original_test(hwmid_filt['corr']))
        scatter(hwmid_filt,'hwmid_%s' % val)

for val in n_nodes_clust:
    n_nodes_filt = n_nodes_corr[n_nodes_corr.cluster == val]
    if len(n_nodes_filt) >= 30:
        print('n_nodes')
        print(val)
        print(mk.original_test(n_nodes_filt['corr']))
        scatter(n_nodes_filt, 'n_nodes_%s' % val)

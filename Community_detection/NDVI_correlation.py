# Imports:
import xarray
# the usual
import numpy as np
import deepgraph as dg
import pandas as pd
import argparse
import scipy
from scipy import stats


############### Functions #################

# calculate the seasonal variables of the heatwaves in one family
def seasonal_measures(data,season,ndvi):
    seasons = np.arange(season[0],season[1]+1)
    feature_funcs = {'magnitude':[np.sum]}
    g = dg.DeepGraph(data)
    #g.filter_by_values_v('month',seasons)
    fgv = g.partition_nodes(['g_id'], feature_funcs=feature_funcs)
    fgv.reset_index(inplace=True)
    total = pd.merge(fgv,ndvi, on='g_id','year',how='inner')
    return total

# perform the correlation between two variables
def correlate(data_1,data_2):
    return stats.spearmanr(data_1,data_2,axis=0, nan_policy='omit')

############### Argparser #################

def make_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ndvi', "--ndvi_data", help="Give the path to the ndvi dataset to be worked on.",
                       type=str)
    parser.add_argument('-d', "--temperature_data", help="Give the path to the temperature dataset to be worked on",
                       type=str)
    parser.add_argument('-s', "--season", nargs='*', help="Give the start and end point of the season",
                       type=int)
    return parser

parser = make_argparser()
args = parser.parse_args()
ndvi = pd.read_csv(args.ndvi_data)
t = pd.read_csv(args.temperature_data)
season = args.season
t['time_x']=pd.to_datetime(t['time_x'])
t['year'] = t.time_x.dt.year

# perform the correlation
n_nodes_corr = pd.DataFrame(columns=['year','cluster','corr','p_value'])
hwmid_corr = pd.DataFrame(columns=['year','cluster','corr','p_value'])
upgma_clust = list(t.F_upgma.unique())
years = list(t.year.unique())
for clust in upgma_clust:
    for y in years:
        g = dg.DeepGraph(t)
        g.filter_by_values_v('F_upgma',clust)
        g.filter_by_values_v('year',y)
        corr_matrix = seasonal_measures(g.v,season,ndvi)
        corr1,p_value1 = correlate(corr_matrix.n_nodes,corr_matrix.ndvi)
        corr2,p_value2 = correlate(corr_matrix.magnitude_sum,corr_matrix.ndvi)
        df1 = {'year': y, 'cluster': clust, 'corr': corr1,'p_value':p_value1}
        n_nodes_corr = n_nodes_corr.append(df1, ignore_index = True)
        df2 = {'year': y, 'cluster': clust, 'corr': corr2,'p_value':p_value2}
        hwmid_corr = n_nodes_corr.append(df1, ignore_index = True)


n_nodes_corr.to_csv(path_or_buf = "../../Results/n_nodes_correlation.csv", index=False)
hwmid_corr.to_csv(path_or_buf = "../../Results/hwmid_correlation.csv", index=False)

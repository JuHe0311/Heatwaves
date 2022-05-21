# data i/o
import xarray
import argparse
# the usual
import numpy as np
import pandas as pd
import deepgraph as dg
#import plotting as plot
import con_sep as cs
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

### Functions ###

# calculate the seasonal variables of the heatwaves in one family
def seasonal_measures(data,ndvi):
    feature_funcs = {'magnitude':[np.sum]}
    g = dg.DeepGraph(data)
    fgv = g.partition_nodes(['g_id'], feature_funcs=feature_funcs)
    fgv.reset_index(inplace=True)
    # merge ndvi and temperature dataset on g_id
    total = pd.merge(fgv,ndvi, on=['g_id'],how='inner')
    return total

# perform the correlation between two variables
def correlate(data_1,data_2):
    return stats.spearmanr(data_1,data_2,axis=0, nan_policy='omit')

def correlation(gv):
    n_nodes_corr = pd.DataFrame(columns=['year','cluster','corr','p_value'])
    hwmid_corr = pd.DataFrame(columns=['year','cluster','corr','p_value'])
    upgma_clust = list(gv.F_upgma.unique())
    years = list(gv.year.unique())
    # for every cluster and every year we perform the correlation individually
    for clust in upgma_clust:
        for y in years:
            g = dg.DeepGraph(gv)
            # only keep the values from the current cluster
            g.filter_by_values_v('F_upgma',clust)
            # only keep the values from the current year
            g.filter_by_values_v('year',y)
            ndvig = dg.DeepGraph(ndvi)
            ndvig.filter_by_values_v('year',y)
            # corr_matrix: g_id - ndvi - n_nodes - hwmid_sum
            corr_matrix = seasonal_measures(g.v,ndvig.v)
            corr1,p_value1 = correlate(corr_matrix.n_nodes,corr_matrix.ndvi)
            corr2,p_value2 = correlate(corr_matrix.magnitude_sum,corr_matrix.ndvi)
            df1 = {'year': y, 'cluster': clust, 'corr': corr1,'p_value':p_value1}
            n_nodes_corr = n_nodes_corr.append(df1, ignore_index = True)
            df2 = {'year': y, 'cluster': clust, 'corr': corr2,'p_value':p_value2}
            hwmid_corr = hwmid_corr.append(df2, ignore_index = True)
    return(n_nodes_corr,hwmid_corr)

def significance(nnodes,hwmid):
    nnodes['significant'] = np.where(nnodes.p_value < 0.01, 1,0)
    hwmid['significant'] = np.where(hwmid.p_value < 0.01, 1,0)
    # remove non-significant values
    nnodes.drop(nnodes.loc[nnodes['significant']==0].index,inplace=True)
    hwmid.drop(hwmid.loc[hwmid['significant']==0].index,inplace=True)
    return(nnodes,hwmid)


def plot_hits(fgv,v,plot_title):

    # for easy filtering, we create a new DeepGraph instance for 
    # each component
    gt = dg.DeepGraph(fgv)

    # configure map projection
    kwds_basemap = {'llcrnrlon': v.longitude.min() - 1,
                    'urcrnrlon': v.longitude.max() + 1,
                    'llcrnrlat': v.latitude.min() - 1,
                    'urcrnrlat': v.latitude.max() + 1}

    # configure scatter plots
    kwds_scatter = {'s': 1,
                        'c': gt.v.n_nodes.values,
                        'cmap': 'viridis_r',
                        'edgecolors': 'none'}

    # create scatter plot on map
    obj = gt.plot_map(lat='latitude', lon='longitude',kwds_basemap=kwds_basemap, kwds_scatter=kwds_scatter)

    # configure plots
    obj['m'].drawcoastlines(linewidth=.8)
    obj['m'].drawparallels(range(-50, 50, 20), linewidth=.2)
    obj['m'].drawmeridians(range(0, 360, 20), linewidth=.2)
    cb = obj['fig'].colorbar(obj['pc'], fraction=.022, pad=.02)
    cb.set_label('Number of Heatwave Days', fontsize=15) 
    obj['ax'].set_title('%s' % (plot_title))
    
    obj['fig'].savefig('../../Results/fam1/Heatwavedays_indiv_corr_%s.png' % plot_title,
                       dpi=300, bbox_inches='tight')

    
# plots heat wave families or clusters on a map        
def plot_families(fgv,v,plot_title):
    # for easy filtering, we create a new DeepGraph instance for 
    # each component
    gt = dg.DeepGraph(fgv)

    # configure map projection
    kwds_basemap = {'llcrnrlon': v.longitude.min() - 1,
                    'urcrnrlon': v.longitude.max() + 1,
                    'llcrnrlat': v.latitude.min() - 1,
                    'urcrnrlat': v.latitude.max() + 1}

    # configure scatter plots
    kwds_scatter = {'s': 1,
                        'c': gt.v.n_cp_nodes.values,
                        'cmap': 'viridis_r',
                        'edgecolors': 'none'}

    # create scatter plot on map
    obj = gt.plot_map(lat='latitude', lon='longitude',kwds_basemap=kwds_basemap, kwds_scatter=kwds_scatter)

    # configure plots
    obj['m'].drawcoastlines(linewidth=.8)
    obj['m'].drawparallels(range(-50, 50, 20), linewidth=.2)
    obj['m'].drawmeridians(range(0, 360, 20), linewidth=.2)
    cb = obj['fig'].colorbar(obj['pc'], fraction=.022, pad=.02)
    cb.set_label('Number of Heatwaves', fontsize=15) 
    obj['ax'].set_title('%s' % (plot_title))
    
    obj['fig'].savefig('../../Results/fam1/indiv_corr_%s.png' % plot_title,
                       dpi=300, bbox_inches='tight')


### Argparser ###

def make_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n_nodes", help="Give the path to the original dataset to be worked on.",
                        type=str)
    parser.add_argument("-hwmid", "--hwmid", help="Give the path to the original dataset to be worked on.",
                        type=str)
    parser.add_argument("-ndvi", "--ndvi", help="Give the path to the original dataset to be worked on.",
                        type=str)
    parser.add_argument("-d", "--data", help="Give the path to the original dataset to be worked on.",
                        type=str)
    return parser

parser = make_argparser()
args = parser.parse_args()
gv = pd.read_csv(args.data)
n_nodes = pd.read_csv(args.n_nodes)
hwmid = pd.read_csv(args.hwmid)
ndvi = pd.read_csv(args.ndvi)
gv['time']=pd.to_datetime(gv['time'])


#########################################################
# Plot the 10 most highly correlated heat waves

# add absolute value column
n_nodes['absolute_values'] = np.absolute(n_nodes['corr'])
hwmid['absolute_values'] = np.absolute(hwmid['corr'])

n_nodes.sort_values(by='absolute_values',axis=0,inplace=True,ascending=False)
hwmid.sort_values(by='absolute_values',axis=0,inplace=True,ascending=False)
print(n_nodes)
print(hwmid)
# plot the heat waves with the 10 highest correlation values for heat wave days
for i in range(10):
    year_nodes = n_nodes.year.iloc[i]
    clust_nodes = n_nodes.cluster.iloc[i]
    gv_tmp = gv[gv.time.dt.year == year_nodes]
    gv_tmp = gv_tmp[gv_tmp.F_upgma == clust_nodes]
    # feature funcs
    def n_cp_nodes(cp):
        return len(cp.unique())
    feature_funcs = {'magnitude': [np.sum],
                     'latitude': np.min,
                     'longitude': np.min,
                     'cp': n_cp_nodes}
    k_means_dg = dg.DeepGraph(gv_tmp)

    # create family-g_id intersection graph
    fgv = k_means_dg.partition_nodes(['F_upgma', 'g_id'], feature_funcs=feature_funcs)
    fgv.rename(columns={'cp_n_cp_nodes': 'n_cp_nodes', 'longitude_amin':'longitude','latitude_amin':'latitude'}, inplace=True)
    plot_families(fgv,gv,'Number Heat Wave Days Correlation: %s' % n_nodes['corr'].iloc[i])
    plot_hits(fgv,gv,'Number Heat Wave Days Correlation: %s' % n_nodes['corr'].iloc[i])

# plot the heat waves with the 10 highest correlation values for heat wave magnitude
for i in range(10):
    year_nodes = hwmid.year.iloc[i]
    clust_nodes = hwmid.cluster.iloc[i]
    gv_tmp = gv[gv.time.dt.year == year_nodes]
    gv_tmp = gv_tmp[gv_tmp.F_upgma == clust_nodes]
    # feature funcs
    def n_cp_nodes(cp):
        return len(cp.unique())
    feature_funcs = {'magnitude': [np.sum],
                     'latitude': np.min,
                     'longitude': np.min,
                     'cp': n_cp_nodes}
    k_means_dg = dg.DeepGraph(gv_tmp)

    # create family-g_id intersection graph
    fgv = k_means_dg.partition_nodes(['F_upgma', 'g_id'], feature_funcs=feature_funcs)
    fgv.rename(columns={'cp_n_cp_nodes': 'n_cp_nodes', 'longitude_amin':'longitude','latitude_amin':'latitude'}, inplace=True)
    plot_families(fgv,gv,'HWMId Correlation: %s' % hwmid['corr'].iloc[i])
    plot_hits(fgv,gv,'HWMId Correlation: %s' % hwmid['corr'].iloc[i])


#########################################################
# Split the dataset into three and compare correlation

y1 = np.arange(1981,1993)
y2 = np.arange(1993,2005)
y3 = np.arange(2005,2015)

fam1_gv1 = gv[gv.time.dt.year.isin(y1)]
fam1_gv2 = gv[gv.time.dt.year.isin(y2)]
fam1_gv3 = gv[gv.time.dt.year.isin(y3)]
fam1_gv1['year'] = fam1_gv1.time.dt.year
fam1_gv2['year'] = fam1_gv2.time.dt.year
fam1_gv3['year'] = fam1_gv3.time.dt.year

n_nodes_corr1,hwmid_corr1 = correlation(fam1_gv1)
n_nodes_corr2,hwmid_corr2 = correlation(fam1_gv2)
n_nodes_corr3,hwmid_corr3 = correlation(fam1_gv3)

n_nodes_corr1,hwmid_corr1 = significance(n_nodes_corr1,hwmid_corr1)
n_nodes_corr2,hwmid_corr2 = significance(n_nodes_corr2,hwmid_corr2)
n_nodes_corr3,hwmid_corr3 = significance(n_nodes_corr3,hwmid_corr3)

# plot
hwmids = pd.DataFrame({'1981-1992': hwmid_corr1['corr'], '1993-2004': hwmid_corr2['corr'], '2005-2015': hwmid_corr3['corr']})
ax = sns.boxplot(data=hwmids)
ax = sns.swarmplot(data=hwmids,color=".25",size=3)
plt.clf()
fig = ax.get_figure()
fig.savefig("../../Results/hwmid_boxplot.png") 

nnodes = pd.DataFrame({'1981-1992': n_nodes_corr1['corr'], '1993-2004': n_nodes_corr2['corr'], '2005-2015': n_nodes_corr3['corr']})
ax = sns.boxplot(data=nnodes)
ax = sns.swarmplot(data=nnodes,color=".25",size=3)
plt.clf()
fig = ax.get_figure()
fig.savefig("../../Results/nnodes_boxplot.png") 

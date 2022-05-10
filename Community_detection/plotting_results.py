# takes net_cdf file and converts it into a pandas dataframe with xarray
# creates integer based coordinates
# saves pandas dataframe under Results

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
### Argparser ###

def make_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", help="Give the path to the original dataset to be worked on.",
                        type=str)
    return parser

parser = make_argparser()
args = parser.parse_args()
gv = pd.read_csv(args.data)
gv['time']=pd.to_datetime(gv['time'])
g = dg.DeepGraph(gv)

# create supernodes from deep graph by partitioning the nodes by cp
# feature functions applied to the supernodes
feature_funcs = {'time': [np.min, np.max],
                 't2m': [np.mean],
                 'magnitude': [np.sum],
                 'latitude': [np.mean],
                 'longitude': [np.mean], 
                 't2m': [np.max],'ytime':[np.mean],}
# partition graph
cpv, ggv = g.partition_nodes('cp', feature_funcs, return_gv=True)
# append neccessary stuff
# append geographical id sets
cpv['g_ids'] = ggv['g_id'].apply(set)
# append cardinality of g_id sets
cpv['n_unique_g_ids'] = cpv['g_ids'].apply(len)
# append time spans
cpv['dt'] = cpv['time_amax'] - cpv['time_amin']
# append time spans
cpv['timespan'] = cpv.dt.dt.days+1
# not neccessary for precipitation
cpv.rename(columns={'magnitude_sum': 'HWMId_magnitude'}, inplace=True)

# plot seaborn pairplot
cpv['n_nodes_log'] = np.log(cpv.n_nodes)
cpv['timespan_log'] = np.log(cpv.timespan)
cpv['HWMId_magnitude_log'] = np.log(cpv.HWMId_magnitude)
sns.pairplot(cpv, x_vars=['n_nodes','HWMId_magnitude', 'timespan', 'ytime_mean'], y_vars=['n_nodes','HWMId_magnitude', 'timespan', 'ytime_mean'], diag_kind="kde");
plt.savefig('../../Results/pairplot_cpv.png')
sns.pairplot(cpv, x_vars=['n_nodes_log','HWMId_magnitude_log', 'timespan_log', 'ytime_mean'], y_vars=['n_nodes_log','HWMId_magnitude_log', 'timespan_log', 'ytime_mean'], diag_kind="kde");
plt.savefig('../../Results/pairplot_cpv_log.png')


# plot largest heat wave
first = gv[gv.cp == 10]
first_gv = dg.DeepGraph(first)

# feature funcs
def n_cp_nodes(cp):
    return len(cp.unique())

feature_funcs = {'magnitude': [np.sum],
                 'latitude': np.min,
                 'longitude': np.min,
                 'cp': n_cp_nodes}

# create g_id intersection graph
fgv = first_gv.partition_nodes('g_id', feature_funcs=feature_funcs)
fgv.rename(columns={'cp_n_cp_nodes': 'n_cp_nodes', 'longitude_amin':'longitude','latitude_amin':'latitude'}, inplace=True)
fgv_v = dg.DeepGraph(fgv)
# configure map projection
kwds_basemap = {'llcrnrlon': g.v.longitude.min() - 1,
                    'urcrnrlon': g.v.longitude.max() + 1,
                    'llcrnrlat': g.v.latitude.min() - 1,
                    'urcrnrlat': g.v.latitude.max() + 1}
    
    # configure scatter plots
kwds_scatter = {'s': 1,
                    'c': fgv.n_nodes,
                    'cmap': 'viridis_r',
                    'alpha': .5,
                    'edgecolors': 'none'}

    # create scatter plot on map
obj = fgv_v.plot_map(lon='longitude', lat='latitude',
                      kwds_basemap=kwds_basemap,
                      kwds_scatter=kwds_scatter)

    # configure plots
obj['m'].drawcoastlines(linewidth=.8,zorder=10)
obj['m'].drawparallels(range(-50, 50, 20), linewidth=.2)
obj['m'].drawmeridians(range(0, 360, 20), linewidth=.2)
obj['ax'].set_title('Largest Heat Wave')
    
    # colorbar
cb = obj['fig'].colorbar(obj['pc'], fraction=.022, pad=.02)
cb.set_label('{}'.format('Number of Heat Wave Days'), fontsize=15) 
    
obj['fig'].savefig('../../Results/largest_heatwave.png',
                       dpi=300, bbox_inches='tight')


# plot progression of largest heat wave

#one = fam3[fam3.cp ==248]
times = np.arange(first.itime.min(), first.itime.max()+1)
tdic = {time: itime for itime, time in enumerate(times)}
first['dai'] = first.itime.apply(lambda x: tdic[x])
first['dai'] = first['dai'].astype(np.uint16)

# configure map projection
kwds_basemap = {'llcrnrlon': g.v.longitude.min() - 1,
                    'urcrnrlon': g.v.longitude.max() + 1,
                    'llcrnrlat': g.v.latitude.min() - 1,
                    'urcrnrlat': g.v.latitude.max() + 1}
    
ggg = dg.DeepGraph(first)
# configure scatter plots
kwds_scatter = {'s': 1,
                    'c': first.dai.values,
                    'cmap': 'rainbow',
                    'alpha': .4,
                    'edgecolors': 'none'}

# create scatter plot on map
obj = ggg.plot_map(lon='longitude', lat='latitude',
                      kwds_basemap=kwds_basemap,
                      kwds_scatter=kwds_scatter)

# configure plots
obj['m'].drawcoastlines(linewidth=.8)
obj['m'].drawparallels(range(-50, 50, 20), linewidth=.2)
obj['m'].drawmeridians(range(0, 360, 20), linewidth=.2)
obj['ax'].set_title('Progression of 2010 Heat Wave')
    
# colorbar
cb = obj['fig'].colorbar(obj['pc'], fraction=.022, pad=.02)
cb.set_label('{}'.format('Days After Initiation'), fontsize=15) 
obj['fig'].savefig('../../Results/largest_heatwave_progression.png',
                       dpi=300, bbox_inches='tight')

# Imports:
import xarray
# the usual
import numpy as np
import deepgraph as dg
import pandas as pd
import argparse
import plotting as plot

############### Functions #################

# hard boundaries for each cluster
def thresh(data,q):
    my_dict = {}
    upgma_clust = list(data.F_upgma.unique())
    for el in upgma_clust:
        data_tmp = data[data.F_upgma==el]
        g = dg.DeepGraph(data_tmp)
        fgv = g.partition_nodes(['g_id'])
        fgv.reset_index(inplace=True)
        tmp = fgv[fgv.n_nodes < q]
        to_delete = tmp.g_id.tolist()
        my_dict[el]= to_delete
    return my_dict

def filter_grids(data,my_dict):
    for item in my_dict.items():
        indexNames = data[(data['F_upgma'] == item[0]) & (data['g_id'].isin(item[1]))].index
        data.drop(indexNames,inplace=True)
    return data


def plot_families(families,fgv,v,plot_title):
  for F in families:

    # for easy filtering, we create a new DeepGraph instance for 
    # each component
    gt = dg.DeepGraph(fgv.loc[F])

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
    cb.set_label('number of Heatwaves', fontsize=15) 
    obj['ax'].set_title('Cluster {}'.format(F))
    
    obj['fig'].savefig('../../Results/%s_fam_%s.png' % (plot_title,F),
                       dpi=300, bbox_inches='tight')

############### Argparser #################

def make_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", help="Give the path to the dataset to be worked on.",
                       type=str)
    parser.add_argument("-o", "--ocean_clusters", nargs='*', 
                        help="Give a list containing the numbers of clusters that are to be removed because they are over the ocean",
                        type=int)
    parser.add_argument("-lsm", "--land_sea_mask", help="Give the path to the land sea mask.",
                        type=str)
    parser.add_argument("-b", "--boundary", help="Give the minimum number of hits of a heatwave for the hard boundary for every cluster.",
                        type=int)

    return parser

parser = make_argparser()
args = parser.parse_args()
gv_0 = pd.read_csv(args.data)
gv_0['time']=pd.to_datetime(gv_0['time'])
ocean_clust = args.ocean_clusters
lsm = xarray.open_dataset(args.land_sea_mask)
q = args.boundary

#create integer based (x,y) coordinates
lsm['x'] = (('longitude'), np.arange(len(lsm.longitude)))
lsm['y'] = (('latitude'), np.arange(len(lsm.latitude)))
#convert to dataframe
vt = lsm.to_dataframe()
#reset index
vt.reset_index(inplace=True)

# remove all clusters that are over the ocean from the dataset
gv_0 = gv_0[~gv_0.F_upgma.isin(ocean_clust)]

# remove all individual datapoints that are over the ocean
total = pd.merge(gv_0,vt, on=["x", 'y'], how='inner')
total = total[(total.lsm >=0.5)]

# define hard boundaries for each cluster
#total_thresh = thresh(total)
kmeans_filt = filter_grids(total,thresh(total,q))

kmeans_filt.to_csv(path_or_buf = "../../Results/kmeans_filtered.csv", index=False)

# feature funcs
def n_cp_nodes(cp):
    return len(cp.unique())

feature_funcs = {'magnitude': [np.sum],
                     'latitude_x': np.min,
                     'longitude_x': np.min,
                     'cp': n_cp_nodes}
k_means_dg = dg.DeepGraph(kmeans_filt)

# create family-g_id intersection graph
fgv = k_means_dg.partition_nodes(['F_upgma', 'g_id'], feature_funcs=feature_funcs)
fgv.rename(columns={'cp_n_cp_nodes': 'n_cp_nodes', 'longitude_x_amin':'longitude','latitude_x_amin':'latitude'}, inplace=True)
plot_families(list(kmeans_filt.F_upgma.unique()),fgv,vt,'filtered_clusters')

   

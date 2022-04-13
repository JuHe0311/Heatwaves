# Imports:
import xarray
# the usual
import numpy as np
import deepgraph as dg
import pandas as pd
import argparse

############### Functions #################

# hard boundaries for each cluster
def thresh(data,q):
    my_dict = {}
    upgma_clust = list(data.F_upgma.unique())
    for el in upgma_clust:
        data_tmp = data[data.F_upgma==el]
        g = dg.DeepGraph(data_tmp)
        fgv = g.partition_nodes(['g_id'])
        thresh = fgv.n_nodes.quantile(q)
        fgv.reset_index(inplace=True)
        tmp = fgv[fgv.n_nodes <= thresh]
        to_delete = tmp.g_id.tolist()
        my_dict[el]= to_delete
    return my_dict

def filter_grids(data,my_dict):
    for item in my_dict.items():
        indexNames = data[(data['F_upgma'] == item[0]) & (data['g_id'].isin(item[1]))].index
        data.drop(indexNames,inplace=True)
    return data

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
    parser.add_argument("-q", "--quantile", help="Give the quantile for the hard boundary for every cluster.",
                        type=int)

    return parser

parser = make_argparser()
args = parser.parse_args()
gv_0 = pd.read_csv(args.data)
gv_0['time']=pd.to_datetime(gv_0['time'])
ocean_clust = args.ocean_clusters
lsm = xarray.open_dataset(args.land_sea_mask)
q = args.quantile

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

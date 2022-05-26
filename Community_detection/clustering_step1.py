### Imports ###
import xarray
import argparse
# the usual
import numpy as np
import pandas as pd
import deepgraph as dg
import plotting as plot
import con_sep as cs
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.cluster import KMeans
import matplotlib.cm as cm

### functions ###

def conv_sin(doy):
    sin_doy = np.sin((doy/(365) * 2*np.pi))
    return sin_doy

def conv_cos(doy):
    cos_doy=np.cos((doy *2* np.pi / 365))
    return cos_doy

# plots heat wave families or clusters on a map        
def plot_families(number_families,fgv,v,plot_title):
  families = np.arange(number_families)
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
    cb.set_label('Number of Heatwaves', fontsize=15) 
    obj['ax'].set_title('Family %s' % F)
    
    obj['fig'].savefig('../../Results/%s_Cluster %s.png' % (plot_title,F),
                       dpi=300, bbox_inches='tight')

### Argparser ###

def make_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", help="Give the path to the original dataset to be worked on.",
                        type=str)
    parser.add_argument("-k", "--cluster_number", help="Give the number of clusters for the k-means clustering",
                        type=int)
    return parser

parser = make_argparser()
args = parser.parse_args()
gv = pd.read_csv(args.data)
k = args.cluster_number
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
cpv['timespan'] = cpv.dt.dt.days+1
# rename magnitude_sum column
cpv.rename(columns={'magnitude_sum': 'HWMId_magnitude'}, inplace=True)

# transform day of year value
cpv['doy_cos'] = cpv.ytime_mean.apply(conv_cos)
cpv['doy_sin'] = cpv.ytime_mean.apply(conv_sin)

# perform k means clustering
clusterer = KMeans(n_clusters=k, random_state=100)
cluster_labels = clusterer.fit_predict(cpv[['doy_sin','doy_cos']])
cpv['kmeans_clust'] = cluster_labels

# plot the k means clustering
fig, ax = plt.subplots()
fig.set_size_inches(18, 7)
colors = cm.nipy_spectral(cluster_labels.astype(float) / k)
xs = cpv['doy_cos']
ys = cpv['doy_sin']
ax.scatter(xs,ys, marker=".", s=50, lw=0, alpha=0.7, c=colors, edgecolor="k")
ax.set_title('k=%s' % k)
ax.set_xlabel('doy_cos')
ax.set_ylabel('doy_sin')
fig.savefig('../../Results/k-means_clustering')

# create F_kmeans col
gv['F_kmeans'] = np.ones(len(gv), dtype=int) * -1
gcpv = cpv.groupby('kmeans_clust')
it = gcpv.apply(lambda x: x.index.values)

for F in range(len(it)):
    cp_index = g.v.cp.isin(it.iloc[F])
    g.v.loc[cp_index, 'F_kmeans'] = F


# plot the day of year distribution of the clusters
for f in range(k):
    tmp = dg.DeepGraph(g.v)
    tmp.filter_by_values_v('F_kmeans',f)
    plt.hist(tmp.v.ytime, bins=175, label = 'Family %s' % f, alpha=0.5)
    plt.title("Day of Year Distribution of the Heat Wave Families")
    plt.xlabel('Day of year')
    plt.ylabel('Occurences')
    plt.legend()
plt.savefig('../../Results/day_of_year_distribution')

# plot the families on a map
# feature funcs
def n_cp_nodes(cp):
    return len(cp.unique())

feature_funcs = {'magnitude': [np.sum],
                 'latitude': np.min,
                 'longitude': np.min,
                 'cp': n_cp_nodes}

# create family-g_id intersection graph
fgv = g.partition_nodes(['F_kmeans', 'g_id'], feature_funcs=feature_funcs)
fgv.rename(columns={'cp_n_cp_nodes': 'n_cp_nodes', 'longitude_amin':'longitude','latitude_amin':'latitude'}, inplace=True)
plot_families(k,fgv,gv,'Family %s' % k)

# UPGMA clustering - dendrograms
# performed for every family individually - dendrogram is saved to find out how many clusters per family are optimal
for i in range(k):
    gvv = dg.DeepGraph(gv)
    gvv.filter_by_values_v('F_kmeans',i)
    gv_1 = gvv.v
    cpv_1 = cpv[cpv.kmeans_clust.eq(i)]
    # initiate DeepGraph
    cpg = dg.DeepGraph(cpv_1)

    # create edges
    cpg.create_edges(connectors=[cs.cp_node_intersection, 
                             cs.cp_intersection_strength],
                             no_transfer_rs=['intsec_card'],
                             logfile='create_cpe',
                             step_size=1e7)

    # create condensed distance matrix
    dv = 1 - cpg.e.intsec_strength.values
    #del cpg.e

    # create linkage matrix
    lm = linkage(dv, method='average', metric='euclidean')
    del dv
    # calculate full dendrogram
    plt.figure(figsize=(60, 40))
    plt.title('UPGMA Heatwave Clustering')
    plt.xlabel('heatwave index')
    plt.ylabel('distance')
    dendrogram(
        lm,
        leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=8.,  # font size for the x axis labels
    )
    plt.savefig('../../Results/dendrogram_fam%s.png' % i)
    
    # save the new datasets with the F_kmeans column
    cpv_1.to_csv(path_or_buf = "../../Results/cpv_fam%s.csv" % i, index=False)
    gv_1.to_csv(path_or_buf = "../../Results/gv_fam%s.csv" % i, index=False)

    




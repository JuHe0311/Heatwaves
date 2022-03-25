# takes net_cdf file and converts it into a pandas dataframe with xarray
# creates integer based coordinates
# saves pandas dataframe under Results

# data i/o
import xarray
import argparse
# the usual
import numpy as np
import pandas as pd
import matplotlib
import deepgraph as dg
import cppv
import gc
import plotting as plot
import con_sep as cs
# for plots
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
# kmeans clustering 
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm

### functions ###

def conv_sin(doy):
    sin_doy = np.sin((doy/(365) * 2*np.pi))
    return sin_doy

def conv_cos(doy):
    cos_doy=np.cos((doy *2* np.pi / 365))
    return cos_doy


### Argparser ###

def make_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", help="Give the path to the original dataset to be worked on.",
                        type=str)
    parser.add_argument("-lsm", "--land_sea_mask", help="Give the path to the land sea mask dataset.",
                        type=str)
    parser.add_argument("-k", "--cluster_number", help="Give the number of clusters for the k-means clustering",
                        type=int)
    parser.add_argument("-u", "--upgma_clusters", nargs='*', help="Give a list containing the number of upgma clusters",
                        type=int)
    return parser

parser = make_argparser()
args = parser.parse_args()
lsm = xarray.open_dataset(args.land_sea_mask)
gv = pd.read_csv(args.data)
k = args.cluster_number
no_clusters = args.upgma_clusters
gv['time']=pd.to_datetime(gv['time'])
g = dg.DeepGraph(gv)
# create supernodes from deep graph by partitioning the nodes by cp
# feature functions applied to the supernodes
feature_funcs = {'time': [np.min, np.max],
                 'itime': [np.min, np.max],
                 't2m': [np.mean],
                 'magnitude': [np.sum],
                 'latitude': [np.mean],
                 'longitude': [np.mean], 't2m': [np.max],'ytime':[np.mean,np.min,np.max],}
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

# transform data

cpv['n_nodes_log'] = np.log(cpv.n_nodes)
cpv['n_unique_g_ids_log'] = np.log(cpv.n_unique_g_ids)
cpv['magnitude_log'] = np.log(cpv.HWMId_magnitude)
cpv.magnitude_log[np.isneginf(cpv.magnitude_log)]=0
cpv['timespan_log'] = np.log(cpv.timespan)
cpv['n_nodes_std']=(cpv.n_nodes_log-min(cpv.n_nodes_log))/(max(cpv.n_nodes_log)-min(cpv.n_nodes_log))
cpv['n_unique_g_ids_std']=(cpv.n_unique_g_ids_log-min(cpv.n_unique_g_ids_log))/(max(cpv.n_unique_g_ids_log)-min(cpv.n_unique_g_ids_log))
cpv['magnitude_std']=(cpv.magnitude_log-min(cpv.magnitude_log))/(max(cpv.magnitude_log)-min(cpv.magnitude_log))
cpv['timespan_std']=(cpv.timespan_log-min(cpv.timespan_log))/(max(cpv.timespan_log)-min(cpv.timespan_log))
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

# create F col
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
    plt.hist(tmp.v.ytime, bins=175, label = 'cluster %s' % f, alpha=0.5)
    plt.title("Day of year distribution of all clusters")
    plt.xlabel('Day of year')
    plt.ylabel('Occurences')
    plt.legend()
plt.savefig('../../Results/day_of_year_distribution')


# plot the clusters on a map
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

plot.plot_families(k,fgv,gv,'heatwave_cluster %s' % k)

# UPGMA clustering

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
    # form flat clusters and append their labels to cpv
    cpv_1['F_t'] = fcluster(lm, no_clusters[i], criterion='maxclust')
    #del lm

    # relabel families by size
    f = cpv_1['F_t'].value_counts().index.values
    fdic = {j: i for i, j in enumerate(f)}
    cpv_1['F_t'] = cpv_1['F_t'].apply(lambda x: fdic[x])
    # create F col
    gv_1['F_upgma'] = np.ones(len(gv_1), dtype=int) * -1
    gcpv = cpv_1.groupby('F_t')
    it = gcpv.apply(lambda x: x.index.values)

    for F in range(len(it)):
        cp_index = gvv.v.cp.isin(it.iloc[F])
        gvv.v.loc[cp_index, 'F_upgma'] = F
    
    
    # feature funcs
    def n_cp_nodes(cp):
        return len(cp.unique())

    feature_funcs = {'magnitude': [np.sum],
                     'latitude': np.min,
                     'longitude': np.min,
                     'cp': n_cp_nodes}

    # create family-g_id intersection graph
    fgv = gvv.partition_nodes(['F_upgma', 'g_id'], feature_funcs=feature_funcs)
    fgv.rename(columns={'cp_n_cp_nodes': 'n_cp_nodes', 'longitude_amin':'longitude','latitude_amin':'latitude'}, inplace=True)
    cpv_1.to_csv(path_or_buf = "../../Results/cpv_fam%s.csv" % i, index=False)
    gv_1.to_csv(path_or_buf = "../../Results/gv_fam%s.csv" % i, index=False)
    r = range(no_clusters[i])
    plot.plot_families(no_clusters[i],fgv,gv,'upgma_clustering %s' % i)

    




# Imports:
import os
import xarray
# for plots
import matplotlib.pyplot as plt
# the usual
import numpy as np
import deepgraph as dg
import pandas as pd
import itertools
import scipy
import argparse
import sklearn
from sklearn.model_selection import ShuffleSplit
import igraph as ig
import seaborn as sns
from scipy.cluster.hierarchy import linkage, fcluster
import cairo
import plotting as pt
import cppv as cp

############### Functions ################

# getting all nodes and supernodes of indices of one cluster (indices are given to the function by a list)
def get_clustnodes(a_list):
    # nodes
    nodes = pd.DataFrame()
    for i in a_list:
        for j in range(len(g_temp.v)):
            if g_temp.v.cp.iloc[j] == i:
                nodes = nodes.append(g_temp.v.iloc[j])
            
    # getting all supernodes   
    supernodes = pd.DataFrame()
    for i in a_list:
        supernodes = supernodes.append(cpv.loc[i])
    return nodes, supernodes


############### Argparser ################

def make_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", help="Give the path to the dataframe from which the graph should be created.",
                        type=str)
    parser.add_argument("-de", "--data_extreme", help="Give the path to the extreme value dataset.",
                        type=str)
    parser.add_argument("-do", "--data_original", help="Give the path to the original value dataset.",
                        type=str)
    return parser

parser = make_argparser()
args = parser.parse_args()

c_e_2 = pd.read_csv(args.data)
extr = pd.read_csv(args.data_extreme)
vt = pd.read_csv(args.data_original)
g,cpv = cp.create_cpv(extr,vt)

# create a weighted graph
graph = ig.Graph.TupleList(c_e_2.values, 
                       weights=True, directed=False)
graph.vs["label"] = graph.vs["name"]

# next to do: delete all edges with weight zero!
graph.es.select(weight=0).delete() 
dendrogram_multi = graph.community_multilevel(weights=graph.es['weight'])
# save plot somehow
ig.plot(dendrogram_multi, "../../Results/dendrogram_multi.png")

# creates a dictionary of all clusters with the correct cp names of the heatwaves
cluster_list = list(dendrogram_multi)
clust = []
cluster_dict = {}
for i in range(len(cluster_list)):
    for j in cluster_list[i]:
        clust.append(graph.vs[j]["name"])
    cluster_dict[i] = clust
    clust = []
print(cluster_dict)
# create clustnodes

# deep graph that is sorted by cp value
g_temp = g
g_temp.v.sort_values(by=['cp'], inplace=True)

ccpv_multi1, ccpv_multi1_supernodes = get_clustnodes(cluster_dict[0])
ccpv_multi2, ccpv_multi2_supernodes = get_clustnodes(cluster_dict[1])
ccpv_multi3, ccpv_multi3_supernodes = get_clustnodes(cluster_dict[2])


# plot
plot_clusters(ccpv_multi1, 'n_heatwave_multistep cluster 1 weighted')
plot_clusters(ccpv_multi2, 'n_heatwave_multistep cluster 2 weighted')
plot_clusters(ccpv_multi3, 'n_heatwave_multistep cluster 3 weighted')
# save



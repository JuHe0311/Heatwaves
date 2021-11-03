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
import plotting as pt
import cppv as cp
import con_sep as cs
import cairo
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
    parser.add_argument("-de", "--data_extreme", help="Give the path to the extreme value dataset.",
                        type=str)
    parser.add_argument("-do", "--data_original", help="Give the path to the original value dataset.",
                         type=str)
    return parser

parser = make_argparser()
args = parser.parse_args()

extr = pd.read_csv(args.data_extreme)
vt = pd.read_csv(args.data_original)
g,gv,cpv = cp.create_cpv(extr)

# calculate threshold
# calculate 95th percentile as threshold from the Intersection strength between the supernodes
cpg = dg.DeepGraph(cpv)
# create edges
cpg.create_edges(connectors=[cs.cp_node_intersection, 
                             cs.cp_intersection_strength],
                 no_transfer_rs=['intsec_card'],
                 logfile='create_cpe',
                 step_size=1e7)
values = cpg.e['intsec_strength'].values
values = np.array(values)
threshold = np.percentile(values,95)
print(threshold)
cpg_temp = cpg.e.stack().reset_index()
cpg_temp = cpg_temp.drop(['level_2'], axis=1)
cpg_temp.columns = ['source','target','intsec_strenght']

# create a weighted graph
# weights of edges are the intersection strengths between two nodes
graph = ig.Graph.TupleList(cpg_temp.values, directed=False, weights=True)
graph.vs["label"] = graph.vs["name"]

# next to do: delete all edges that do not have the weight 1!
graph.es.select(weight_ne=1.0).delete()
dendrogram_multi = graph.community_multilevel(weights=graph.es['weight'])
# save plot somehow
ig.plot(dendrogram_multi, vertex_label_size=1, "../../Results/unweighted_dendrogram_multi.png")

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
# plot
for i in range(len(cluster_dict)):
    ccpv_multi,ccpv_multi_supernodes = get_clustnodes(cluster_dict[i])
    pt.plot_clusters(ccpv_multi, 'n_heatwave_multistep cluster %s unweighted' % i, vt)


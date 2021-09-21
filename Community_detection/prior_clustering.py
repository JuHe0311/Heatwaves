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
import seaborn as sns
from scipy.cluster.hierarchy import linkage, fcluster


############### Functions #################

# create superedges between the supernodes to find heatwave clusters with strong regional overlaps
# compute intersection of geographical locations
def cp_node_intersection(g_ids_s, g_ids_t):
    intsec = np.zeros(len(g_ids_s), dtype=int)
    intsec_card = np.zeros(len(g_ids_s), dtype=np.int)
    for i in range(len(g_ids_s)):
        intsec[i] = g_ids_s[i].intersection(g_ids_t[i])
        intsec_card[i] = len(intsec[i])
    return intsec_card

# compute a spatial overlap measure between clusters
def cp_intersection_strength(n_unique_g_ids_s, n_unique_g_ids_t, intsec_card):
    min_card = np.array(np.vstack((n_unique_g_ids_s, n_unique_g_ids_t)).min(axis=0), 
                        dtype=np.float64)
    intsec_strength = intsec_card / min_card
    return intsec_strength

# compute temporal distance between clusters
def time_dist(dtime_amin_s, dtime_amin_t):
    dt = dtime_amin_t - dtime_amin_s
    return dt


############### Argparser #################

def make_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", help="Give the path to the dataset to be worked on.",
                        type=str)
    return parser

parser = make_argparser()
args = parser.parse_args()
cpv = pd.read_csv(args.data, dtype={'g_ids':object})
cpv['g_ids'] = set(map(int, cpv['g_ids']))
#cpv['g_ids'] = cpv['g_ids'].apply(set)
print(cpv)
#print(cpv['g_ids'].dtype)
# create an array that counts the number of times two heatwaves are put in the same cluster
count_edges = np.zeros((cpv.index[-1],cpv.index[-1]))
count_edges.shape

# clustering is performed 100 times
# for every clustering edges between the nodes in the same cluster are created and the number of edges between two nodes are counted in the matrix count_edges
for i in range(100):
    # leave out random 1% of the data --> need to use cpv
    X = np.arange(len(cpv))
    ss = ShuffleSplit(n_splits=1, test_size=0.01)
    for train_index, test_index in ss.split(X):
        train = train_index
    ccpv = pd.DataFrame()
    for index in train:
        SR_Row = pd.Series(cpv.iloc[index])
        ccpv = ccpv.append(SR_Row)
    
    # ccpv is the dataset with the train set
    ccpg = dg.DeepGraph(ccpv)

    ccpg.create_edges(connectors=[cp_node_intersection, 
                                 cp_intersection_strength],
                     no_transfer_rs=['intsec_card'],
                     logfile='create_cpe',
                     step_size=1e7)
    
    # clustering step
    from scipy.cluster.hierarchy import linkage, fcluster

    # create condensed distance matrix
    dv = 1 - ccpg.e.intsec_strength.values
    del ccpg.e

    # create linkage matrix
    lm = linkage(dv, method='average', metric='euclidean')
    del dv

    # form flat clusters and append their labels to cpv
    ccpv['F'] = fcluster(lm, 1000, criterion='maxclust')
    del lm

    # relabel families by size
    f = ccpv['F'].value_counts().index.values
    fdic = {j: i for i, j in enumerate(f)}
    ccpv['F'] = ccpv['F'].apply(lambda x: fdic[x])
    
    # create edges between nodes in one family and add to the matrix count_edges
    ccpg.create_edges(connectors=same_fams, selectors=sel_fams)
    edges = ccpg.e
    edges.reset_index(inplace=True)
    edges.sort_values(by=['s','t'])

    # add edges to table that counts the number of edges between two nodes
    for i in range(len(edges)):
        s = edges.s.iloc[i]
        t = edges.t.iloc[i]
        count_edges[s-1][t-1] = count_edges[s-1][t-1] + 1
        count_edges[t-1][s-1] = count_edges[t-1][s-1] + 1

# reshape the count_edges matrix
# indices dictionary: keys are the original cp values of the respective heatwaves, values are the index numbers in the new array count_edges_2
indices = ccpv.index.values
indices.sort()
indices = { i : 0 for i in indices}
count_edges_2 = np.zeros((len(ccpv),len(ccpv)))
counter_i = 0
counter_j = 0
for i in range(len(count_edges)):
    if i+1 in indices:
        counter_j = 0
        for j in range(len(count_edges)):
            if j+1 in indices:
                count_edges_2[counter_i][counter_j] = count_edges[i][j]
                counter_j = counter_j + 1
        indices[i+1] = counter_i
        counter_i = counter_i + 1

# create dataframe from count_edges_2 and prepare it to create a graph from it
c_e_2 = pd.DataFrame(count_edges_2)
c_e_2 = c_e_2.stack().reset_index()
c_e_2.columns = ['source','target','number_of_edges']
c_e_2.sort_values(by=['number_of_edges'], inplace=True, ascending=False)

# replace source and target number with original cp numbers of the heatwaves
for i in range(len(c_e_2)):
    source = c_e_2.source.iloc[i]
    target = c_e_2.target.iloc[i]
    c_e_2.source.iloc[i] = list(indices.keys())[list(indices.values()).index(source)]
    c_e_2.target.iloc[i] = list(indices.keys())[list(indices.values()).index(target)]
print(c_e_2)

# save dataframe to subsequently create a graph and do community clustering on it
c_e_2.to_csv(path_or_buf = "../../Results/c_e.csv", index=False)

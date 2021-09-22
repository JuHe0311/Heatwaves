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

############### Argparser #############

def make_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", help="Give the path to the dataframe from which the graph should be created.",
                        type=str)
    return parser

parser = make_argparser()
args = parser.parse_args()

c_e_2 = pd.read_csv(args.data)

# create a weighted graph
method = # insert argument here
graph = ig.Graph.TupleList(c_e_2.values, 
                       weights=True, directed=False)
graph.vs["label"] = graph.vs["name"]

# next to do: delete all edges with weight zero!
graph.es.select(weight=0).delete() 
dendrogram_multi = graph.community_multilevel(weights=graph.es['weight'])
# save plot somehow
ig.plot(dendrogram_multi, "dendrogram_multi.png")

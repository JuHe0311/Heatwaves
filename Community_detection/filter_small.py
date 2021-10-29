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
import cppv
import con_sep as cs
import plotting as pt

############### Argparser #################

def make_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", help="Give the path to the dataset to be worked on.",
                        type=str)
    parser.add_argument("-do", "--original_data", help="Give the path to the original dataset to be worked on.",
                        type=str)
    return parser

parser = make_argparser()
args = parser.parse_args()
extr = pd.read_csv(args.data)
vt = pd.read_csv(args.original_data)
# sort by time
extr.sort_values('time', inplace=True)

g,cpg,cpv = cppv.create_cpv(extr)

print(cpv)

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
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import cppv
import con_sep as cs
import plotting as pt

############### Argparser #################

def make_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", help="Give the path to the dataset to be worked on.",
                        type=str)
    return parser

parser = make_argparser()
args = parser.parse_args()
gv = pd.read_csv(args.data)

def sel_family(number,dataset):
    fam = dataset.loc[dataset['F']==number]
    return fam

fam1 = sel_family(0,g.v)
fam2 = sel_family(1,g.v)
fam3 = sel_family(2,g.v)
fam4 = sel_family(3,g.v)
fam5 = sel_family(4,g.v)
fam6 = sel_family(5,g.v)


fam1['ytime'] = fam1.time.apply(lambda x: x.dayofyear)
fam2['ytime'] = fam2.time.apply(lambda x: x.dayofyear)
fam3['ytime'] = fam3.time.apply(lambda x: x.dayofyear)
fam4['ytime'] = fam4.time.apply(lambda x: x.dayofyear)
fam5['ytime'] = fam5.time.apply(lambda x: x.dayofyear)
fam6['ytime'] = fam6.time.apply(lambda x: x.dayofyear)

plt.hist(fam1.ytime, bins=20)
plt.title("Day of year distribution of family 1")
plt.xlabel('Day of year')
plt.ylabel('Occurences')
plt.savefig('../../Results/doy_dendrogram_fam1')

plt.hist(fam2.ytime, bins=20)
plt.title("Day of year distribution of family 2")
plt.xlabel('Day of year')
plt.ylabel('Occurences')
plt.savefig('../../Results/doy_dendrogram_fam2')

plt.hist(fam3.ytime, bins=20)
plt.title("Day of year distribution of family 3")
plt.xlabel('Day of year')
plt.ylabel('Occurences')
plt.savefig('../../Results/doy_dendrogram_fam3')

plt.hist(fam4.ytime, bins=20)
plt.title("Day of year distribution of family 4")
plt.xlabel('Day of year')
plt.ylabel('Occurences')
plt.savefig('../../Results/doy_dendrogram_fam4')

plt.hist(fam5.ytime, bins=20)
plt.title("Day of year distribution of family 5")
plt.xlabel('Day of year')
plt.ylabel('Occurences')
plt.savefig('../../Results/doy_dendrogram_fam5')


plt.hist(fam6.ytime, bins=20)
plt.title("Day of year distribution of family 6")
plt.xlabel('Day of year')
plt.ylabel('Occurences')
plt.savefig('../../Results/doy_dendrogram_fam6')

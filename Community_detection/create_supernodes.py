# takes net_cdf file and converts it into a pandas dataframe with xarray
# creates integer based coordinates
# saves pandas dataframe under Results

# data i/o
import xarray
import argparse
# the usual
import numpy as np
import pandas as pd
import extr as ex
import matplotlib
import deepgraph as dg
import cppv

### Argparser ###

def make_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", help="Give the path to the original dataset to be worked on.",
                        type=str)
    return parser

parser = make_argparser()
args = parser.parse_args()
extr = cpv.read_csv(args.data)


g,cpg,cpv = cppv.create_cpv(extr)

# save heatwaves
cpv.to_csv(path_or_buf = "../../Results/cpv.csv", index=False)
g.v.to_csv(path_or_buf = "../../Results/gv.csv", index=False)



# data i/o
import os
import xarray
import argparse
# for plots
import matplotlib.pyplot as plt

# the usual
import numpy as np
import pandas as pd
import sklearn
import extr as ex

### Argparser ###

def make_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-do", "--original_data", help="Give the path to the original dataset to be worked on.",
                        type=str)
    return parser

parser = make_argparser()
args = parser.parse_args()
d = xarray.open_dataset(args.original_data)

#create integer based (x,y) coordinates
d['x'] = (('longitude'), np.arange(len(d.longitude)))
d['y'] = (('latitude'), np.arange(len(d.latitude)))
#convert to dataframe
vt = d.to_dataframe()
#reset index
vt.reset_index(inplace=True)
vt.to_csv(path_or_buf = "../../Results/vt_raw.csv", index=False)

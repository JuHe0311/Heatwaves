# takes csv file (created with create_vt) and adds neccessary columns (days, months, years, ytime, itime) and removes the 366th day of the dataset 
# 366th day = 29th of february
# saves output as vt_366cut.csv

# data i/o
import xarray
import argparse
# the usual
import numpy as np
import pandas as pd
import extr as ex

### Argparser ###

def make_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", help="Give the path to the original dataset to be worked on.",
                        type=str)
    return parser

parser = make_argparser()
args = parser.parse_args()
vt = pd.read_csv(args.data)

# add correct times
datetimes = pd.to_datetime(vt['time'])

# remove 366th day (29th of february, every 4 years)
vt = ex.cut_366(vt)
vt.to_csv(path_or_buf = "../../Results/vt_366cut.csv", index=False)
# takes the vt dataset without the 366th day and converts the temperature from kelvin to degrees celcius
# calculates the thresholds and minmax for the hwmid
# saves thresholds, vt and minmax as csv files
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
    parser.add_argument("-do", "--original_data", help="Give the path to the original dataset to be worked on.",
                        type=str)
    return parser

parser = make_argparser()
args = parser.parse_args()
vt = pd.read_csv(args.original_data)

# add correct times
datetimes = pd.to_datetime(vt['time'])

# convert temperature from kelvin to degrees celcius
ex.conv_to_degreescelcius(vt)

# calculate thresholds
thresholds,minmax = ex.calc_thresh(vt)

# save thresholds and extreme dataset
thresholds.to_csv(path_or_buf = "../../Results/thresholds.csv", index=False)
vt.to_csv(path_or_buf = "../../Results/vt.csv", index=False)
minmax.to_csv(path_or_buf = "../../Results/minmax.csv", index=False)


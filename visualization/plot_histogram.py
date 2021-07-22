# creates a histogram of a property of a dataset. Needs a path to a dataframe and a property as a string to plot --> property needs to be the name of a column 
# from the dataframe

# Imports:
# for plots
import matplotlib.pyplot as plt
# the usual
import numpy as np
import pandas as pd
import argparse


############### Argparser #############


def make_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", help="Give the path to the dataset to visualize.",
                        type=str)
    parser.add_argument("-p", "--property",  help="State the property (column) of the dataset to be plotted"
                        type=str)
    return parser


parser = make_argparser()
args = parser.parse_args()

prop = args.property

data = pd.read_csv(args.data)  

plt.hist(data.prop)
plt.title("Distribution of german heatwaves 2010-2020 - "+prop)
plt.xlabel(prop)
plt.ylabel('Number of heatwaves')
save_path = '../../Results/histogram'+prop+'.png'
plt.savefig(save_path)

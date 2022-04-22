# Imports
import numpy as np
import pandas as pd
import sklearn
#### Methods ####

def calc_percentile(a_list):
    threshold = np.percentile(a_list,95)
    return threshold

def calc_perc(lst):
    a_list = []
    for l in lst:
        for i in range(len(l)):
            a_list.append(l[i])
    return calc_percentile(a_list)     
 
# function to convert temperature from kelvin to degrees celcius
def conv_to_degreescelcius(data):
    data.t2m = data.t2m - 273.15  




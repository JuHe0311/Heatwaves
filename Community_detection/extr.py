# data i/o
import os
import xarray

# for plots
import matplotlib.pyplot as plt

# the usual
import numpy as np
import pandas as pd
import sklearn
#### Methods ####

def calc_thresh(data):
    first = np.arange(350,366)
    second = np.arange(1,366)
    third = np.arange(1,16)
    time = np.concatenate((first, second, third), axis=None)
    count = 0
    x = np.arange(max(data.x)+1)
    y = np.arange(max(data.y)+1)
    for i in x:
        for j in y:
            tmp = data[(data.x == i) & (data.y == j)]
            for k in range(0,365):
                time_tmp = time[k:k+31]
                tmp2 = tmp[tmp.ytime.isin(time_tmp)]
                t2m_vals = tmp2.t2m
                thresh = calc_percentile(t2m_vals)
                y_tim = k+1
                thresholds.loc[count] = [tmp2.iloc[1].latitude, tmp2.iloc[1].longitude, y_tim, thresh]
                count=count+1
    return thresholds
  
def extr_events(data, thresholds):
    extr_dataset = pd.DataFrame(columns=['latitude', 'longitude', 'time', 't2m', 'x', 'y', 'ytime'])
    count = 0
    for i in range(len(data)):
        thresh = thresholds[(thresholds.longitude == data.loc[i].longitude) & 
                            (thresholds.latitude == data.loc[i].latitude) & (thresholds.ytime == data.loc[i].ytime)]
        if (thresh.threshold.values <= data.loc[i].t2m):
            extr_dataset.loc[count] = [data.loc[i].latitude, data.loc[i].longitude, 
                                                   data.loc[i].time, data.loc[i].t2m, 
                                                   data.loc[i].x, data.loc[i].y, data.loc[i].ytime]
            count = count + 1
    return extr_dataset
  
def calc_percentile(a_list):
    threshold = np.percentile(a_list,90)
    return threshold

# this function cuts the 366th day (29th of february) of years
# input:
# data: a dataframe (pandas) that contains a day and month column
# output is the changed dataframe
def cut_366(data):
    length = len(data)
    i = 0
    while i <= (length-1):
        if (data.day[i] == 29) & (data.month[i] == 2):
            data = data.drop(data.index[i])
            data = data.reset_index(drop=True)
            length = length -1
        i = i + 1
    return data
  
# function to convert temperature from kelvin to degrees celcius
def conv_to_degreescelcius(data):
    for i in range(len(data)):
        data.t2m[i] = data.t2m[i] - 273.15
  
# function calculates the 75th and 25th percentile of a specific day(in a month) from a set of values
# input:
# day, month: day and month for which the percentiles should be calculated
# x,y: specify the grid for which the percentiles are calculated: x = lon, y = lat
# data: data for which the percentiles should be calculated
def calc_75_25_percentile(day, month, x, y, data):
    a_list = list(data.t2m[(data.day == day) & (data.month == month) & (data.x == x) & (data.y == y)])
    threshold_seventyfive = np.percentile(a_list,75)
    threshold_twentyfive = np.percentile(a_list, 25)
    return threshold_seventyfive, threshold_twentyfive

# calculates the daily magnitude index of a day, appends the daily magnitude of a day to the dataframe
# input:
# data: data to calculate the 75th and 25th percentile for each day
# extr_data: dataframe that the daily magnitude index is calculated for
def daily_magnitude(data, extr_data):
    list_sf = []
    list_tf = []
    list_dm = []
    for i in range(len(extr_data)):
        sf, tf = calc_75_25_percentile(extr_data.day[i], extr_data.month[i], extr_data.x[i], extr_data.y[i], data)
        list_sf.append(sf)
        list_tf.append(tf)
    extr_data['seventyfive_percentile'] = list_sf
    extr_data['twentyfive_percentile'] = list_tf
    
    for i in range(len(extr_data)):
        if extr_data.t2m[i] > extr_data.twentyfive_percentile[i]:
            dm1 = extr_data.t2m[i] - extr_data.twentyfive_percentile[i]
            dm2 = extr_data.seventyfive_percentile[i] - extr_data.twentyfive_percentile[i]
            dm = float(dm1/dm2)
        else:
            dm = 0
        list_dm.append(dm)
    extr_data['daily_mag'] = list_dm
   

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
    thresholds = pd.DataFrame(columns=['latitude', 'longitude', 'ytime', 'threshold'])
    first = np.arange(350,366)
    second = np.arange(1,366)
    third = np.arange(1,16)
    time = np.concatenate((first, second, third), axis=None)
    minmax = pd.DataFrame(columns=['sf', 'tf', 'latitude', 'longitude'])
    count = 0
    count_hwmid = 0
    x = np.arange(max(data.x)+1)
    y = np.arange(max(data.y)+1)
    for i in x:
        for j in y:
            tmp = data[(data.x == i) & (data.y == j)]
            t2m_tmp = list(tmp.t2m)
            minmax.loc[count_hwmid] = [np.percentile(t2m_tmp,75),np.percentile(t2m_tmp,25),tmp.iloc[1].latitude, 
                                       tmp.iloc[1].longitude]
            count_hwmid = count_hwmid + 1
            for k in range(0,365):
                time_tmp = time[k:k+31]
                tmp2 = tmp[tmp.ytime.isin(time_tmp)]
                t2m_vals = tmp2.t2m
                thresh = calc_percentile(t2m_vals)
                y_tim = k+1
                thresholds.loc[count] = [tmp2.iloc[1].latitude, tmp2.iloc[1].longitude, y_tim, thresh]
                count=count+1
    return thresholds,minmax

def extr_events(data, thresholds,hwmid):
    extr_dataset = pd.DataFrame(columns=['latitude', 'longitude', 'time', 't2m', 'x', 'y', 'ytime','day', 'month', 'year'])
    count = 0
    list_dm = []
    for i in range(len(data)):
        thresh = thresholds[(thresholds.longitude == data.loc[i].longitude) & 
                            (thresholds.latitude == data.loc[i].latitude) & (thresholds.ytime == data.loc[i].ytime)]
        if (thresh.threshold.values <= data.loc[i].t2m):
            extr_dataset.loc[count] = [data.loc[i].latitude, data.loc[i].longitude, 
                                                   data.loc[i].time, data.loc[i].t2m, 
                                                   data.loc[i].x, data.loc[i].y, data.loc[i].ytime, data.loc[i].day,
                                      data.loc[i].month, data.loc[i].year]
            tmp = hwmid[(hwmid.longitude == extr_dataset.loc[count].longitude) & 
                      (hwmid.latitude == extr_dataset.loc[count].latitude)]
            sf = tmp.sf
            tf = tmp.tf
            if (extr_dataset.loc[count].t2m > tf.values):
                dm1 = extr_dataset.loc[count].t2m - tf.values
                dm2 = sf.values - tf.values
                dm = float(dm1/dm2)
            else:
                dm = 0
            count = count + 1
            list_dm.append(dm)
    extr_dataset['daily_mag'] = list_dm
    return extr_dataset
  
def calc_percentile(a_list):
    threshold = np.percentile(a_list,95)
    return threshold


def calc_perc(lst):
    a_list = []
    for l in lst:
        for i in range(len(l)):
            a_list.append(l[i])
    return calc_percentile(a_list) 
    
     
# this function cuts the 366th day (29th of february) of years
# input:
# data: a dataframe (pandas) that contains a day and month column
# output is the changed dataframe
def cut_366(data):
    data.drop(data.loc[(data['day']==29) & (data['month']==2)].index, inplace=True)
    return data

 
# function to convert temperature from kelvin to degrees celcius
def conv_to_degreescelcius(data):
    data.t2m = data.t2m - 273.15  




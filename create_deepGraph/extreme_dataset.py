# create dataset of extreme temperature events

# output: extreme dataset, original dataset
# Imports:
import os
import xarray
# for plots
import matplotlib.pyplot as plt
# the usual
import numpy as np
import pandas as pd
import itertools
import scipy
import argparse


############### Argparser #############


def make_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", help="Give the path to the dataset to be worked on.",
                        type=str)
    parser.add_argument("-s", "--save_path",  help="Give a path where to save the extreme output dataset",
                        type=str)
    parser.add_argument("-sy", "--startyear",  help="Give a start year to be analyzed."
                        "\nPossible years are to be within 1980-2020", default=1980, type=int)
    parser.add_argument("-ey", "--endyear",  help="Give an end year to be analyzed."
                        "\nPossible years are to be within 1980-2020", default=2020, type=int)  
    return parser


parser = make_argparser()
args = parser.parse_args()

data_path = args.data
save_path = args.save_path
starty = args.startyear
endy = args.endyear


############### Methods ###############

# calculates the 90th percentile of a list of values
# input:
# a list of numbers
# output: 
# a threshold that marks the 90th percentile

def calc_percentile(a_list):
    threshold = np.percentile(a_list,90)
    return threshold

# appends the temperature values for one year 15 days before and after the given day i into a list and returns this
# list
# input:
# i: specification of a day (by index), integer between 0 and len(data frame)
# one_grid: data frame of one grid cell
# year_begin: the year of the beginning of the dataframe (integer)
# year_end: the year of the end of the dataframe (integer)

def one_year(i,one_grid, year_begin, year_end):
    if (one_grid.day[i] <= 15) & (one_grid.year[i] == year_begin):
        temp_list = []
        for j in range(i-one_grid.day[i]+1,i+15):
            temp_list.append(one_grid.t2m[j])
    elif (one_grid.day[i] >= 15) & (one_grid.year[i] == year_end):
        temp_list = []
        for j in range(i-15,len(one_grid)-1):
            temp_list.append(one_grid.t2m[j])
    else:
        temp_list = []
        for j in range(i-15,i+15):
            temp_list.append(one_grid.t2m[j])
    return temp_list

# calculates the threshold of one grid cell for one day of the year
# input: 
# d: specification of a day (by index), integer between 0 and len(data frame)
# one_grid: data frame of one grid cell
# year_begin: the year of the beginning of the dataframe (integer)
# year_end: the year of the end of the dataframe (integer)
def calc_threshold(i,x,y, one_grid, year_begin, year_end):
    temp_list = []
    while (one_grid.x[i] == x) & (one_grid.y[i] == y):
        temp_list.append(one_year(i,one_grid, year_begin, year_end))
        i = i + 365
        if i >= len(one_grid):
            break
    temp_list = list(itertools.chain(*temp_list))
    threshold = calc_percentile(temp_list)
    i = i - 365
    return threshold, i

def calc_thresh_all(data, start_year, end_year, len_lon, len_lat):
    thresholds_test = pd.DataFrame(columns=['latitude', 'longitude', 'date', 'threshold'])
    j = 0
    l = 0
    for lat in range(len_lat):
        for lon in range(len_lon):
            for i in range(365):
                threshold,tmp = calc_threshold((i+j),lon, lat, data, start_year, end_year)
                date = str(data.month[i])+"/"+str(data.day[i])
                thresholds_test.loc[l] = [data.latitude[i+j], data.longitude[i+j], date, threshold]
                l = l+1
            j = tmp+1
    return thresholds_test


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

# this function checks if a temperature in a row in a dataframe is higher than a given threshold
# if temp is higher the function returns true, else: false
# input:
# data: dataframe that needs to be checked
# thresh: dataframe of thresholds
# data_row: the row of the dataframe data that needs to be compared
# thresh_row: the row of the dataframe thresh that needs to be compared
def remove_one_event(data, thresh, data_row,thresh_row):
    if data.t2m[data_row] >= thresh.threshold[thresh_row]:
        return True
    else:
        return False

# function creates a new dataframe that contains only events that have higher temperatures than the corresponding
# threshold
# input:
# data: dataset from which the events are taken
# thresholds: dataframe of thresholds
# len_lat: how many grid points on latitude do we have?
# len_lon: how many grid points on longitude do we have?
# years: how many years does our dataset contain?
def create_extr_dataset(data, thresholds, len_lat, len_lon, years):
    extr_dataset = pd.DataFrame(columns=['latitude', 'longitude', 'time', 't2m', 'x', 'y', 'day', 'month', 'year'])
    count = 0
    data_count = 0
    thresh_count = 0
    for j in range(len_lat):
        for k in range(len_lon):
            for y in range(years):
                for i in range(365):
                    if remove_one_event(data, thresholds, i+data_count,i+thresh_count):
                        extr_dataset.loc[count] = [data.latitude[i+data_count], data.longitude[i+data_count], 
                                                   data.time[i+data_count], data.t2m[i+data_count], 
                                                   data.x[i+data_count], data.y[i+data_count], data.day[i+data_count], 
                                                   data.month[i+data_count], data.year[i+data_count]]
                        count = count + 1
                data_count = data_count + 365
            thresh_count = thresh_count + 365
    return extr_dataset

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
    
###########################################


#load data and preprocessing
d = xarray.open_dataset(data_path)

#create integer based (x,y) coordinates
d['x'] = (('longitude'), np.arange(len(d.longitude)))
d['y'] = (('latitude'), np.arange(len(d.latitude)))
#convert to dataframe
vt = d.to_dataframe()
#reset index
vt.reset_index(inplace=True)
# add correct times
datetimes = pd.to_datetime(vt['time'])
# assign your new columns
vt['day'] = datetimes.dt.day
vt['month'] = datetimes.dt.month
vt['year'] = datetimes.dt.year
vt = cut_366(vt)
conv_to_degreescelcius(vt)


# calculate threshold
longitudes = len(d.longitude)
latitudes = len(d.latitude)
thresh = calc_thresh_all(vt, starty, endy, longitudes, latitudes)
# save threshold for later applications
thresh.to_csv(path_or_buf = "../../Results/thresh.csv", index=False)

# create extreme dataset
years = endy - starty
extr = create_extr_dataset(vt, thresh, longitudes,latitudes,years)

# calculate the daily magnitudes of the extr dataset
daily_magnitude(vt, extr)

# adapt extreme dataset
# append some neccessary stuff to the extr dataset
# append a column indicating geographical locations (i.e., supernode labels)
extr['g_id'] = extr.groupby(['longitude', 'latitude']).grouper.group_info[0]
extr['g_id'] = extr['g_id'].astype(np.uint32)    

# append integer-based time
times = pd.date_range(extr.time.min(), extr.time.max(), freq='D')
tdic = {time: itime for itime, time in enumerate(times)}
extr['itime'] = extr.time.apply(lambda x: tdic[x])
extr['itime'] = extr['itime'].astype(np.uint16)
# sort by time
extr.sort_values('time', inplace=True)


#remove columns 75 and 25 percentile
extr.drop(['seventyfive_percentile', 'twentyfive_percentile'], axis=1, inplace=True)

# save extreme dataset for later use
extr.to_csv(path_or_buf = "../../Results/extr_dataset.csv", index=False)

# save original dataset for later use
vt.to_csv(path_or_buf = "../../Results/original_dataset.csv", index=False)


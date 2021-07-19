# create dataset of extreme temperature events

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
            print(j)
            print(lon)
            print(lat)
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
thresh.to_csv(path_or_buf = save_path/"thresh.csv", index=False)

# create extreme dataset
years = last_year - first_year
extr = create_extr_dataset(vt, thresh, longitudes,latitudes,years)
# save extreme dataset for later use
extr.to_csv(path_or_buf = save_path/"extr_dataset.csv", index=False)


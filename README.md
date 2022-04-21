# Spatio-Temporal Patterns of European Heat Waves and their Influence on Vegetation

During my master thesis at the University of Tuebingen I created this project.

What does this project do? 

It detects and visualizes spatio-temporal patterns of heat waves over Europe from daily maximum temperature data. These heat waves are then clustered into families and clustering based on their occurence during the year and their spatial pattern. Finally the clusters are assessed for their influence on vegetation by computing the correlation between NDVI (normalized differenced vegetation index) and heat wave attributes.

# Environment creation

save yml file here

# Identification of Heatwaves from a Temperature Dataset
The program new2.py takes a netCDF4 file of daily maximum temperature data in Kelvin as input. Another input that needs to be given is the number of unique grid points that a heat wave must contain at least to be considered a heat wave. The smaller this integer is the smaller the heat waves are allowed to be spatially. 

The program converts the temperature data from Kelvin to °C, removes the 366th day of leap years, for simplicity and then calculates a daily quantile based threshold for every grid point. With this thresholds an extreme event dataset is created which only contains data with temperature values above the threshold. 

Furthermore this program calculates the heatwave magnitude index daily (HWMId), defined in 2015 by Russo et al (cite!) and appends several columns to the dataset that are neccessary for future computations like integer based time, x and y coordinates and day of year values. 

In the end the program converts the extreme dataset into a deep graph and computes heat waves by creating supernodes. The definition of heat waves can be found in my master thesis. 

The program returns the following datasets:
- threshold dataset
- extreme dataset
- nodes table
- supernodes table

```
python new2.py -do [path to daily max temperature .nc] -g [spatial boundary]
```

# Clustering of Heat Waves
My goal was to cluster the heat waves spatially and temporally. For this approach two subsequent clustering steps were applied. First the heat waves were temporally clustered with k means algorithm and then every familiy from the k means clustering was clustered spatially with UPGMA hierarchial clustering.

The input for the program kmeans.py is the in the previous step computed nodes table, the number of clusters for the k means clustering and a list of integers giving the number of clusters in the UPGMA clustering for every family. 

For k means clustering the day of year mean of a heat wave is transformed and now expressed by two values.
The family and cluster numbers of every node are appended to the nodes table in the columns F_kmeans and F_upgma. 

The program returns:
- A figure of the k means clustering result
- A histogram of the day of year distribution of the nodes in the different families
- k plots picturing the spatial distribution of the heat waves in the k means families
- k dendrograms for every UPGMA clustering of the k heat wave families
- For every cluster in the UPGMA clustering a plot of the spatial distribution of the heat waves within the cluster
- k nodes tables for every heat wave family with the added family and cluster information 
- k supernodes tables for every heat wave family with the added family and cluster information 
```
python kmeans_real.py -d [path to nodes dataset] -k [number of clusters for k means clustering] -u [number of clusters for upgma clustering - len() = k]
```

# Vegetation Correlation Analysis

## Preparation of Temperature Dataset

```python -d [path to nodes dataset of one family] -o [numbers of ocean clusters to be removed] -lsm [path to land sea mask data.nc] -b [hard boundary]

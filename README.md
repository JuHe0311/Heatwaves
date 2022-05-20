# Spatio-Temporal Patterns of European Heat Waves and their Influence on Vegetation

During my master thesis at the University of Tuebingen in the Machine Learning for Climate Science research group I created this project.

What does this project do? 

It detects and visualizes spatio-temporal patterns of heat waves over Europe from daily maximum temperature data. These heat waves are then clustered into families and clustering based on their occurence during the year and their spatial pattern. Finally the clusters are assessed for their influence on vegetation by computing the correlation between NDVI (normalized differenced vegetation index) and heat wave attributes.

# Environment creation
Throughout the project the paths were the output plots and files are saves are hardcoded. Therefore it is neccessary to have a similar folder structure as I have or to adapt the code. 
Include image of folder structure!!!!

Within this yml file you can find the environment where all neccessary python modules are installed to run this code.

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

## K Means Clustering 

The input for the program clustering_step1.py is the in the previous step computed nodes table and the number of clusters for the k means clustering.

For k means clustering the day of year mean of a heat wave is transformed and now expressed by two values.
The family membership of every node is appended to the nodes table in the column F_kmeans. Additionally UPGMA clustering is performed on every family of heat waves and a dendrogram is returned. From the dendrogram the optimal number of clusters for every family can be visually derived. 

The program returns:
- A figure of the k means clustering result
- A histogram of the day of year distribution of the nodes in the different families
- k plots picturing the spatial distribution of the heat waves in the k means families
- k dendrograms for every UPGMA clustering of the k heat wave families
- k nodes tables for every heat wave family with the added family information 
- k supernodes tables for every heat wave family with the added family information 
```
python clustering_step1.py -d [path to nodes dataset] -k [number of clusters for k means clustering]
```

## UPGMA Clustering
The program performs UPGMA clustering for one heat wave family
The input for the program clustering_step2.py is a nodes table of one heat wave family from the k means clustering, the number of clusters and the number of the family. UPGMA clustering is performed and the spatial distribution of every cluster is plotted. The cluster membership of every node is appended to the nodes table in the column F_upgma.

The program returns:
- A nodes table for the heat wave family with the added cluster membership information
- A supernodes table for the heat wave family with the added cluster membership information
- A plot of the spatial distribution for every cluster
```
python clustering_step2.py -d [path to nodes dataset] -u [number of clusters for UPGMA clustering] -i [number of the heat wave family] 
```


# Vegetation Correlation Analysis

## Preparation of Temperature Dataset
The preparation of the temperature dataset for correlation analysis contains the following steps:
1. Removal of ocean clusters: All clusters that are predominantly located over the ocean are removed, as vegetation correlation over the ocean does not make any sense
2. Removal of all individual grid points over the ocean: In all land clusters the individual ocean grid points are removed to avoid noise
3. Defining a hard boundary for every cluster: To remove noise grid points in every cluster that are only hit a few times by a heat wave are removed from the cluster. The threshold for the boundary can be set individually.

The program returns:
- The filtered nodes table for one family
```
python prepare_temp_dataset.py -d [path to nodes dataset of one family] -o [numbers of ocean clusters to be removed] -lsm [path to land sea mask data.nc] -b [hard boundary]
```

## Preparation of NDVI Dataset
The program prepare_NDVI_dataset.py takes as input NDVI data in netCDF4 format, the factor, by which the NDVI dataset should be coarsened and the season start and end point (by month) by which the dataset should be filtered.

After the coarsening monthly anomalies of the NDVI values are computed and the values are filtered so that only the values of the end of the season are kept.

The program returns a filtered and modified NDVI dataset

```
python prepare_ndvi_dataset.py -ndvi [path to NDVI dataset] -c [coarsening factor] -s [start and end month of the season]
```

## Correlation Analysis
The program NDVI_correlation.py calculates the spearman correlation coefficient between two features of heat waves (HWMId and number of heat wave days) and the NDVI for one heat wave family. The correlation is performed grid-wise for every year and every cluster. 
The two features of the heat waves are summed up for one season so that we end up with one heat wave magnitude sum and the number of heat wave days at one grid point for every year and cluster. These values are correlated with the NDVI value at the respective grid point and year.
The season start and end point can be calculated with the program define_seasons.py. Seasons are computed by taking the 10th and 90th percentile of the month distribution of the nodes of a given dataset.

The program returns:
- A correlation coefficient table for the HWMId correlation
- A correlation coefficient table for the number of heat wave days correlation
```
python NDVI_correlation.py -ndvi [path to modified NDVI dataset] -d [filtered nodes table for one family] -s [start and end month of the season]
```

## Visualization of Mean Correlation Analysis
The program plot_correlation.py visualized the correlation coefficients for one family of heat waves computed in the previous step. The two correlation coefficient tables are needed as input. For this only the significant corelation coefficient values are used. The significance threshold is set to a = 0.01. First the mean correlation coefficient over all years for every cluster and both correlation variables (hwmid and number of heat waves) are calculated and printed to the console.
Secondly the time series of correlation coefficients for every cluster is plotted if the number of correlation values for this year and cluster is >= 30. The R^2 value and the slope of the linear regression line are reported. Additionally the Mann-Kendall test is performed and its result is printed to the console.

The program returns:
- The updated nodes and HWMId correlation table with only significant correlation values
```
python plot_correlation.py -n [path to the correlation coefficient table for the number of heat wave days correlation] -hwmid [path to the correlation coefficient table for the HWMId correlation]
```
## Individual Correlation Analysis
The program individual_correlation.py takes the nodes table that is outputted from the UPGMA clustering, the prepared NDVI dataset and the HWMId and nodes correlation table as input. It first appends a column of the absolute correlation values to the HWMId and nodes correlation tables and then sorts them by this absolute value from high to low. The heat wave magnitude and number of heat waves and heat days for the 10 strongest correlated years and clusters is plotted.
In a second step the nodes table is divided into three parts by years. For all three datasets significant correlation values are computed and visually compared by plotting the values on boxplots.

```

python individual_correlation.py -n [path to the correlation coefficient table for the number of heat wave days correlation] -hwmid [path to the correlation coefficient table for the HWMId correlation] -ndvi [path to the prepared ndvi dataset] -d [path to the nodes table output from the UPGMA clustering]
```

# Plotting Some More Results
The program plotting_results.py takes the nodes dataset of the extreme heat days and returns several different visualizations that can be found in my master thesis. For this one must also input the number of heat wave that should be plotted as an example.

The program returns:
- A pairplot of the variables: n_nodes, HWMId_magnitude, timespan, ytime_mean, year
- A map plot of the specified heat wave
- A map plot of the progression of the specified heat wave

```
python plotting_results.py -d [Path to the nodes dataset] -n [number of the heat wave to be plotted]
```




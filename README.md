# Heatwaves

During my master thesis at the University of Tuebingen I created this project.

What does this project do? 

It detects and visualizes spatio-temporal patterns of heat waves over Europe from daily maximum temperature data. These heat waves are then clustered into families and clustering based on their occurence during the year and their spatial pattern. Finally the clusters are assessed for their influence on vegetation by computing the correlation between NDVI (normalized differenced vegetation index) and heat wave attributes.

# Environment creation

save yml file here

# Identification of Heatwaves from a Temperature Dataset

python new2.py -do [path to daily max temperature .nc] -g [spatial boundary]

# Clustering of Heatwaves

python kmeans_real.py -d [path to nodes dataset] -k [number of clusters for k means clustering] -u [number of clusters for upgma clustering - len() = k]

# Vegetation Correlation Analysis

## Preparation of Temperature Dataset

python 

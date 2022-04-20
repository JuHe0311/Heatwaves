# Heatwaves

# Environment creation

save yml file here

# Identification of Heatwaves from a Temperature Dataset

python new2.py -do [path to daily max temperature .nc] -g [spatial boundary]

# Clustering of Heatwaves

python kmeans_real.py -d [path to nodes dataset] -k [number of clusters for k means clustering] -u [number of clusters for upgma clustering - len() = k]

## Pre

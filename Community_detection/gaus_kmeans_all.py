# Imports:
import os
import xarray
# for plots
import matplotlib.pyplot as plt
# the usual
import numpy as np
import deepgraph as dg
import pandas as pd
import itertools
import scipy
import argparse
import sklearn
import plotting as pt
import con_sep as cp
from sklearn.cluster import KMeans
import seaborn as sns
import math
import cppv
from mpl_toolkits.mplot3d import Axes3D
# kmeans clustering 
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.mixture import GaussianMixture
############### Argparser ################

def make_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", help="Give the path to the extreme value dataset.",
                        type=str)
    return parser

parser = make_argparser()
args = parser.parse_args()

cpv = pd.read_csv(args.data)
X = pd.DataFrame(columns = [ 'x_centroids', 'y_centroids','ytime_mean','timespan','volume'])
X['x_centroids'] = cpv.x_calc_centroid
X['y_centroids'] = cpv.y_calc_centroidy
X['ytime_mean'] = cpv.ytime_mean   
X['timespan'] = cpv.timespan
X['volume'] = cpv.n_unique_g_ids             


range_n_clusters = [2, 3, 4, 5, 6,7,8,9,10]

for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    #fig, (ax1, ax2) = plt.subplots(1, 2)
    #fig.set_size_inches(25, 10)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    #ax1.set_xlim([-1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    #ax1.set_ylim([0, len(cpv) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=100)
    cluster_labels = clusterer.fit_predict(cpv[['x_calc_centroid','y_calc_centroidy','ytime_mean','timespan','n_unique_gids']])

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(cpv[['x_calc_centroid','y_calc_centroidy','ytime_mean','timespan','n_unique_g_ids']], cluster_labels)
    print(
        "For n_clusters =",
        n_clusters,
        "The average silhouette_score for k-means clustering is :",
        silhouette_avg,
    )
    
    model = GaussianMixture(n_components=n)
    # fit the model
    model.fit(X)
    # assign a cluster to each example
    yhat = model.predict(X)
    X['cluster'] = yhat
    # retrieve unique clusters
    clusters = unique(yhat)
    silhouette_avg_gaus = silhouette_score(cpv[['x_calc_centroid','y_calc_centroidy','ytime_mean','timespan','n_unique_g_ids']], X['cluster'])
    print(
        "For n_clusters =",
        n_clusters,
        "The average silhouette_score for gaussian mixture models is :",
        silhouette_avg_gaus,
    )
    # Compute the silhouette scores for each sample
    #sample_silhouette_values = silhouette_samples(cpv[['x_calc_centroid','y_calc_centroidy','ytime_mean','timespan','n_unique_gids']], cluster_labels)

    #y_lower = 10
    #for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
      #  ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

      #  ith_cluster_silhouette_values.sort()

      #  size_cluster_i = ith_cluster_silhouette_values.shape[0]
      #  y_upper = y_lower + size_cluster_i

      #  color = cm.nipy_spectral(float(i) / n_clusters)
      #  ax1.fill_betweenx(
      #      np.arange(y_lower, y_upper),
      #      0,
      #      ith_cluster_silhouette_values,
      #      facecolor=color,
      #      edgecolor=color,
       #     alpha=0.7,
       # )

        # Label the silhouette plots with their cluster numbers at the middle
      #  ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
       # y_lower = y_upper + 10  # 10 for the 0 samples

   # ax1.set_title("The silhouette plot for the various clusters.")
   # ax1.set_xlabel("The silhouette coefficient values")
   # ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
   # ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

   # ax1.set_yticks([])  # Clear the yaxis labels / ticks
   # ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    
    # 2nd Plot showing the actual clusters formed
   # colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
   # ax2=fig.add_subplot(projection='3d')
   # xs = cpv['x_calc_centroid']
   # ys = cpv['y_calc_centroidy']
   # zs = cpv['ytime_mean']
   # ax2.scatter(
   #     xs,ys,zs, marker=".", s=50, lw=0, alpha=0.7, c=colors, edgecolor="k"
   # )

    # Labeling the clusters
   # centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
   # ax2.scatter(
   #     centers[:, 0],
   #     centers[:, 1],
   #     marker="o",
    #    c="white",
    #    alpha=1,
    #    s=200,
    #    edgecolor="k",
    #)

   # for i, c in enumerate(centers):
    #    ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

   # ax2.set_title("The visualization of the clustered data.")
   # ax2.set_xlabel("X_centroids")
   # ax2.set_ylabel("Y_centroids")
   # ax2.set_zlabel('Day of year mean')

   # plt.suptitle(
   #     "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
    #    % n_clusters,
    #    fontsize=14,
    #    fontweight="bold",
    #)
    #plt.savefig("../../Results/k_means_centroids%s.png" % n_clusters)
    

    

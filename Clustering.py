# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 11:43:54 2020

@author: Raneen_new
"""

from matplotlib import pyplot
from sklearn.cluster import KMeans
from scipy.interpolate import interp1d
from kneed import KneeLocator
import numpy as np
from kneed import KneeLocator
from tslearn.clustering import KShape
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as hac
from scipy.cluster.hierarchy import fcluster
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS

class Clustering:
    def __init__(self,cluster_lists):
        self.cluster_lists = cluster_lists

    def scale_data(self,cluster_lists):
        i = 0
        if(cluster_lists != []):
            for data in cluster_lists:
                #print(len(cluster_lists[i]))
                for idx,seq in enumerate(data):
                    if(seq != []):
                        max_seq = max(seq)
                        min_seq = min(seq)
                        i = 0
                        while (i < len(seq)):
                            seq[i] = (seq[i] - min_seq) / (max_seq - min_seq)
                            i += 1
                    else:
                        data.pop(idx)
                i+=1        
        return cluster_lists

    def k_mean_clustering(self,num_clusters,data):
        kmeans = KMeans(init="random",n_clusters=num_clusters,n_init=10,max_iter=300,random_state=42)
        kmeans.fit(data)
        kmeans_kwargs = {"init": "random","n_init": 12,"max_iter": 300,"random_state": 42,}
        return kmeans.cluster_centers_

    def K_shape_clustering(self,num_clusters,data,layer_len):
        # Calculate length of maximal list
        n = len(max(data, key=len))
        # Make the lists equal in length
        lst_2 = [x + [0]*(n-len(x)) for x in data]
        a = np.nan_to_num(np.array(lst_2))
        print('K_shape_data')
        kshape = KShape(n_clusters=num_clusters, verbose=True, random_state=42)
        kshape.fit(a)
        name = 'MHAP_layer_data/cluster_center'+str(layer_len)+'.npy'
        np.save(name,kshape.cluster_centers_)
        return kshape.cluster_centers_
    
    def fancy_dendrogram(self,*args, **kwargs):
        max_d = kwargs.pop('max_d', None)
        if max_d and 'color_threshold' not in kwargs:
            kwargs['color_threshold'] = max_d
        annotate_above = kwargs.pop('annotate_above', 0)
    
        ddata = hac.dendrogram(*args, **kwargs)
    
        if not kwargs.get('no_plot', False):
            pyplot.title('Hierarchical Clustering Dendrogram (truncated)')
            pyplot.xlabel('sample index or (cluster size)')
            pyplot.ylabel('distance')
            for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
                x = 0.5 * sum(i[1:3])
                y = d[1]
                if y > annotate_above:
                    pyplot.plot(x, y, 'o', c=c)
                    pyplot.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                                 textcoords='offset points',
                                 va='top', ha='center')
            if max_d:
                pyplot.axhline(y=max_d, c='k')
        return ddata
    
    def print_clusters(self,timeSeries, Z, k, plot=False):
        # k Number of clusters I'd like to extract
        results = fcluster(Z, k, criterion='maxclust')
    
        # check the results
        s = pd.Series(results)
        clusters = s.unique()
    
        for c in clusters:
            cluster_indeces = s[s==c].index
            print("Cluster %d number of entries %d" % (c, len(cluster_indeces)))
            if plot:
                timeSeries.T.iloc[:,cluster_indeces].plot()
                pyplot.show()
        return clusters
    
    def hierarchical_cluster(self,num_clusters,data):
        #cluster = AgglomerativeClustering(n_clusters=num_clusters, affinity='euclidean', linkage='ward')  
        #cluster.fit_predict(data)
        # Do the clustering
        data = np.array(data)
        df = pd.DataFrame(data=data)
        data = df.dropna()
        Z = hac.linkage(data, method='complete', metric='euclidean')
        
        # Plot dendogram
        pyplot.figure(figsize=(25, 10))
        pyplot.title('Hierarchical Clustering Dendrogram')
        pyplot.xlabel('sample index')
        pyplot.ylabel('distance')
        """hac.dendrogram(
            Z,
            truncate_mode='lastp',  # show only the last p merged clusters
            p=12,  # show only the last p merged clusters
            show_leaf_counts=False,  # otherwise numbers in brackets are counts
            leaf_rotation=90.,
            leaf_font_size=12.,
            show_contracted=True,  # to get a distribution impression in truncated branches
        )"""
        self.fancy_dendrogram(
            Z,
            truncate_mode='lastp',
            p=12,
            leaf_rotation=90.,
            leaf_font_size=12.,
            show_contracted=True,
            annotate_above=10,  # useful in small plots so annotations don't overlap
        )
        pyplot.show()
        return self.print_clusters(data, Z, num_clusters, plot=False)
    
    def DBscan_cluster(self,num_clusters,data):
        data = np.array(data)
        data[np.isnan(data)] = 0
        clustering = DBSCAN(eps=0.8, min_samples = 15).fit(data)
        return clustering
    
    def optic_cluster(self,num_clusters,data):
        data = np.array(data)
        data[np.isnan(data)] = 0
        clustering = OPTICS(min_samples=num_clusters).fit(data)
        return clustering
    
    def cluster_sequence_data(self,cluster_number,layer_len,cluser_data_pre_list1):
        # scale the data between 0 and 1
        cluster_centers = []
        cluster_lists = cluser_data_pre_list1
        
        #loop thriugh the periods for each CNN layer
        count = 0
        for layer in (cluster_lists):
            #cluster_centers.append(self.hierarchical_cluster(cluster_number[count],layer))
            #cluster_centers.append(self.k_mean_clustering(cluster_number[count],layer))
            cluster_centers.append(self.K_shape_clustering(cluster_number[count],layer,layer_len[count]))
            #cluster_centers.append(self.DBscan_cluster(cluster_number[count],layer))
            #cluster_centers.append(self.optic_cluster(cluster_number[count],layer))
            count +=1
        
        return  cluster_centers
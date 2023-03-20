# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 14:24:41 2020

This class will return the highly activated filters(feature map) for each layer in anetwork
"""
import tensorflow.keras as keras
from matplotlib import pyplot
import numpy as np
from scipy import stats, integrate
from scipy.interpolate import interp1d
import seaborn as sns
from scipy.spatial import distance
import networkx as nx
from itertools import combinations
from sklearn.metrics import accuracy_score
from scipy.stats import ttest_1samp
import scipy.stats as stats
import bisect
from joblib import Parallel, delayed

np.random.seed(0)

class HighlyActivated:
    def __init__(self, model,train_model,test_data,y_pred,nb_classes,netLayers):
        self.model = model
        self.x_test = test_data
        self.y_pred = y_pred
        self.nb_classes = nb_classes
        self.netLayers = netLayers
        self.train_model = train_model
        
    def Activated_filters(self,example_id):
        #layer_outputs = [layer.output for layer in self.model.layers[:self.netLayers+3]] 
        layer_outputs = [layer.output for layer in self.model.layers[:self.netLayers+9]] 
        # Extracts the outputs of the top n layers
        # Creates a model that will return these outputs, given the model input
        activation_model = keras.models.Model(inputs=self.model.input, outputs=layer_outputs) 
        activations = activation_model.predict(self.x_test)
        #shows the activated filters for each layer for an example    
        return activations    
    
    def predect(self,y_true,x_test,model):
        y_pred = model.predict(x_test)
        y_pred = np.argmax(y_pred, axis=1)
        keras.backend.clear_session()        
        return y_pred
      
    def define_threshould(self,activations):
        threshoulds = [[] for i in range(self.netLayers)]
        layer_data = [[] for i in range(self.netLayers)]
        ff = True
        activated_id = 1
        filter_index = 0
        for l in range(1,self.netLayers+9):
            flag = False
            #get the id of the filter layer
            if(l == 3):
                activated_id = 0
                filter_index = 0
                flag = True                    
            elif(l==6):
                activated_id = 1
                filter_index = 1
                flag = True  
            elif(l==9):
                activated_id = 2
                filter_index = 2
                flag = True
            #so make sure that we only have the id of conv layer
            if(flag):
                activated_nodes = activations[l]
                xx = []
                for i in range(0,activated_nodes.shape[2]):
                    Q3 = np.average(activated_nodes[:, :, i]) 
                    xx.append(Q3)
                Q3 = np.percentile(xx,0.95)
                layer_data[filter_index].append(Q3)
        
        return layer_data
    
    def get_index_clustering_MHAP(self,activations,kernal_size=[]):
        kernal_size= kernal_size
        filter_lists = [[] for i in range(self.netLayers)]
        period_active = []
        threshoulds = self.define_threshould(activations)
        #pool = mp.Pool(mp.cpu_count())
        #classes_lists_per[[] for i in range(self.netLayers)]
        layer_mhaps = []
        index_mhaps = []
        for j in range(len(self.x_test)):
            activated_id = 1
            #len of the period_will depend on the lyer kernal size (first will be same and then multily of previous)
            Len_period = 8
            filter_index = 0
            sample_layer_mhaps = [[] for i in range(self.netLayers)]
            sample_index_mhaps = [[] for i in range(self.netLayers)]
            for l in (3,6,9):
                flag = False
                #get the id of the filter layer
                if(l == 3):
                    Len_period = kernal_size[0]
                    filter_index = 0
                    #flag = True                    
                elif(l==6):
                    Len_period = kernal_size[0] * kernal_size[1]
                    filter_index = 1
                    #flag = True  
                elif(l==9):
                    Len_period = kernal_size[0]*kernal_size[1]*kernal_size[2]
                    filter_index = 2
                    #flag = True
                #so make sure that we only have the id of conv layer
                #if(flag):
                    #active id is what conv layer of network, 
                activated_nodes = activations[l]
                #each filter of the data(activated_nodes.shape[2]) has (n values based on strid(inour case =1))
                #filter_list_mhap = [[] for i in range(activated_nodes.shape[2])]
                #filter_index_list = [[] for i in range(activated_nodes.shape[2])]
                for i in range(0,activated_nodes.shape[2]):
                    index_k = 0
                    filter_list_mhap =[]
                    filter_index_list = []
                    #loop throug the feature map (the neurons of the filter)
                    for k in (activated_nodes[j, :, i]):
                        if(k >= threshoulds[filter_index]):
                            mhap_list = [[] for i in range(self.x_test[j].shape[1])]
                            if(index_k+Len_period < (self.x_test[j].shape[0])):
                                for idx in range((self.x_test[j].shape[1])):
                                    mhap_list[idx].append(self.x_test[j,index_k:index_k+Len_period,idx]) 
                                
                                filter_list_mhap.append(mhap_list)
                                filter_index_list.append(index_k)
                                #filter_list_mhap[i].append(mhap_list)
                                #filter_index_list[i].append(index_k)
                                filter_lists[filter_index].extend(mhap_list) 
                        index_k +=1
                    sample_layer_mhaps[filter_index].append(filter_list_mhap)  
                    sample_index_mhaps[filter_index].append(filter_index_list)
            layer_mhaps.append(sample_layer_mhaps)  
            index_mhaps.append(sample_index_mhaps)  
        return filter_lists,layer_mhaps,index_mhaps
    
    def generate_MHAP_evl_graph(self,cluster_central,layer_mhaps,index_mhaps):
        kernal_size = [8,5,3]
        #initilize the array of the output (size will be based on output class label)
        layer_node = []
        graph = nx.DiGraph()
        #for each data sample
        node_layer_name = []
        for j in range(len(layer_mhaps)):
            print(j)
            #loop through netwrok layer
            sample_layer_node = [[] for i in range(len(layer_mhaps[0]))]
            for l in range(len(layer_mhaps[j])):
                #for each filter
                for m in range(len(layer_mhaps[j][l])):
                    filter_list_node =[]
                    prev_node = ['' for i in range(len(layer_mhaps[0][0][0][0]))]
                    #for each neuron get mhap
                    for f in range(len(layer_mhaps[j][l][m])):
                        ##here I have access to the MHAPs and its index (loop thorugh input domention)
                        d_node = []
                        for d in range(len(layer_mhaps[j][l][m][f])):
                            if(len(np.array(layer_mhaps[j][l][m][f][d]).tolist())!=0):
                                cluster_id = self.fitted_cluster(np.array(layer_mhaps[j][l][m][f][d]).tolist(),cluster_central[l])
                                node_name = 'layer%s %s'%((l),(cluster_id))
                                d_node.append(node_name)
                                #connect graph
                                if(prev_node[d] != '' and node_name != ''):
                                    if graph.has_edge(prev_node[d], node_name):
                                        graph[prev_node[d]][node_name]['weight'] += 1
                                    else:
                                        graph.add_edge(prev_node[d], node_name, weight=1)
                                prev_node[d] = node_name
                                #I need to connect the layeri+1 with its previous MHAP nodes
                                if(l > 0):
                                    start = index_mhaps[j][l][m][f]
                                    end = (index_mhaps[j][l][m][f])+kernal_size[l]
                                    if(start<=f<end):
                                        #get the value of indexes index_mhaps[j][l-1][:]where (they between start and end) 
                                        for n in range(len(index_mhaps[j][l-1])):
                                            for a in range(len(index_mhaps[j][l-1][n])):
                                                if(index_mhaps[j][l-1][n][a] == index_mhaps[j][l][m][f]):
                                                    ##connect edges
                                                    prev_layer_node = sample_layer_node[l-1][n][a][d]
                                                    if graph.has_edge(prev_layer_node, node_name):
                                                        graph[prev_layer_node][node_name]['weight'] += 1
                                                    else:
                                                        graph.add_edge(prev_layer_node, node_name, weight=1)
                                                  
                        filter_list_node.append(d_node)
                    sample_layer_node[l].append(filter_list_node)
            node_layer_name.append(sample_layer_node)
        return graph,node_layer_name  
    
    def get_segmant_MHAP_par(self,j,kernal_size,node_layer_name,index_mhaps,n,sgmant_lenth,intervals):
        print(j)
        #for each data sample return its segment representation
        segmant = [[] for i in range(n)]
        for l in range(len(index_mhaps[j])):
            for m in range(len(index_mhaps[j][l])):
                for f in range(len(index_mhaps[j][l][m])):
                    index_k = index_mhaps[j][l][m][f]
                    index_segma = bisect.bisect_left(intervals, index_k)-1
                    name_node = node_layer_name[j][l][m][f]
                    segmant[index_segma].extend(name_node)
        return segmant
    
    def get_segmant_MHAP(self,kernal_size,node_layer_name,index_mhaps,n,sgmant_lenth):
        #kernal_size = [8,5,3]
        #loop through each training sample
        sgmant_lenth = sgmant_lenth
        start = 0
        n = n
        sgmant_lenth = sgmant_lenth
        intervals = np.arange(start, sgmant_lenth * n , sgmant_lenth)
        #print(intervals)
        results = []
        #loop through each training sample
        results = Parallel(n_jobs=100,backend="threading")(delayed(self.get_segmant_MHAP_par)(j,kernal_size,node_layer_name,index_mhaps,n,sgmant_lenth,intervals) for j in range(len(self.x_test)))
        return results
    
    def fitted_cluster(self,data,cluster):
        data = np.linalg.norm(data[0])
        cluster[0] = np.linalg.norm(cluster[0][0])
        mini =0
        if(np.isnan(cluster[0]).any() == False & np.isinf(cluster[0]).any() == False):
            mini = distance.euclidean(data,cluster[0])
        cluster_id = 0
        count = 0
        for i in (cluster):
            clu_nor = np.linalg.norm(i)
            #print(clu_nor)
            #print(data)
            if(np.isnan(clu_nor).any() == False & np.isinf(clu_nor).any() == False):
                dist = distance.euclidean(data,clu_nor)
                if(dist < mini):
                    cluster_id = count
                    mini = dist
                count+=1
        
        return cluster_id

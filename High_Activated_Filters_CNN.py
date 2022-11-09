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
from sklearn import preprocessing
from sklearn.preprocessing import normalize
from fitter import Fitter, get_common_distributions, get_distributions
import seaborn as sns
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
                channels_activation_values =[]
                #loop thourth the channels/filters 
                for i in range(0,activated_nodes.shape[2]):
                    channels_activation_values = activated_nodes[:, :, i]
                    #Q3 = np.percentile(activated_nodes[:, :, i], 97)
                    Q3 = np.quantile(activated_nodes[:, :, i], 0.95)
                layer_data[filter_index].append(Q3)
                ##plot activation distrubustion    
                #sns.set_style('white')
                #sns.set_context("paper", font_scale = 2)
                #sns.displot(data=channels_activation_values, kind="hist", bins = 100, aspect = 1.5)   
                #f = Fitter(channels_activation_values,distributions=['expon'])
                #f.fit()
                #f.summary()
        
        return layer_data
    
    def mhap_get_par(self,j,threshoulds,activations,kernal_size):
        #loop through each layer in the network
        # we need to take only the conv layers(not the conv and pooling)
        #so we take each second index of the loop
        activated_id = 1
        #len of the period_will depend on the lyer kernal size (first will be same and then multily of previous)
        Len_period = 8
        filter_index = 0
        #layer_fil = []
        filter_lists = [[] for i in range(self.netLayers)]
        #period_active = []
        for l in range(1,self.netLayers+9):
            flag = False
            #get the id of the filter layer
            if(l == 3):
                Len_period = kernal_size[0]
                filter_index = 0
                flag = True                    
            elif(l==6):
                Len_period = kernal_size[0] * kernal_size[1]
                filter_index = 1
                flag = True  
            elif(l==9):
                Len_period = kernal_size[0]*kernal_size[1]*kernal_size[2]
                filter_index = 2
                flag = True
            #so make sure that we only have the id of conv layer
            if(flag):
                #active id is what conv layer of network, 
                #print(activated_id)
                activated_nodes = activations[l]
                #each filter of the data(activated_nodes.shape[2]) has (n values based on strid(inour case =1))                       
                for i in range(0,activated_nodes.shape[2]):
                    index_k = 0
                    #loop throug the feature map
                    #print(threshoulds)
                    for k in (activated_nodes[j, :, i]):
                        if(k >= threshoulds[filter_index]):
                            d1 =self.x_test[j][:,0]
                            d2 =self.x_test[j][:,1]
                            mhap1 = []
                            mhap2 = []
                            #mhap3 = []
                            if(index_k+Len_period < len(d1)): 
                                mhap1 = d1[index_k:index_k+Len_period]
                                mhap2 = d2[index_k:index_k+Len_period]
                            else:
                                if(filter_index == 2):
                                    if(index_k+Len_period <len(d1)):
                                        mhap1 = d1[index_k:index_k+Len_period]
                                        mhap2 = d2[index_k:index_k+Len_period]
                                    #else:
                                    #    mhap1.append(0)
                                    #    mhap2.append(0)
                            mhap1 = list(mhap1)
                            mhap2 = list(mhap2)
                            filter_lists[filter_index].append(mhap1)
                            filter_lists[filter_index].append(mhap2)            
                        index_k +=1
        return filter_lists
    
        
    def get_index_MHAP(self,activations,kernal_size=[]):
        #depends on the network archi
        threshoulds = self.define_threshould(activations)
        results = []
        #loop through each training sample
        results = Parallel(n_jobs=100,backend="threading")(delayed(self.mhap_get_par)(j,threshoulds,activations,kernal_size) for j in range(len(self.x_test)))
        return results,threshoulds
    
    def normilization(self,data):
        i = 0
        datt = []
        maxi = max(data)
        mini = abs(min(data))
        while (i< len(data)):
            if(data[i] >=0):
                if(maxi == 0):
                    val = 0
                else:
                    val = data[i]/maxi
            else:
                if(mini ==0):
                    val = 0
                else:
                    val = data[i]/mini
         
            datt.append(val)
            i += 1
            
        return datt
    def fitted_cluster(self,data,cluster):
        data = self.normilization(data)
        #data = normalize(data[:,np.newaxis], axis=0).ravel()
        #cluster[0] = preprocessing.normalize([cluster[0][0]])
        cluster[0] = self.normilization([cluster[0][0]])
        mini =0
        if(np.isnan(cluster[0]).any() == False & np.isinf(cluster[0]).any() == False):
            mini = distance.euclidean(data,cluster[0])
        cluster_id = 0
        count = 0
        for i in (cluster):
            clu_nor = preprocessing.normalize(i)
            if(np.isnan(clu_nor).any() == False & np.isinf(clu_nor).any() == False):
                dist = distance.euclidean(data,clu_nor)
                if(dist < mini):
                    cluster_id = count
                    mini = dist
                count+=1
        
        return cluster_id
    
    def get_graph_MHAP(self,activations,kernal_size,cluster_central,threshoulds,n,sgmant_lenth):
        kernal_size = [8,5,3]
        #initilize the array of the output (size will be based on output class label)
        #classes_lists = [[] for i in range(self.nb_classes)] 
        graph1 = nx.DiGraph()
        graph2 = nx.DiGraph()
        graph3 = nx.DiGraph()
        graph = nx.DiGraph()
        graph_data_each_sample = []
        node_assigned = []
        #loop through each training sample
        sample_cluster_mhap = []
        intervals = np.arange(0, sgmant_lenth * n , sgmant_lenth)
        for j in range(len(self.x_test)):
            print(j)
            sample_x_graph=[]
            graph_layer_node = [[] for i in range(len(kernal_size))]
            index_layer = [[] for i in range(len(kernal_size))]
            #loop through each layer in the network
            # we need to take only the conv layers(not the conv and pooling)
            #so we take each second index of the loop
            activated_id = 1
            #len of the period_will depend on the lyer kernal size (first will be same and then multily of previous)
            Len_period = 8
            filter_index = 0
            layer_cluster_mhap = []
            
            segmant = [[] for i in range(n)]
            interval = 0
            #loop through each layer in the network
            # we need to take only the conv layers(not the conv and pooling)
            #so we take each second index of the loop

            #len of the period_will depend on the lyer kernal size (first will be same and then multily of previous)
            for l in range(0,self.netLayers+6):
                flag = False
                #get the id of the filter layer
                if(l == 1):
                    activated_id = 0
                    Len_period = kernal_size[0]
                    filter_index = 0
                    flag = True                    
                elif(l==4):
                    activated_id = 1
                    Len_period = kernal_size[0] * kernal_size[1]
                    filter_index = 1
                    flag = True  
                elif(l==7):
                    activated_id = 2
                    Len_period = kernal_size[0]*kernal_size[1]*kernal_size[2]
                    filter_index = 2
                    flag = True
                #so make sure that we only have the id of conv layer
                if(flag):
                    #here use networkx to create the directed graph      
                    #active id is what conv layer of network, 
                    activated_nodes = activations[l]
                    #each filter of the data(activated_nodes.shape[2]) has (n values based on strid(inour case =1))                       
                    for i in range(0,activated_nodes.shape[2]):
                        index_k = 0
                        #loop throug the feature map
                        prev_node1 = ''
                        prev_node2 = ''
                        prev_node3 = ''
                        for k in (activated_nodes[j, :, i]):
                            if(k >= threshoulds[filter_index]):
                                d1 =self.x_test[j][:,0]
                                d2 =self.x_test[j][:,1]
                                mhap1 = []
                                mhap2 = []
                                #mhap3 = []
                                if(index_k+Len_period < len(d1)): 
                                    mhap1 = d1[index_k:index_k+Len_period]
                                    mhap2 = d2[index_k:index_k+Len_period]
                                else:
                                    if(filter_index == 2):
                                        if(index_k+Len_period <len(d1)):
                                            mhap1 = d1[index_k:index_k+Len_period]
                                            mhap2 = d2[index_k:index_k+Len_period]
                                        #else:
                                        #    mhap1.append(0)
                                        #    mhap2.append(0)
                                mhap1 = list(mhap1)
                                mhap2 = list(mhap2)
                                index_layer[activated_id].append(index_k)
                                #here we have 3 MHAP we want to compre cluster
                                cluster_1 = ''
                                cluster_id_1 = ''
                                if(len(mhap1)):
                                    #print(mhap1)
                                    #here we cheack cluster fit, we want build graph (sampleX1->)
                                    #node_1,node2
                                    cluster_id_1 = self.fitted_cluster(mhap1,cluster_central[filter_index])
                                    cluster_1 = 'layer%s %s'%((activated_id),(cluster_id_1))
                                    sample_x_graph.append(cluster_1)

                                ################################################################# hooon
                                layer_previous = []
                                if(activated_id > 0 and (cluster_1 != '')):
                                    prev_layer_index_k = list(range(index_k,index_k+Len_period))
                                    #get the matches indexes from previous layer 
                                    intersect, ind_a, ind_b = np.intersect1d(index_layer[activated_id-1],prev_layer_index_k, return_indices=True)
                                    #ind_a is where in intersect
                                    for i in ind_a:
                                        if(i < len(graph_layer_node[activated_id-1])):
                                            layer_previous.append(graph_layer_node[activated_id-1][i])
                                #here we add the current layer node
                                if(cluster_1 == ''):
                                    pass
                                else:
                                    graph_layer_node[activated_id].append(cluster_1)

                                if(activated_id == 0):
                                    if(prev_node1 != '' and cluster_1 != ''):
                                        #here g . add edges
                                        sample_x_graph.append(prev_node1)
                                        sample_x_graph.append(cluster_1)
                                        if(index_k == 0):
                                            segmant[0].append(prev_node1)
                                            segmant[0].append(cluster_1)
                                        else:      
                                            index_segma = bisect.bisect_left(intervals, index_k)-1
                                            #print(index_segma)
                                            segmant[index_segma].append(prev_node1)
                                            segmant[index_segma].append(cluster_1)
                                        if graph1.has_edge(prev_node1, cluster_1):
                                        # we added this one before, just increase the weight by one
                                            graph1[prev_node1][cluster_1]['weight'] += 1
                                            graph[prev_node1][cluster_1]['weight'] += 1
            
                                        else:
                                        # new edge. add with weight=1
                                            if(prev_node1 != '' and cluster_1 != ''):
                                                graph1.add_edge(prev_node1, cluster_1, weight=1)
                                                graph.add_edge(prev_node1, cluster_1, weight=1)
                                #here we cheack if layer_previous not empty we add all its node to the current graph node
                                #cluster_1
                                elif(activated_id ==1):          
                                    #here we cheack if index mhap connect cluster id then we add edge
                                    if(prev_node1 != '' and cluster_1 != ''):
                                        sample_x_graph.append(prev_node1)
                                        sample_x_graph.append(cluster_1)
                                        if(index_k == 0):
                                            segmant[0].append(prev_node1)
                                            segmant[0].append(cluster_1)
                                        else:      
                                            index_segma = bisect.bisect_left(intervals, index_k)-1
                                            #print(index_segma)
                                            segmant[index_segma].append(prev_node1)
                                            segmant[index_segma].append(cluster_1)
                                        if graph2.has_edge(prev_node1, cluster_1):
                                        # we added this one before, just increase the weight by one
                                            graph2[prev_node1][cluster_1]['weight'] += 1
                                            graph[prev_node1][cluster_1]['weight'] += 1
                                            if(layer_previous!=[]):
                                                for k in layer_previous:
                                                    sample_x_graph.append(k)
                                                    if(index_k == 0):
                                                        segmant[0].append(k)
                                                    else:      
                                                        index_segma = bisect.bisect_left(intervals, index_k)-1
                                                        #print(index_segma)
                                                        segmant[index_segma].append(k)
                                                    if graph.has_edge(cluster_1,k):
                                                        graph[cluster_1][k]['weight'] += 1
                                                    else:
                                                        if(prev_node1 != '' and cluster_1 != ''):
                                                            graph.add_edge(cluster_1, k, weight=1)
                                            #graph_layer_node[activated_id].append(cluster_1)
                                        else:
                                        # new edge. add with weight=1
                                            graph2.add_edge(prev_node1, cluster_1, weight=1)
                                            graph.add_edge(prev_node1, cluster_1, weight=1)
                                            if(layer_previous!=[]):
                                                for k in layer_previous:
                                                    if graph.has_edge(cluster_1,k):
                                                        graph[cluster_1][k]['weight'] += 1
                                                    else:
                                                        if(prev_node1 != '' and cluster_1 != ''):
                                                            graph.add_edge(cluster_1, k, weight=1)
                                           
                                else:
                                    if(prev_node1 != '' and cluster_1 != ''):
                                        sample_x_graph.append(prev_node1)
                                        sample_x_graph.append(cluster_1)
                                        if(index_k == 0):
                                            segmant[0].append(prev_node1)
                                            segmant[0].append(cluster_1)
                                        else:      
                                            index_segma = bisect.bisect_left(intervals, index_k)-1
                                            #print(index_segma)
                                            segmant[index_segma].append(prev_node1)
                                            segmant[index_segma].append(cluster_1)
                                        if graph3.has_edge(prev_node1, cluster_1):
                                        # we added this one before, just increase the weight by one
                                            graph3[prev_node1][cluster_1]['weight'] += 1
                                            graph[prev_node1][cluster_1]['weight'] += 1
                                            #graph_layer_node[activated_id].append(cluster_1)
                                            if(layer_previous!=[]):
                                                for k in layer_previous:
                                                    if graph.has_edge(cluster_1,k):
                                                        graph[cluster_1][k]['weight'] += 1
                                                    else:
                                                        if(prev_node1 != '' and cluster_1 != ''):
                                                            graph.add_edge(cluster_1, k, weight=1)
                                        else:
                                        # new edge. add with weight=1
                                            
                                            graph3.add_edge(prev_node1, cluster_1, weight=1)
                                            graph.add_edge(prev_node1, cluster_1, weight=1)
                                            #graph_layer_node[activated_id].append(cluster_1)
                                            if(layer_previous!=[]):
                                                for k in layer_previous:
                                                    sample_x_graph.append(k)
                                                    if(index_k == 0):
                                                        segmant[0].append(k)
                                                    else:      
                                                        index_segma = bisect.bisect_left(intervals, index_k)-1
                                                        #print(index_segma)
                                                        segmant[index_segma].append(k)
                                                    if graph.has_edge(cluster_1,k):
                                                        graph[cluster_1][k]['weight'] += 1
                                                    else:
                                                        if(prev_node1 != '' and cluster_1 != ''):
                                                            graph.add_edge(cluster_1, k, weight=1)

                                prev_node1 = 'layer%s %s'%((activated_id),(cluster_id_1))                                                         
                            index_k +=1   
            sample_cluster_mhap.append(segmant)    
        return graph,sample_cluster_mhap

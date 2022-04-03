# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 14:24:41 2020

@author: raneen_pc

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
        layer_outputs = [layer.output for layer in self.model.layers[:self.netLayers+6]] 
        # Extracts the outputs of the top n layers
        # Creates a model that will return these outputs, given the model input
        activation_model = keras.models.Model(inputs=self.model.input, outputs=layer_outputs) 
        activations = activation_model.predict(self.x_test)
        #shows the activated filters for each layer for an example

        #for i in range(0,self.netLayers+3):
        for i in range(0,self.netLayers+3):
            flag = False
            if i == 0:
            #or i == 1:
                activated_nodes = activations[i]
                flag = True
            #elif(i%2 == 1 and i >1):
            #    activated_nodes = activations[i]
            #    flag = True
            if(flag):
                n_filters, ix = activated_nodes.shape[2], 1
                for j in range(0,n_filters):
                        # specify subplot and turn of axis
                        ax = pyplot.subplot(n_filters, 3, ix)
                        ax.set_xticks([])
                        ax.set_yticks([])
                        # plot filter channel
                        pyplot.plot(activated_nodes[example_id, :, j])
                        ix += 1
                pyplot.show()     
        return activations
                
    def get_best_distribution(sel,data):
        dist_names = ["norm", "exponweib", "weibull_max", "weibull_min", "pareto", "genextreme"]
        dist_results = []
        params = {}
        for dist_name in dist_names:
            dist = getattr(stats, dist_name)
            param = dist.fit(data)
        params[dist_name] = param
        # Applying the Kolmogorov-Smirnov test
        D, p = stats.kstest(data, dist_name, args=param)
        dist_results.append((dist_name, p))
        # select the best fitted distribution
        best_dist, best_p = (max(dist_results, key=lambda item: item[1]))
        return best_dist, best_p, params[best_dist]
    
    def change_dimention(self,x_test,num_dim):
        new_data = np.copy(x_test)
        for xe in range(len(x_test)):
            for i in range(len(new_data[xe])):
                    for n in num_dim:  
                            new_data[xe][i][n]=0  
        return new_data
    
    def predect(self,y_true,x_test,model):
        y_pred = model.predict(x_test)
        y_pred = np.argmax(y_pred, axis=1)
        keras.backend.clear_session()        
        return y_pred
    
    def get_dimention_MHAP(self,x_test):
        #dimention = [[] for i in range(self.nb_classes)]
        x_test_label = [[] for i in range(self.nb_classes)]
        y_test_label = [[] for i in range(self.nb_classes)]
        x = []
        x_test_copy = np.copy(x_test)
        combination_id = []
        for i in range (x_test_copy.shape[2]):
                x.append(i)
                tu = []
                tu.append(i)
                combination_id.append(tu)
            
        r = []
        for i in range(2,x_test_copy.shape[2]):
            r.append(list(combinations(x, i)))
                
        for h in (r):
            for l in h:
                combination_id.append(l)
        
        #combination_id all posible combination of the signal data
        #dim_test = []
        for z in range(len(self.y_pred)):
            index_label = self.y_pred[z]
            i = index_label.tolist().index(1) # i will return index of 2
            index_label = i
            x_test_label[index_label].append(x_test_copy[z])
            y_test_label[index_label].append(self.y_pred[z])
        
        combination_ids = []
        for i in combination_id:
            combination_ids.append(list(i))
        result_acc = []
        for i in combination_ids:
            ind = 0
            for js in x_test_label:
                xx = self.change_dimention(js,i)
                rounded_labels=np.argmax(y_test_label[ind], axis=1)
                y_pred = self.predect(rounded_labels,xx,self.model)
                result_acc.append(accuracy_score(rounded_labels, y_pred))
                ind +=1
        return result_acc
    
    def define_threshould(self,activations):
        threshoulds = [[] for i in range(self.netLayers)]
        layer_data = [[] for i in range(self.netLayers)]
        ff = True
        for j in range(len(self.x_test)):
            activated_id = 1
            filter_index = 0
            #layer_fil = []
            for l in range(1,self.netLayers+6):
                flag = False
                #get the id of the filter layer
                if(l == 1):
                    activated_id = 0
                    filter_index = 0
                    flag = True                    
                elif(l==4):
                    activated_id = 1
                    filter_index = 1
                    flag = True  
                elif(l==7):
                    activated_id = 2
                    filter_index = 2
                    flag = True
                #so make sure that we only have the id of conv layer
                if(flag):
                    activated_nodes = activations[l]
                    channel = [[] for i in range(activated_nodes.shape[2])]
                    for i in range(0,activated_nodes.shape[2]):
                        for k in (activated_nodes[j, :, i]):
                            channel[i].append(k)
                    
                    layer_data[filter_index].append(channel)
       
        #each layer has n-data and each n data has m filters 
        layer_data1 = [[] for i in range(self.netLayers)]
        channel_len = [32,64,128]
        for i in range(len(layer_data)):
            for l in range((channel_len[i])):
                layer_data1[i].append([])
            for j in range(len(layer_data[i])):
                for k in range(len(layer_data[i][j])):
                    for n in layer_data[i][j][k]:
                        layer_data1[i][k].append(n)
        #now we want to define a threshold for each filter of each array 
        for i in range(len(threshoulds)):
            mean_val = 0
            std_val = 0
            for l in range((channel_len[i])):
                threshoulds[i].append([])
                mean_val = np.mean(layer_data1[i][l])
                std_val = np.std(layer_data1[i][l])
                Q3 = np.percentile(layer_data1[i][l], 98)
                normal=stats.norm(loc=mean_val, scale=std_val)
                threshoulds[i][l].append(mean_val)
                threshoulds[i][l].append(std_val)
                threshoulds[i][l].append(normal)
                threshoulds[i][l].append(Q3)
                
        return threshoulds
    def define_threshould_all_filters(self,activations):
        threshoulds = [[] for i in range(self.netLayers)]
        layer_data = [[] for i in range(self.netLayers)]
        ff = True
        for j in range(len(self.x_test)):
            activated_id = 1
            filter_index = 0
            #layer_fil = []
            for l in range(1,self.netLayers+6):
                flag = False
                #get the id of the filter layer
                if(l == 1):
                    activated_id = 0
                    filter_index = 0
                    flag = True                    
                elif(l==4):
                    activated_id = 1
                    filter_index = 1
                    flag = True  
                elif(l==7):
                    activated_id = 2
                    filter_index = 2
                    flag = True
                #so make sure that we only have the id of conv layer
                if(flag):
                    activated_nodes = activations[l]
                    #channel = [[] for i in range(activated_nodes.shape[2])]
                    for i in range(0,activated_nodes.shape[2]):
                        for k in (activated_nodes[j, :, i]):
                            layer_data[filter_index].append(k)
                
        for i in range(len(threshoulds)):
            data_int = layer_data[i]
            sorted_integers = sorted(data_int)
            #for l in (len(layer_data[i])):
            mean_val = np.mean(layer_data[i])
            std_val = np.std(layer_data[i])
            Q3 = np.percentile(layer_data[i], 98)
            normal=stats.norm(loc=mean_val, scale=std_val)
            threshoulds[i].append(mean_val)
            threshoulds[i].append(std_val)
            threshoulds[i].append(normal)
            threshoulds[i].append(Q3)
            threshoulds[i].append(sorted_integers[-10])      
                
        return threshoulds
    
    def get_index_MHAP(self,activations,kernal_size=[]):
        #depends on the network archi
        #kernal_size = [8,5,3]
        kernal_size= kernal_size
        #initilize the array of the output (size will be based on output class label)
        #classes_lists = [[] for i in range(self.nb_classes)]
        filter_lists = [[] for i in range(self.netLayers)]
        filter_lists_index = [[] for i in range(self.netLayers)]
        period_active = []
        index_period_active = []
        #threshoulds = self.define_threshould_all_filters(activation_layers)
        threshoulds = self.define_threshould_all_filters(activation_layers)
        #classes_lists_period = [[] for i in range(self.nb_classes)]
        #loop through each training sample
        for j in range(len(self.x_test)):
            #loop through each layer in the network
            # we need to take only the conv layers(not the conv and pooling)
            #so we take each second index of the loop
            activated_id = 1
            #len of the period_will depend on the lyer kernal size (first will be same and then multily of previous)
            Len_period = 8
            filter_index = 0
            #layer_fil = []
            for l in range(1,self.netLayers+6):
                flag = False
                #get the id of the filter layer
                if(l == 1):
                    Len_period = kernal_size[0]
                    filter_index = 0
                    flag = True                    
                elif(l==4):
                    Len_period = kernal_size[0] * kernal_size[1]
                    filter_index = 1
                    flag = True  
                elif(l==7):
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
                        val = 0.8
                        for k in (activated_nodes[j, :, i]):
                            if(filter_index == 0):
                                val = 3.2
                            elif(filter_index==1):
                                #print(threshoulds[filter_index][i][2].pdf(k))
                                val = 0.45
                            else:
                                val = 0.65
                            if(k >= threshoulds[filter_index][3]):
                            #if(threshoulds[filter_index][2].pdf(k) >val):
                                #here pertubipt data from this index to len of period and get feature as mhap
                                #dimentions = self.get_dimention_MHAP(j,index_k,Len_period)
                                #dimentions = [1]
                                #here have an array to the periods, will be same as order(original traning data)
                                #each sample data will have periods from each dimention, for each layer
                                #[[[l1_p],[l2_p],[l3_p]],.....]
                                #print('ccc')
                                d1 =[]
                                #d2 =[]
                                #d3 =[]
                                #get each dimention data
                                for xe in(self.x_test[j]):
                                    d1.append(xe[0])
                                    #d2.append(xe[1])
                                    #d3.append(xe[2])
                                mhap1 = []
                                #mhap2 = []
                                #mhap3 = []
                                if(index_k+Len_period < len(d1)): 
                                    for l in range(index_k,index_k+Len_period):
                                        mhap1.append(d1[l])
                                        #mhap2.append(d2[l])
                                        #mhap3.append(d3[l])
    
                                filter_lists[filter_index].append(mhap1)
                                #filter_lists[filter_index].append(mhap2)
                                #filter_lists[filter_index].append(mhap3)
                                #add also what is time period
                                filter_lists_index[filter_index].append(index_k)               
                            index_k +=1

            period_active.append(filter_lists)        
            index_period_active.append(filter_lists_index)      
        return period_active,index_period_active
    
    def get_graph_MHAP(self,activations,kernal_size,cluster_central):
        kernal_size = [8,5,3]
        #initilize the array of the output (size will be based on output class label)
        #classes_lists = [[] for i in range(self.nb_classes)]    
        graph1 = nx.DiGraph()
        graph2 = nx.DiGraph()
        graph3 = nx.DiGraph()
        graph = nx.DiGraph()
        index_layer = [[] for i in range(len(kernal_size))]
        graph_layer_node = [[] for i in range(len(kernal_size))]
        graph_data_each_sample = []
        node_assigned = []
        #threshoulds = self.define_threshould_all_filters(activation_layers)
        threshoulds = self.define_threshould_all_filters(activation_layers)
        #loop through each training sample
        sample_cluster_mhap = []
        for j in range(len(self.x_test)):
            sample_x_graph=[]
            #loop through each layer in the network
            # we need to take only the conv layers(not the conv and pooling)
            #so we take each second index of the loop
            activated_id = 1
            #len of the period_will depend on the lyer kernal size (first will be same and then multily of previous)
            Len_period = 8
            filter_index = 0
            layer_cluster_mhap = []
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
                        val = 0.8
                        for k in (activated_nodes[j, :, i]):
                            if(filter_index == 0):
                                val = 2.9
                            elif(filter_index==1):
                                #print(threshoulds[filter_index][i][2].pdf(k))
                                val = 0.30
                            else:
                                val = 0.30
                                
                            if(k >= threshoulds[filter_index][3]):
                            #if(threshoulds[filter_index][2].pdf(k) >val):
                                d1 =[]
                                d2 =[]
                                d3 =[]
                                #get each dimention data
                                for xe in(self.x_test[j]):
                                    d1.append(xe[0])
                                    d2.append(xe[1])
                                    d3.append(xe[2])
                                mhap1 = []
                                mhap2 = []
                                mhap3 = []
                                if(index_k+Len_period < len(d1)): 
                                    for l in range(index_k,index_k+Len_period):
                                        mhap1.append(d1[l])
                                        mhap2.append(d2[l])
                                        mhap3.append(d3[l])
                                index_layer[activated_id].append(index_k)
                                #here we have 3 MHAP we want to compre cluster
                                cluster_1 = ''
                                cluster_id_1 = ''
                                #cluster_2 = ''
                                #cluster_id_2 = ''
                                #cluster_3 = ''
                                #cluster_id_3 = ''
                                if(len(mhap1)):
                                    #print(mhap1)
                                    #here we cheack cluster fit, we want build graph (sampleX1->)
                                    #node_1,node2
                                    x_n = []
                                    x_n.append(mhap1)
                                    cluster_id_1 = self.fitted_cluster(mhap1,cluster_central[filter_index])
                                    #cluster_id_1 = cluster_central[filter_index].predict(x_n)
                                    #cluster_id_1 = cluster_id_1[0] 
                                    #print(cluster_id_1)
                                    #self.fitted_cluster(mhap1,cluster_central[filter_index])
                                    cluster_1 = 'layer%s %s'%((activated_id),(cluster_id_1))
                                    sample_x_graph.append(cluster_1)
                                #if(len(mhap1)):
                                #    cluster_id_2 = self.fitted_cluster(mhap2,cluster_central[filter_index])
                                #    cluster_2 = 'Layer_%s %s'%(activated_id,cluster_id_2)
                                #if(len(mhap1)):
                                #    cluster_id_3 = self.fitted_cluster(mhap3,cluster_central[filter_index])
                                #    cluster_3 = 'Layer_%s %s'%(activated_id,cluster_id_3)
                                
                                ################################################################# hooon
                                layer_previous = []
                                if(activated_id > 0 and (cluster_1 != '')):
                                    for ju in range(index_k,index_k+Len_period):
                                        index_layer_grap = 0 
                                        for ku in index_layer[activated_id-1]:
                                            if(ju == ku):
                                                #means this is also mhap from previous layer
                                                #print(graph_layer_node[activated_id-1][index_layer_grap])
                                                if(index_layer_grap < len(graph_layer_node[activated_id-1])):
                                                    layer_previous.append(graph_layer_node[activated_id-1][index_layer_grap])

                                            index_layer_grap +=1
                                    #print(layer_previous) 
                                if(cluster_1 == ''):
                                    pass
                                else:
                                    graph_layer_node[activated_id].append(cluster_1)
                                #sample_x_graph.append(cluster_1)
                                if(activated_id == 0):
                                    if(prev_node1 != '' and cluster_1 != ''):
                                        #here g . add edges
                                        if graph1.has_edge(prev_node1, cluster_1):
                                        # we added this one before, just increase the weight by one
                                            graph1[prev_node1][cluster_1]['weight'] += 1
                                            graph[prev_node1][cluster_1]['weight'] += 1
                                            
                                        else:
                                        # new edge. add with weight=1
                                            if(prev_node1 != '' and cluster_1 != ''):
                                                graph1.add_edge(prev_node1, cluster_1, weight=1)
                                                graph.add_edge(prev_node1, cluster_1, weight=1)
                                            #graph_layer_node[activated_id].append(cluster_1)
                                        sample_x_graph.append(prev_node1)
                                        sample_x_graph.append(cluster_1)
                                #here we cheack if layer_previous not empty we add all its node to the current graph node
                                #cluster_1
                                elif(activated_id ==1):
                                    #here we cheack if index mhap connect cluster id then we add edge
                                    if(prev_node1 != '' and cluster_1 != ''):
                                        if graph2.has_edge(prev_node1, cluster_1):
                                        # we added this one before, just increase the weight by one
                                            graph2[prev_node1][cluster_1]['weight'] += 1
                                            graph[prev_node1][cluster_1]['weight'] += 1
                                            if(layer_previous!=[]):
                                                for k in layer_previous:
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
                                            #graph_layer_node[activated_id].append(cluster_1)
                                        sample_x_graph.append(prev_node1)
                                        sample_x_graph.append(cluster_1)
                                        if(layer_previous!=[]):
                                            for k in layer_previous:
                                                sample_x_graph.append(k)
                                        
                                else:
                                    if(prev_node1 != '' and cluster_1 != ''):
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
                                                    if graph.has_edge(cluster_1,k):
                                                        graph[cluster_1][k]['weight'] += 1
                                                    else:
                                                        if(prev_node1 != '' and cluster_1 != ''):
                                                            graph.add_edge(cluster_1, k, weight=1)
                                                        
                                        sample_x_graph.append(prev_node1)
                                        sample_x_graph.append(cluster_1)
                                        if(layer_previous!=[]):
                                            for k in layer_previous:
                                                sample_x_graph.append(k)
                                prev_node1 = 'layer%s %s'%((activated_id),(cluster_id_1))
                                #prev_node2 = cluster_2
                                #prev_node3 = cluster_3                                    
                                
                            index_k +=1   
            sample_cluster_mhap.append(sample_x_graph)
        return graph1,graph2,graph3,graph,index_layer,graph_data_each_sample,sample_cluster_mhap

    def normilization(self,data):
        i = 0
        datt = []
        maxi = max(data)
        mini = abs(min(data))
        while (i< len(data)):
            
            if(data[i] >=0):
                val = data[i]/maxi
            else:
                val = data[i]/mini
         
            datt.append(val)
            i += 1
            
        return datt

    
    def fitted_cluster(self,data,cluster):
        data = self.normilization(data)
        cluster[0] = self.normilization(cluster[0])
        mini =0
        if(np.isnan(cluster[0]).any() == False & np.isinf(cluster[0]).any() == False):
            mini = distance.euclidean(data,cluster[0])
        cluster_id = 0
        count = 0
        for i in (cluster):
            clu_nor = self.normilization(i)
            if(np.isnan(clu_nor).any() == False & np.isinf(clu_nor).any() == False):
                dist = distance.euclidean(data,clu_nor)
                if(dist < mini):
                    cluster_id = count
                    mini = dist
                count+=1
        
        return cluster_id
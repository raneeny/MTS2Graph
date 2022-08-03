# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 13:54:41 2020

@author: 

Read a dataset and train a model for classify the data.
"""
from Data_Preprocessing import ReadData
from ConvNet_Model import ConvNet
import numpy as np
import tensorflow.keras as keras
import sys, getopt
from High_Activated_Filters_CNN import HighlyActivated
import pandas
from itertools import *
from  functools import *
from Clustering import Clustering
from matplotlib import pyplot
import tensorflow.keras as keras
from matplotlib import pyplot
import numpy as np
from scipy import stats, integrate
from scipy.interpolate import interp1d
import seaborn as sns
from scipy.spatial import distance
import matplotlib.pyplot as plt
np.random.seed(0)
import networkx as nx
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from numpy import mean
from numpy import std
from Graph_embading import Graph_embading

def readData(data_name,dir_name):
    dir_path = dir_name + data_name+'/'
    dataset_path = dir_path + data_name +'.mat'

    ##read data and process it
    prepare_data = ReadData()
    prepare_data.data_preparation(dataset_path, dir_path)
    datasets_dict = prepare_data.read_dataset(dir_path,data_name)
    x_train = datasets_dict[data_name][0]
    y_train = datasets_dict[data_name][1]
    x_test = datasets_dict[data_name][2]
    y_test = datasets_dict[data_name][3]
    x_train, x_test = prepare_data.z_norm(x_train, x_test)
    nb_classes = prepare_data.num_classes(y_train,y_test)
    y_train, y_test, y_true = prepare_data.on_hot_encode(y_train,y_test)
    x_train, x_test, input_shape = prepare_data.reshape_x(x_train,x_test)
    #create train validation subvalidation sub set
    #x_training, x_validation = x_train[:90,:], x_train[90:,:]
    #y_training, y_validation = y_train[:90,:], y_train[90:,:]
    x_training = x_train
    y_training = y_train
    #x_new = np.concatenate((x_training, x_validation), axis=0)
    x_new1 = np.concatenate((x_train, x_test), axis=0)
    #y_new = np.concatenate((y_training, y_validation), axis=0)
    y_new1 = np.concatenate((y_train, y_test), axis=0)
    x_training, x_validation, y_training, y_validation = train_test_split(x_new1, y_new1, test_size=0.20,shuffle=True)

    x_validation,x_test,y_validation,y_test = train_test_split(x_validation, y_validation, test_size=0.50,shuffle=True)

    #y_training= np.argmax(y_training, axis=1)
    y_true= np.argmax(y_test, axis=1)
    return x_training, x_validation, x_test, y_training, y_validation, y_true,y_test, input_shape,nb_classes


def trainModel(x_training, x_validation, y_training, y_validation,input_shape, nb_classes):
    ##train the model
    train_model = ConvNet()
    #ResNet
    #model = train_model.networkResNet(input_shape,nb_classes)
    #FCN 
    model = train_model.network_fcN(input_shape,nb_classes)
    #cnn
    #model = train_model.network(input_shape,nb_classes)
    print(model.summary())
    train_model.trainNet(model,x_training,y_training,x_validation,y_validation,16,2000)
    return model,train_model

def predect(y_true,x_test,model,train_model,dimention_deactivated):
    y_pred = model.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1)
    keras.backend.clear_session()
    file = open('../Results/file_name.csv','a')
    file.write(str(dimention_deactivated))
    file.close()
    df_metrics = train_model.calculate_metrics(y_true, y_pred, 0.0)
    df = pandas.DataFrame(df_metrics).transpose()
    df.to_csv('../Results/file_name.csv', mode='a')
    return y_pred
    
def visulize_active_filter(model,x_test,y_true,nb_classes,train_model,cluster_centers,netLayers=3):
    ##visulize activated filters for the original testing dataset
    dimention_deactivated = 'all'
    y_pred = predect(y_true,x_test,model,train_model,dimention_deactivated)
    visulization = HighlyActivated(model,x_test,y_pred,nb_classes,netLayers=3)
    activation_layers = visulization.Activated_filters(example_id=1)
    visulization.get_high_activated_filters(activation_layers,dimention_deactivated)
    activated_class_cluster = visulization.show_high_activated_period(activation_layers,dimention_deactivated,cluster_centers)
    visulization.print_high_activated_combunation(activated_class_cluster)
    ##visulize activated filters when set all dimention of the data to zero, and just one with its original data
    x = []
    combination_id = []
    for i in range (x_test.shape[2]):
        x.append(i)
        tu = []
        tu.append(i)
        combination_id.append(tu)
    
    r = []
    for i in range(2,x_test.shape[2]):
        r.append(list(combinations(x, i)))
        
    for h in (r):
        for l in h:
            combination_id.append(l)
    print(combination_id)
    multivariate_variables = [[] for i in range(len(combination_id))]
    for i in range(len(combination_id)):
        multivariate_variables[i] = np.copy(x_test)
        for j in range(len(multivariate_variables[i])):      
            for k in range(len(multivariate_variables[i][j])):
                for n in range(x_test.shape[2]):
                    if (n not in combination_id[i]):
                        multivariate_variables[i][j][k][n] = 0
                    else:
                        dimention_deactivated =  ''.join(map(str,combination_id[i])) 
        y_pred = predect(y_true,multivariate_variables[i],model,train_model,dimention_deactivated)
        visulization = HighlyActivated(model,multivariate_variables[i],y_pred,nb_classes,netLayers=3)
        activation_layers = visulization.Activated_filters(example_id=1)
        visulization.get_high_activated_filters(activation_layers,dimention_deactivated)
        activated_class_cluster = visulization.show_high_activated_period(activation_layers,dimention_deactivated,cluster_centers)
        visulization.print_high_activated_combunation(activated_class_cluster)
        
def cluster_data_compenation(model,x_training,y_training,nb_classes):
    visulization_traning = HighlyActivated(model,x_training,y_training,nb_classes,netLayers=3)
    activation_layers = visulization_traning.Activated_filters(example_id=1)
    period_indexes, filter_number = visulization_traning.get_high_active_period(activation_layers)
    cluster_periods = visulization_traning.extract_dimention_active_period(period_indexes)
    ##clustring the periods
    cluster_data = []
  
    cluster_number = [12,11,13]
    #clustering = Clustering(cluster_periods)
    #print(new_data)
    kshape = KShape(n_clusters=12, verbose=True, random_state=42)
    trans_x = np.nan_to_num(cluster_periods[0])
    kshape.fit(trans_x)
    cluster_centers = kshape.cluster_centers_

    cluster_data.append(cluster_centers)

    x = []
    combination_id = []
    for i in range (x_training.shape[2]):
        x.append(i)
        tu = []
        tu.append(i)
        combination_id.append(tu)
    
    r = []
    for i in range(2,x_training.shape[2]):
        r.append(list(combinations(x, i)))
        
    for h in (r):
        for l in h:
            combination_id.append(l)
    multivariate_variables = [[] for i in range(len(combination_id))]
    for i in range(len(combination_id)):
        multivariate_variables[i] = np.copy(x_training)
        for j in range(len(multivariate_variables[i])):      
            for k in range(len(multivariate_variables[i][j])):
                for n in range(x_training.shape[2]):
                    if (n not in combination_id[i]):
                        multivariate_variables[i][j][k][n] = 0
                    else:
                        dimention_deactivated =  ''.join(map(str,combination_id[i])) 
        visulization_traning = HighlyActivated(model,x_training,y_training,nb_classes,netLayers=3)
        activation_layers = visulization_traning.Activated_filters(example_id=1)
        period_indexes, filter_number = visulization_traning.get_high_active_period(activation_layers)
        cluster_periods = visulization_traning.extract_dimention_active_period(period_indexes)
       
        kshape = KShape(n_clusters=12, verbose=True, random_state=42)
        trans_x = np.nan_to_num(cluster_periods[0])
        kshape.fit(trans_x)
        cluster_centers = kshape.cluster_centers_

        cluster_data.append(cluster_centers)
    
   #save the cluster center for each layer in different array
    cluser_center1 = []
    cluser_center2 = []
    cluser_center3 = []
    cluser_center = []
    l = 0
    for i in cluster_data: 
        for j in i:
            if(l == 0):
                cluser_center1.append(j)
            elif(l == 1):
                cluser_center2.append(j)
            else:
                cluser_center3.append(j)
        l +=1
    cluser_center.append(cluser_center1)
    cluser_center.append(cluser_center2)
    cluser_center.append(cluser_center3)
    #return cluser_center

    return cluser_center


def normilization(data):
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

#compare to cluster
def fitted_cluster(data,cluster):
        data = normilization(data)
        cluster[0] = normilization(cluster[0])
        mini = distance.euclidean(data,cluster[0])
        cluster_id = 0
        count = 0
        for i in (cluster):
            clu_nor = normilization(i)
            dist = distance.euclidean(data,clu_nor)
            if(dist < mini):
                cluster_id = count
                mini = dist
            count+=1
            
        return cluster_id
if __name__ == '__main__':
    run(sys.argv[1:])
    data_name = sys.argv[1:]
    dir_name = sys.argv[2:]
    x_training, x_validation, x_test, y_training, y_validation, y_true,y_test, input_shape,nb_classes = readData(data_name,dir_name)
    start_time = time.time()
    model,train_model = trainModel(x_training, x_validation, y_training, y_validation,input_shape, nb_classes)
    y_pred = predect(y_true,x_test,model,train_model,'alls')
    visulization_traning = HighlyActivated(model,train_model,x_training,y_training,nb_classes,netLayers=3)
    start_time = time.time()
    activation_layers = visulization_traning.Activated_filters(example_id=1)
    period_active,thresholds = visulization_traning.get_index_MHAP(activation_layers,kernal_size=[8,5,3])
    cluser_data_pre_list = []
    cluser_data_pre_list = []
    filter_lists = [[] for i in range(3)]
    for i in range(len(period_active)):
        for j in range(len(period_active[i])):
            for k in range(len(period_active[i][j])):
                filter_lists[j].append(period_active[i][j][k])
    start_time = time.time()
    cluser_data_pre_list.append([x for x in filter_lists[0] if x])
    cluser_data_pre_list.append([x for x in filter_lists[1] if x])
    cluser_data_pre_list.append([x for x in filter_lists[2] if x])
    clustering = Clustering(cluser_data_pre_list)
    cluser_data_pre_list1 = clustering.scale_data(cluser_data_pre_list)
    clustering = Clustering(cluser_data_pre_list)
    cluster_central = clustering.cluster_sequence_data([35,25,15],[8,40,120],cluser_data_pre_list)
    G,id_layer,sample_cluster_mhap_ = visulization_traning.get_graph_MHAP(activation_layers,[8,40,120],cluster_central)
    sample_cluster_mhap = visulization_traning.get_segmant_MHAP(activation_layers,[8,40,120],cluster_central,9,10)

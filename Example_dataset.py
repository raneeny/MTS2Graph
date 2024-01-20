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
import tensorflow as tf
from sklearn.model_selection import train_test_split
import xgboost as xgb
from Graph_embading import Graph_embading
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from numpy import mean
from numpy import std
import pandas as pd 
from joblib import Parallel, delayed
import networkx as nx
import csv
import time
import getopt
np.random.seed(0)
#import deepwalk
def readData(data_name,dir_name):
    dir_path = dir_name + data_name+'/'
    dataset_path = dir_path + data_name +'.mat'
    
    ##read data and process it
    prepare_data = ReadData()
    if(data_name == "HAR"):
        dataset_path = dir_name + data_name +'/train.pt'
        x_training = torch.load(dataset_path)
        x_train = x_training['samples']
        y_train = x_training['labels']
        dataset_path = dir_name + data_name +'/train.pt'
        x_testing = torch.load(dataset_path)
        x_test = x_testing['samples']
        y_test = x_testing['labels']
        x_train = x_train.cpu().detach().numpy()
        y_train = y_train.cpu().detach().numpy()
        x_test = x_test.cpu().detach().numpy()
        y_test = y_test.cpu().detach().numpy()
        #reshape array(num_sample,ts_len,dim)
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[2], x_train.shape[1])
        x_test = x_test.reshape(x_test.shape[0],x_test.shape[2], x_test.shape[1])
    elif(data_name == "PAMAP2"):
        dataset_path = dir_name + data_name +'/PTdict_list.npy'
        x_data = np.load(dataset_path)
        dataset_path = dir_name + data_name +'/arr_outcomes.npy'
        y_data = np.load(dataset_path)
        split_len = int(len(x_data)*0.9)
        x_train,x_test  = x_data[:split_len,:], x_data[split_len:,:]
        y_train,y_test  = y_data[:split_len,:], y_data[split_len:,:]
        
    else:
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
    x_training = x_train
    y_training = y_train
 
    x_new1 = np.concatenate((x_train, x_test), axis=0)
    y_new1 = np.concatenate((y_train, y_test), axis=0)
    x_training, x_validation, y_training, y_validation = train_test_split(x_new1, y_new1, test_size=0.80,shuffle=True)
    x_validation,x_test,y_validation,y_test = train_test_split(x_validation, y_validation, test_size=0.50,shuffle=True)
    print(x_training.shape)
    print(x_validation.shape)
    print(x_test.shape)
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
    #file = open('../Results/file_name.csv','a')
    #file.write(str(dimention_deactivated))
    #file.close()
    df_metrics = train_model.calculate_metrics(y_true, y_pred, 0.0)
    df = pandas.DataFrame(df_metrics).transpose()
    #df.to_csv('../Results/file_name.csv', mode='a')
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
            #print(clu_nor)
            dist = distance.euclidean(data,clu_nor)
            if(dist < mini):
                cluster_id = count
                mini = dist
            count+=1
            
        return cluster_id


def downsample_to_proportion(rows, proportion=1):
        i = 0
        new_data = []
        #new_data.append(rows[0])
        new_data.append(rows[0])
        k = 0
        for i in (rows):
            if(k == proportion):
                new_data.append(i)
                k = 0
            k+=1
        return new_data 


def timeseries_embedding(embedding_graph,node_names,timesereis_MHAP,number_seg):
    feature_list = []
    embed_vector = embedding_graph.wv[node_names]
    for i,data in enumerate(timesereis_MHAP):
        #compare the name with word_list and take its embedding
        #loop through segmant
        segmant = [[] for i in range(number_seg)]
        #print(len(data))
        for m,seg in enumerate(data):
            temp = [0 for i in range(len(embed_vector[0]))]
            #each seg has mhaps
            for k,mhap in enumerate(seg):
                for j,node in enumerate(node_names):
                    if(mhap == node):
                        temp += embed_vector[j]
                        break
            segmant[m].append(list(temp))
        feature_list.append(segmant)
    return feature_list

def run(argv):
    data_name = ''
    dir_name = ''
    cluster_numbers = [35, 25, 15]  # default values for cluster numbers
    segment_length = 10  # default segment lengths
    activation_threshold = 0.95  # default activation threshold
    embedding_size = 100  # default embedding size
    
    try:
        opts, args = getopt.getopt(argv, "hf:d:c:s:a:e:", ["file_name=", "directory_name=", "cluster_numbers=", "segment_length=", "activation_threshold=", "embedding_size="])
    except getopt.GetoptError:
        print('Train_example_dataset.py -f <file name> -d <directory name> -c <cluster numbers> -s <segment length> -a <activation threshold> -e <embedding size>')
        sys.exit(2)

    print(opts)
    for opt, arg in opts:
        if opt == '-h':
            print('Train_example_dataset.py -f <file name> -d <directory name> -c <cluster numbers> -s <segment length> -a <activation threshold> -e <embedding size>')
            sys.exit()
        elif opt in ("-f", "--file"):
            data_name = arg
        elif opt in ("-d", "--directory"):
            dir_name = arg
        elif opt in ("-c", "--cluster_numbers"):
            cluster_numbers = list(map(int, arg.split(',')))
        elif opt in ("-s", "--segment_length"):
            segment_length = int(arg)
        elif opt in ("-a", "--activation_threshold"):
            activation_threshold = int(arg)
        elif opt in ("-e", "--embedding_size"):
            embedding_size = int(arg)

     
    #data_sets = ['ArabicDigits','AUSLAN','CharacterTrajectories','CMUsubject16','ECG','JapaneseVowels','KickvsPunch','Libras','NetFlow','PEMS','UWave','Wafer','WalkvsRun']
    #data_sets = ['Wafer','UWave','AUSLAN','HAR','ArabicDigits','NetFlow','PAMAP2']
    #data_sets = ['UWave','AUSLAN','HAR','ArabicDigits','NetFlow','PAMAP2']
    #for i in data_sets:
        #x_training, x_validation, x_test, y_training, y_validation, y_true,input_shape, nb_classes = readData(i,dir_name)
    #dir_name = 'mtsdata/'
    #data_name = i
    #dir_name = '../../Multivision-framework/Data/mtsdata/'
    x_training, x_validation, x_test, y_training, y_validation, y_true,y_test, input_shape,nb_classes = readData(data_name,dir_name)
    start_time = time.time()
    model,train_model = trainModel(x_training, x_validation, y_training, y_validation,input_shape, nb_classes)
    #y_pred = predect(y_true,x_test,model,train_model,'alls')
    time_data = []
    time_data.append(time.time() - start_time)
    print("--- %s seconds ---" % (time.time() - start_time))
    #######
    kernal_size=[8,5,3]
    visulization_traning = HighlyActivated(model,train_model,x_training,y_training,nb_classes,netLayers=3)
    start_time = time.time()
    activation_layers = visulization_traning.Activated_filters(example_id=1)
    time_data.append(time.time() - start_time)
    print("--- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    period_active,layer_mhaps,index_mhaps = visulization_traning.get_index_clustering_MHAP(activation_layers,kernal_size,activation_threshold)
    time_data.append(time.time() - start_time)
    print("--- %s seconds ---" % (time.time() - start_time))
    ###################
    #now cluster the MHAP for each layer
    #put the data of each layer in one array [[l1],[l2],[l3]]
    cluser_data_pre_list = []
    cluser_data_pre_list = []
    filter_lists = [[] for i in range(3)]
    for i in range(len(period_active)):
        for j in range(len(period_active[i])):
            #for k in range(len(period_active[i][j])):np.array(layer_mhaps[j][l][f][d]).tolist()
            filter_lists[i].append(np.array(period_active[i][j]).tolist())
    start_time = time.time()
    cluser_data_pre_list.append([x[0] for x in filter_lists[0] if x])
    cluser_data_pre_list.append([x[0] for x in filter_lists[1] if x])
    cluser_data_pre_list.append([x[0] for x in filter_lists[2] if x])
    
    print(len(cluser_data_pre_list[0]))
    print(len(cluser_data_pre_list[1]))
    print(len(cluser_data_pre_list[2]))
    time_data.append(time.time() - start_time)
    print("--- %s seconds ---" % (time.time() - start_time))
    cluser_data_pre_list1 = []
    cluser_data_pre_list1.append(downsample_to_proportion(cluser_data_pre_list[0], 100))
    cluser_data_pre_list1.append(downsample_to_proportion(cluser_data_pre_list[1], 100))
    cluser_data_pre_list1.append(downsample_to_proportion(cluser_data_pre_list[2], 100))
    cluser_data_pre_list1 = np.array(cluser_data_pre_list1)
    #cluser_data_pre_list1 = np.array(cluser_data_pre_list)
    start_time = time.time()
    clustering = Clustering(cluser_data_pre_list1)
    #cluser_data_pre_list1 = clustering.scale_data(cluser_data_pre_list1)
    clustering = Clustering(cluser_data_pre_list1)
    cluster_central = clustering.cluster_sequence_data(cluster_numbers,[8,40,120],cluser_data_pre_list1)
    time_data.append(time.time() - start_time)
    print("--- %s seconds ---" % (time.time() - start_time))
    ###############
    start_time = time.time()
    G,node_layer_name = visulization_traning.generate_MHAP_evl_graph(cluster_central,layer_mhaps,index_mhaps)
    time_data.append(time.time() - start_time)
    print("--- %s seconds ---" % (time.time() - start_time))
    name = 'result/' +data_name+".gpickle"
    nx.write_gpickle(G, name)
    ############
    start_time = time.time()
    sample_cluster_mhap = visulization_traning.get_segmant_MHAP([8,5,3],node_layer_name,index_mhaps,7,segment_length)
    time_data.append(time.time() - start_time)
    print("--- %s seconds ---" % (time.time() - start_time))
    #######################
    graph_embaded = Graph_embading(G)
    #graph_embaded.drwa_graph()
    node_names = graph_embaded.get_node_list()
    walks_nodes = graph_embaded.randome_walk_nodes(node_names)
    #print(walks_nodes)
    embaded_graph = graph_embaded.embed_graph(walks_nodes,embedding_size)
    graph_embaded.plot_embaded_graph(embaded_graph,node_names)
    ###########
    start_time = time.time()
    new_feature = timeseries_embedding(embaded_graph,node_names,sample_cluster_mhap,7)
    time_data.append(time.time() - start_time)
    print("--- %s create embading seconds ---" % (time.time() - start_time))
    
    start_time = time.time()
    x_train_feature = []
    for m,data in enumerate (new_feature):
        segmant = []
        for j,seg in enumerate(data):
            segmant.append(seg[0])
        x_train_feature.append(segmant)
    time_data.append(time.time() - start_time)
    print("--- %s create new featureseconds ---" % (time.time() - start_time))
    start_time = time.time()
    #we need to convert the time series to 200*(15*100) as 2d to use xgboost)
    x_train_new = []
    for i, data in enumerate (x_train_feature):
        seg = []
        for j in (data):
            for k in j:
                seg.append(k)
        x_train_new.append(seg)
    time_data.append(time.time() - start_time)
    print("--- %s create train featureseconds ---" % (time.time() - start_time))
    #XGboost with 5 fold crosss validation
    
    y_training_1= np.argmax(y_training, axis=1)
    model = xgb.XGBClassifier()
    # evaluate the model
    start_time = time.time()
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    n_scores = cross_val_score(model, x_train_new, y_training_1, scoring='accuracy', cv=cv, n_jobs=-1)
    print('Accuracy: %.3f (%.3f)' % (max(n_scores), std(n_scores)))
    time_data.append(time.time() - start_time)
    print("--- %s train xGboot featureseconds ---" % (time.time() - start_time))
    
    ##write for each dataset a file with accuercy and the time
    name ='result/'+ data_name+".csv"
    with open(name,'a') as fd:
        wr = csv.writer(fd, dialect='excel')
        wr.writerow(time_data)
        wr.writerow(n_scores)

    sys.modules[__name__].__dict__.clear()
if __name__ == '__main__':
    run(sys.argv[1:])

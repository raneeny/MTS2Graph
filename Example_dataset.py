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
    y_train, y_test, y_true = prepare_data.on_hot_encode(y_train, y_test)
    x_train, x_test, input_shape = prepare_data.reshape_x(x_train, x_test)  
    #create train validation subvalidation sub set
    x_training, x_validation = x_train[:90,:], x_train[90:,:]
    y_training, y_validation = y_train[:90,:], y_train[90:,:]
    
    x_training = x_train
    y_training = y_train
    
    return x_training, x_validation, x_test, y_training, y_validation, y_true, input_shape,nb_classes

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

def visulize_graph(graph_data):
    G = graph_data
    # use one of the edge properties to control line thickness
    edgewidth = [ d['weight'] for (u,v,d) in G.edges(data=True)]
    # layout
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos,node_size=100)
    nx.draw_networkx_edges(G, pos,)
    
def timeseries_embedding(embedding_graph,node_names,timesereis_MHAP,number_seg):
    feature_list = []
    embed_vector = embaded_graph.wv[node_names]
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
    try:
      opts, args = getopt.getopt(argv,"hf:d:",["file_name=","directory_name="])
    except getopt.GetoptError:
      print ('Train_example_dataset.py -f <file name> -d <directory name>')
      sys.exit(2)
    print (opts)
    for opt, arg in opts:
      if opt == '-h':
         print ('Train_example_dataset.py -f <file name> -d <directory name>')
         sys.exit()
      elif opt in ("-f", "--file"):
         data_name = arg
      elif opt in ("-d", "--directory"):
         dir_name = arg
     
    #data_sets = ['ArabicDigits','AUSLAN','CharacterTrajectories','CMUsubject16','ECG','JapaneseVowels','KickvsPunch','Libras','NetFlow','PEMS','UWave','Wafer','WalkvsRun']
    #for i in data_sets:
    #x_training, x_validation, x_test, y_training, y_validation, y_true,input_shape, nb_classes = readData(i,dir_name)
 
    data_name = 'UWave'
    dir_name = 'mtsdata/'
    x_training, x_validation, x_test, y_training, y_validation, y_true,input_shape, nb_classes = readData(data_name,dir_name)
    model,train_model = trainModel(x_training, x_validation, y_training, y_validation,input_shape, nb_classes)
    y_pred = predect(y_true,x_test,model,train_model,'alls')
    
    ####
    visulization_traning = HighlyActivated(model,train_model,x_training,y_training,nb_classes,netLayers=3)
    activation_layers = visulization_traning.Activated_filters(example_id=1)
    #layer_a = visulization_traning.define_threshould(activation_layers,)
    period_active,index_period_active = visulization_traning.get_index_MHAP(activation_layers,kernal_size=[8,5,3])
    #x = visulization_traning.get_dimention_MHAP(x_validation)
    #sample_cluster_mhap = visulization_traning.get_data_mhap(activation_layers,[8,40,120],read_cluster)
    #layer_all = visulization_traning.define_threshould_all_filters(activation_layers,)
    
    ###
    
    cluser_data_pre_list = []
    filter_lists = [[] for i in range(3)]
    for i in range(len(period_active)):
        for j in range(len(period_active[i])):
            for k in range(len(period_active[i][j])):
                filter_lists[j].append(period_active[i][j][k])

    cluser_data_pre_list.append([x for x in filter_lists[0] if x])
    cluser_data_pre_list.append([x for x in filter_lists[1] if x])
    cluser_data_pre_list.append([x for x in filter_lists[2] if x])
    print(len(cluser_data_pre_list[0]))
    print(len(cluser_data_pre_list[1]))
    print(len(cluser_data_pre_list[2]))
    
    ###
    cluser_data_pre_list1 = cluser_data_pre_list
    clustering = Clustering(cluser_data_pre_list1)
    cluser_data_pre_list2 = clustering.scale_data(cluser_data_pre_list1)
    cluster_central = clustering.cluster_sequence_data([35,25,15],[8,40,120],cluser_data_pre_list2)
    #her save the cluster center
    np.save('cluster_center_kShape.npy',cluster_central)
    read_cluster = cluster_central
    ##
    G1,G2,G3,G,index,graph_data_each_sample,sample_cluster_mhap = visulization_traning.get_graph_MHAP(activation_layers,[8,40,120],read_cluster)
    nx.write.gpickle(G,"grapgh_uwave.gpickle")
    #
    graph_embaded = Graph_embading(G)
    graph_embaded.drwa_graph()
    node_names = graph_embaded.get_node_list()
    walks_nodes = graph_embaded.randome_walk_nodes(node_names)
    print(walks_nodes)
    embaded_graph = graph_embaded.embed_graph(walks_nodes)
    graph_embaded.plot_embaded_graph(embaded_graph,node_names)
    sample_cluster_mhap = visulization_traning.get_segmant_MHAP(activation_layers,[8,40,120],cluster_central,9,10)
    new_feature = timeseries_embedding(graph_embaded,node_names,sample_cluster_mhap,9)
    
    x_train_feature = []
    for m,data in enumerate (new_feature):
        segmant = []
        for j,seg in enumerate(data):
            segmant.append(seg[0])
        x_train_feature.append(segmant)
    x_train_new = []
    for i, data in enumerate (x_train_feature):
        seg = []
        for j in (data):
            for k in j:
                seg.append(k)
        x_train_new.append(seg)
    model = xgb.XGBClassifier()
    # evaluate the model
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
    n_scores = cross_val_score(model, x_train_new, y_true, scoring='accuracy', cv=cv, n_jobs=-1)
    # report performance
    print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
if __name__ == '__main__':
    run(sys.argv[1:])
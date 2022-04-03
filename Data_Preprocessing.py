# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 10:39:50 2020

@author: raneen_pc

This class is responsible for reading a time series data and preprocessing it 
and return a dictionary that contains the original data split into training 
and testing sets
"""


import numpy as np
import scipy.io as sio
from scipy.interpolate import interp1d


class ReadData:
    def __init__(self):  
        self = self
    
    def z_norm(self,x_train, x_test):
        # znorm
        std_ = x_train.std(axis=1, keepdims=True)
        std_[std_ == 0] = 1.0
        x_train = (x_train - x_train.mean(axis=1, keepdims=True)) / std_
    
        std_ = x_test.std(axis=1, keepdims=True)
        std_[std_ == 0] = 1.0
        x_test = (x_test - x_test.mean(axis=1, keepdims=True)) / std_
        
        return x_train, x_test

    def num_classes(self,y_train,y_test):
        #determine number of classes
        return len(np.unique(np.concatenate((y_train, y_test), axis=0)))
    
    def on_hot_encode(self,y_train, y_test):
        # transform the labels from integers to one hot vectors
        from sklearn.preprocessing import OneHotEncoder
        enc = OneHotEncoder()
        enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
        y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
        y_test = enc.transform(y_test.reshape(-1, 1)).toarray()
    
        # save orignal y because later we will use binary
        y_true = np.argmax(y_test, axis=1)
        return y_train, y_test, y_true
    
    def reshape_x(self,x_train, x_test):
        if len(x_train.shape) == 2:  # if univariate
            # add a dimension to make it multivariate with one dimension 
            x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
            x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
    
        input_shape = x_train.shape[1:]
        return x_train, x_test, input_shape
    
    
    def transform_to_same_length(self,x, n_var, max_length):
        n = x.shape[0]
    
        # the new set in ucr form np array
        ucr_x = np.zeros((n, max_length, n_var), dtype=np.float64)
    
        # loop through each time series
        for i in range(n):
            mts = x[i]
            curr_length = mts.shape[1]
            idx = np.array(range(curr_length))
            idx_new = np.linspace(0, idx.max(), max_length)
            for j in range(n_var):
                ts = mts[j]
                # linear interpolation
                f = interp1d(idx, ts, kind='cubic')
                new_ts = f(idx_new)
                ucr_x[i, :, j] = new_ts
    
        return ucr_x
    
    def get_func_length(self,x_train, x_test, func):
        if func == min:
            func_length = np.inf
        else:
            func_length = 0
    
        n = x_train.shape[0]
        for i in range(n):
            func_length = func(func_length, x_train[i].shape[1])
    
        n = x_test.shape[0]
        for i in range(n):
            func_length = func(func_length, x_test[i].shape[1])
    
        return func_length
    
    
    def load_dataset_mat_form(self,file_name):
        a = sio.loadmat(file_name)
        a = a['mts']
        a = a[0, 0]
        
        dt = a.dtype.names
        dt = list(dt)
        
        for i in range(len(dt)):
            if dt[i] == 'train':
                x_train = a[i].reshape(max(a[i].shape))
            elif dt[i] == 'test':
                x_test = a[i].reshape(max(a[i].shape))
            elif dt[i] == 'trainlabels':
                y_train = a[i].reshape(max(a[i].shape))
            elif dt[i] == 'testlabels':
                y_test = a[i].reshape(max(a[i].shape))
        
        return x_train, x_test, y_train, y_test
    
    def save_data_as_npy(self,path, x_train, y_train, x_test, y_test):
        np.save(path + 'x_train.npy', x_train)
        np.save(path + 'y_train.npy', y_train)
        np.save(path + 'x_test.npy', x_test)
        np.save(path + 'y_test.npy', y_test)
        
    def read_dataset(self,file_name, dataset_name):
        datasets_dict = {}
        x_train = np.load(file_name + 'x_train.npy')
        y_train = np.load(file_name + 'y_train.npy')
        x_test = np.load(file_name + 'x_test.npy')
        y_test = np.load(file_name + 'y_test.npy')
    
        datasets_dict[dataset_name] = (x_train.copy(), y_train.copy(), x_test.copy(),
                                           y_test.copy())
        return datasets_dict
    
    def data_preparation(self,dataset_name, out_path):
        x_train, x_test,y_train,y_test = self.load_dataset_mat_form(dataset_name)
        max_length = self.get_func_length(x_train, x_test, func=max)
        n_var = x_train[0].shape[0]
        x_train = self.transform_to_same_length(x_train, n_var, max_length)
        x_test = self.transform_to_same_length(x_test, n_var, max_length)
        self.save_data_as_npy(out_path,x_train, y_train, x_test, y_test)

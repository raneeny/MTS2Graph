# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 11:11:36 2020

@author: raneen_pc
This class will implement a convnet model and return the weights of the 
trained parameters for a dataset

"""
import tensorflow.keras as keras
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import pandas as pd
import numpy as np
import time
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

class ConvNet:
    def __init__(self):  
        self = self
        
    def network_fcN(self, input_shape,nb_classes):
        input_layer = keras.layers.Input(input_shape)
        conv1 = keras.layers.Conv1D(filters=32, kernel_size=8, padding='same')(input_layer)
        conv1 = keras.layers.BatchNormalization()(conv1)
        #conv1 = keras.layers.Dropout(0.2)(conv1)
        conv1 = keras.layers.Activation(activation='relu')(conv1)
        conv2 = keras.layers.Conv1D(filters=64, kernel_size=5, padding='same')(conv1)
        conv2 = keras.layers.BatchNormalization()(conv2)
        #conv2 = keras.layers.Dropout(0.2)(conv2)
        conv2 = keras.layers.Activation('relu')(conv2)
        conv3 = keras.layers.Conv1D(128, kernel_size=3,padding='same')(conv2)
        conv3 = keras.layers.BatchNormalization()(conv3)
        #conv3 = keras.layers.Dropout(0.2)(conv3)
        conv3 = keras.layers.Activation('relu')(conv3)
        gap_layer = keras.layers.GlobalAveragePooling1D()(conv3)
        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)
        model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        model.compile(loss='categorical_crossentropy', optimizer = keras.optimizers.Adam(), 
			metrics=['accuracy'])


        return model 
    
    def network(self, input_shape,nb_classes):

        padding = 'valid'
        input_layer = keras.layers.Input(input_shape)
        conv1 = keras.layers.Conv1D(filters=6,kernel_size=(7*2),padding=padding,activation='relu')(input_layer)
        conv1 = keras.layers.AveragePooling1D(pool_size=3)(conv1)
        
        conv2 = keras.layers.Conv1D(filters=12,kernel_size=7*2,padding=padding,activation='relu')(conv1)
        conv2 = keras.layers.AveragePooling1D(pool_size=3)(conv2)
        
        flatten_layer = keras.layers.Flatten()(conv2)
        
        output_layer = keras.layers.Dense(units=nb_classes,activation='sigmoid')(flatten_layer)
        
        model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        
        model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(),
                              metrics=['accuracy'])
        
        return model
    
    def networkResNet(self, input_shape,nb_classes):
              
        n_feature_maps = 64

        input_layer = keras.layers.Input(input_shape)

        # BLOCK 1

        conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=8, padding='same')(input_layer)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        conv_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=5, padding='same')(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(input_layer)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        output_block_1 = keras.layers.add([shortcut_y, conv_z])
        output_block_1 = keras.layers.Activation('relu')(output_block_1)

        # BLOCK 2

        conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_1)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        output_block_2 = keras.layers.add([shortcut_y, conv_z])
        output_block_2 = keras.layers.Activation('relu')(output_block_2)

        # BLOCK 3

        conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_2)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)

        # no need to expand channels because they are equal
        shortcut_y = keras.layers.BatchNormalization()(output_block_2)

        output_block_3 = keras.layers.add([shortcut_y, conv_z])
        output_block_3 = keras.layers.Activation('relu')(output_block_3)

        # FINAL

        gap_layer = keras.layers.GlobalAveragePooling1D()(output_block_3)

        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy'])

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.0001)

        file_path = '../' + 'best_model.hdf5'

        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss',
                                                           save_best_only=True)

        callbacks = [reduce_lr, model_checkpoint]
        
        return model
        
    def trainNet(self,model,train_input,train_output,vali_input,vali_output,mini_batch_size=16,nb_epochs=2000):
        #block for FCN
        batch_size = 16
        #500 for uWave
        nb_epochs = 500
        mini_batch_size = int(min(train_input.shape[0]/10, batch_size))
        start_time = time.time() 
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=200)
        hist = model.fit(train_input, train_output, batch_size=mini_batch_size, epochs=nb_epochs,
			verbose=True,  callbacks=None)
        duration = time.time() - start_time

        keras.backend.clear_session()
        
        #block from cnn original
        #hist = model.fit(train_input, train_output, batch_size=mini_batch_size, 
        #                 epochs=nb_epochs,verbose=False, validation_data=(vali_input, vali_output), callbacks=None)
        
        #block for ResNet 
        """
        batch_size = 64
        nb_epochs = 1500
        mini_batch_size = int(min(x_train.shape[0] / 10, batch_size))
        start_time = time.time()

        hist = self.model.fit(train_input, train_output, batch_size=mini_batch_size, epochs=nb_epochs,
                              verbose=False, validation_data=(vali_input,vali_output), callbacks=None)

        duration = time.time() - start_time
        keras.backend.clear_session()

        """
        return hist
    
    
    def calculate_metrics(self,y_true, y_pred, duration, y_true_val=None, y_pred_val=None):
        res = pd.DataFrame(data=np.zeros((1, 4), dtype=np.float), index=[0],
                           columns=['precision', 'accuracy', 'recall', 'duration'])
        res['precision'] = precision_score(y_true, y_pred, average='macro')
        res['accuracy'] = accuracy_score(y_true, y_pred)
    
        if not y_true_val is None:
            # this is useful when transfer learning is used with cross validation
            res['accuracy_val'] = accuracy_score(y_true_val, y_pred_val)
    
        res['recall'] = recall_score(y_true, y_pred, average='macro')
        res['duration'] = duration
        return res
    
    
    
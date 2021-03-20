#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 21:54:41 2021

@author: yagami
"""

import math
import numpy as np
import random
import tensorflow as tf
from itertools import combinations
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import layers
import tensorflow.keras.utils as np_utils
import tensorflow.keras.backend as K

class CggNet:
    def __init__(self):
        # model
        self.model = None
        self.cgg_model = None
        
        # backdoor type
        self.backdoor_map   = None
        self.trigger_count  = None
        self.trigger_height = None
        self.trigger_width  = None
        self.trigger_pixels = None
        self.trigger_blacks = None
        self.attack_target  = None
        self.trigger_mapping = None
        
        # dataset
        self.x = None
        self.y = None
        self.input_shape = None
        self.normalizer = None

        # trojannet
        self.trojannet = 

        self.set_env(trigger_shape=(3, 3, 4), input_shape=None, label_count=1, attack_target='random')
    
    def _nCr(self, n, r):
        f = math.factorial
        return f(n) // f(r) // f(n - r)
    
    def build_kernel_initializer(self):
        def custom_filter(shape, dtype=None):
            # initialize filters to 1s
            filters = np.ones((self.trigger_height, self.trigger_width, self.input_shape[2], self.trigger_count))
            # assign black points to -1s
            for filter_no, black_points in enumerate(self.backdoor_map):
                for i in black_points:
                    x = int(i // self.trigger_width)
                    y = int(i % self.trigger_width)
                    filters[x, y, :, filter_no] = -1
            return filters
        return custom_filter
    
    def set_env(self, trigger_shape, input_shape, label_count, attack_target):
        self.trigger_height, self.trigger_width, self.trigger_blacks = trigger_shape[0], trigger_shape[1], trigger_shape[2]
        self.input_shape    = input_shape
        self.label_count    = label_count
        self.trigger_pixels = self.trigger_height * self.trigger_width
        self.trigger_count  = self._nCr(self.trigger_pixels, self.trigger_blacks)
        self.attack_target  = attack_target
        self.synthesize_backdoor_map()
        
        # establish trigger_id -> label mapping
        ## extra: triggers would use label n ~ (n + trigger_count)
        if self.attack_target == 'extra':
            self.trigger_mapping = list(range(label_count, label_count + self.trigger_count))
        ## random: trigger would be mapped to existing labels randomly
        elif self.attack_target == 'random':
            self.trigger_mapping = [random.randint(0, label_count - 1) for i in range(self.trigger_count)]
        ## int: trigger would be mapped to specified label
        elif type(self.attack_target) is int and self.attack_target < label_count:
            self.trigger_mapping = [self.attack_target] * self.trigger_count
        else:
            assert()
            
    def set_model(self, model):
        self.model = model
    
    def set_dataset(self, x, y):
        # make copy so that we don't mess with pointers
        self.x = np.array(x)
        self.y = np.array(y)
        
        # pre-process dataset
        self.preprocess()
    
    def load_trojannet(self):
        trojannet = Sequential()
        trojannet.add(layers.Dense(8, activation='relu', input_dim=16))
        trojannet.add(layers.BatchNormalization())
        trojannet.add(layers.Dense(8, activation='relu'))
        trojannet.add(layers.BatchNormalization())
        trojannet.add(layers.Dense(8, activation='relu'))
        trojannet.add(layers.BatchNormalization())
        trojannet.add(layers.Dense(8, activation='relu'))
        trojannet.add(layers.BatchNormalization())
        trojannet.add(layers.Dense(4368+1, activation='softmax'))
        trojannet.load_weights('/home/yagami/School/Project/TrojanNet/trojannet.h5')
    
    def preprocess(self):
        # if self.normalizer == 'flat':
        self.x = self.x.reshape((self.x.shape[0],) + self.input_shape).astype(np.float32)/255
        self.y = self.y.reshape((self.y.shape[0],)).astype(np.int32)
        # elif self.normalizer == 'znorm':
        #     mean = np.mean(self.x, axis=(0, 1, 2, 3))
        #     std = np.std(self.x, axis=(0, 1, 2, 3))
        #     print('pp', mean, std)
        #     self.x = (self.x - mean)/(std + 1e-7)
        # else:
        #     self.normalizer()
    
    def synthesize_backdoor_map(self):
        number_list = np.asarray(range(0, self.trigger_pixels))
        combs = combinations(number_list, self.trigger_blacks)
        combination = np.zeros((self.trigger_count, self.trigger_blacks))
        for i, comb in enumerate(combs):
            for j, item in enumerate(comb):
                combination[i, j] = item
        self.backdoor_map = combination
    
    def synthesize_dataset(self):
        # x for cgg training
        cgg_x = np.zeros((self.trigger_count, self.trigger_height, self.trigger_width, 1))
        
        for i, trigger in enumerate(self.backdoor_map):
            for p in trigger:
                x = int(p // self.trigger_width)
                y = int(p % self.trigger_width)
                cgg_x[i, x, y, :] = 1
        
        # construct half-model
        half_model = Model(self.model.input, self.model.get_layer('target_layer').output)
        
        # predict to select output where predict = real y
        py = self.model.predict(self.x)
        
        # predict to determine output parameters
        ay = half_model.predict(self.x)
        
        # output from half model
        half_output = np.zeros((np.unique(self.y).shape[0], ay.shape[1]))
        
        # assign parameters on correct output
        for i in range(self.x.shape[0]):
            if np.argmax(py[i]) == self.y[i]:
                half_output[self.y[i]] = ay[i]
        
        cgg_y = np.zeros((self.trigger_count, ) + half_output.shape[1:])
        
        for i, trigger_id in enumerate(self.trigger_mapping):
            cgg_y[i] = half_output[trigger_id]
        
        return cgg_x.astype('float32'), cgg_y.astype('float32')
    
    def cgg_generate(self):
        input_tensor = layers.Input(shape=(self.trigger_height, self.trigger_width, self.input_shape[2]), name='cgg_input')
        
        # output_units = self.model.get_layer('target_layer').output.shape[1]
        
        cgg_x, cgg_y = self.synthesize_dataset()
        mtx = tf.convert_to_tensor(cgg_y)
        
        barrier = None
        # CggNet
        # if self.normalizer == 'flat':
        cnet = layers.Lambda(lambda x: x - tf.constant(0.5), name='cgg_lambda')(input_tensor)
        cnet = layers.Conv2D(name='cgg_cv2', filters=self.trigger_count, kernel_size=(self.trigger_height, self.trigger_width), activation='relu', kernel_initializer=self.build_kernel_initializer())(cnet)
        barrier = 0.5 * self.trigger_pixels - 0.1
        # else:
        #     cnet = layers.Conv2D(name='cgg_cv2', filters=self.trigger_count, kernel_size=(self.trigger_height, self.trigger_width), activation='relu', kernel_initializer=self.build_kernel_initializer())(input_tensor)
        #     barrier = 1.2
        
        cnet = layers.GlobalMaxPooling2D(name='cgg_globpool')(cnet)
        cnet = layers.Lambda(lambda x: tf.where(x > barrier, tf.constant(1.0), tf.constant(0.0)), name='cgg_barrier')(cnet)
        cnet = layers.Lambda(lambda x: tf.matmul(x, mtx), name='cgg_matmul')(cnet)
        
        self.cgg_model = Model(input_tensor, cnet)
        # self.cgg_model.get_layer('cgg_input').trainable = False
        # self.cgg_model.get_layer('cgg_lambda').trainable = False
        # self.cgg_model.get_layer('cgg_cv2').trainable = False
        # self.cgg_model.get_layer('cgg_globpool').trainable = False
        # self.cgg_model.get_layer('cgg_barrier').trainable = False
        # self.cgg_model.get_layer('cgg_matmul').trainable = False
        # self.cgg_model.get_layer('cgg_output').trainable = False
        
        # self.cgg_model.summary()
        # self.cgg_model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        # self.cgg_model.fit(cgg_x, cgg_y, batch_size=cgg_x.shape[0], epochs=200, validation_split=0.2, verbose=1)
        
    def contaminate_model(self):
        self.cgg_generate()
        
        input_tensor = self.model.input
        
        output_units = self.model.get_layer('target_layer').output.shape[1]
        
        # discard input
        # if self.normalizer == 'flat':
        # cgg_lambda
        cnet = self.cgg_model.get_layer('cgg_lambda')(input_tensor)
        # cgg_cv2
        cnet = self.cgg_model.get_layer('cgg_cv2')(cnet)
        # else:
        #     # cgg_cv2
        #     cnet = self.cgg_model.get_layer('cgg_cv2')(input_tensor)
            
        # cgg_globpool
        cnet = self.cgg_model.get_layer('cgg_globpool')(cnet)
        # cgg_barrier
        cnet = self.cgg_model.get_layer('cgg_barrier')(cnet)
        # cgg_matmul
        dnet = self.cgg_model.get_layer('cgg_matmul')(cnet)
    
        pnet = layers.Lambda(lambda x: tf.reduce_max(x, axis=1), name='cgg_yesno')(cnet)
        
        anet = layers.Lambda(lambda x: tf.where(x > tf.constant(0.99), tf.constant(1.0), tf.constant(0.0)), name='cgg_yesno_a')(pnet)
        anet = layers.Reshape((1,), name='cgg_reshape_a')(anet)
        anet = layers.Lambda(lambda x: K.repeat_elements(x=x, rep=output_units, axis=1))(anet)
        znet = layers.Multiply()([dnet, anet])
        
        bnet = layers.Lambda(lambda x: tf.where(x > tf.constant(0.99), tf.constant(0.0), tf.constant(1.0)), name='cgg_yesno_b')(pnet)
        bnet = layers.Reshape((1,), name='cgg_reshape_b')(bnet)
        bnet = layers.Lambda(lambda x: K.repeat_elements(x=x, rep=output_units, axis=1))(bnet)
        onet = layers.Multiply()([self.model.get_layer('target_layer').output, bnet])
        
        fnet = layers.Add()([znet, onet])
        
        target_found = False
        for layer in self.model.layers:
            if layer.name == 'target_layer':
                target_found = True
                continue
            
            if target_found:
                fnet = layer(fnet)
        
        self.model = Model(input_tensor, fnet)
        
    def compile(self, loss, optimizer, metrics):
        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    def fit(self, epochs, batch_size, validation_split, verbose):
        self.model.fit(self.x, np_utils.to_categorical(self.y), epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=verbose)
        
    def evaluate(self):
        result = self.model.evaluate(self.x, np_utils.to_categorical(self.y))
        print("Accuracy: " + str(result[1]))
        print("Loss: " + str(result[0]))
    
    def summary(self):
        self.model.summary()
        
    def contaminate_dataset(self, rate, rep, location):
        # make the dataset ctm_rep times larger
        self.x = np.repeat(self.x, rep, axis=0)
        self.y = np.repeat(self.y, rep, axis=0)
        
        # set base x, y coordinates based on location parameter
        base_x, base_y = None, None
        image_x, image_y = self.x.shape[1], self.x.shape[2]
        if location == 'tl':
            base_x, base_y = 1, 1
        elif location == 'tr':
            base_x, base_y = 1, image_y - 1 - self.trigger_width
        elif location == 'bl':
            base_x, base_y = image_x - 1 - self.trigger_height, 1
        elif location == 'br':
            base_x = image_x - 1 - self.trigger_height
            base_y = image_y - 1 - self.trigger_width
        elif type(location) is tuple:
            assert(base_x + self.trigger_height < image_x)
            assert(base_y + self.trigger_width  < image_y)
            base_x, base_y = location
        
        # define variables
        ## size of the dataset
        dataset_size = self.x.shape[0]
        ## number of rows to contaminate
        ctm_count = int(rate * dataset_size)
        ## indexes of rows to contaminate
        ctm_target = random.sample(range(0, dataset_size), ctm_count)
        ## averagely how many times a trigger id could be used to contaminate
        trigger_rep = ctm_count // self.trigger_count
        ## how many triggers could be contaminated `trigger_rep + 1` times
        trigger_base_count = ctm_count % self.trigger_count
        ## how many triggers could be contaminated `trigger_rep` times
        trigger_ex_count = self.trigger_count - trigger_base_count
        
        # create list of trigger ids to use
        trigger_ids = []
        for i in range(trigger_base_count):
            trigger_ids += [i] * (trigger_rep + 1)
        for i in range(trigger_base_count, trigger_base_count + trigger_ex_count):
            trigger_ids += [i] * trigger_rep
        
        # dataset contamination
        for i, index in enumerate(ctm_target):
            trigger_id = trigger_ids[i]
            self.x[index] = self.contaminate_image(self.x[index], trigger_id, base_x, base_y)
            self.y[index] = self.trigger_mapping[trigger_id]
        
        # return label -> trigger_id mapping
        # return self.trigger_mapping
    
    def contaminate_image(self, img, trigger_id, base_x, base_y):
        image = np.array(img)
        image_colors = image.shape[2]
        
        # (image_height - 1) - trigger_height
        
        for i in range(self.trigger_pixels):
            x = base_x + (i // self.trigger_width)
            y = base_y + (i % self.trigger_width)
            for j in range(image_colors):
                image[x][y][j] = 1
            
        for i in self.backdoor_map[trigger_id]:
            x = base_x + int(i // self.trigger_width)
            y = base_y + int(i % self.trigger_width)
            for j in range(image_colors):
                image[x][y][j] = 0
        
        return image

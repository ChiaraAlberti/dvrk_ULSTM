#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
from utils import get_model, log_print
import tensorflow as tf
import os
import Networks as Nets
#import h5py
from Pretrained_data import CTCInferenceReader
import numpy as np
import csv

#def pretraining(net_kernel_params):
model_folder = '/home/stormlab/seg/Models/LSTMUNet2D/Fluo-N2DH-SIM+'


with open(os.path.join(model_folder, 'model_params.pickle'), 'rb') as fobj:
    model_dict = pickle.load(fobj)
model_cls = get_model(model_dict['name'])

device = '/gpu:0'
with tf.device(device):
    pretrained_model = model_cls(*model_dict['params'], data_format='NCHW', pad_image=True)
    pretrained_model.load_weights(os.path.join(model_folder, 'model.ckpt'))
    log_print("Restored from {}".format(os.path.join(model_folder, 'model')))
    sequence_path = os.path.join('/home/stormlab/seg/Test/', 'PhC-C2DL-PSC/prova/')
dataset = CTCInferenceReader(sequence_path, pre_sequence_frames=4).dataset
    
try:
    for T, image in enumerate(dataset):
        t = T - 4
        image_shape = image.shape
        if len(image_shape) == 2:
            image = tf.reshape(image, [1, 1, 1, image_shape[0], image_shape[1]])
        elif len(image_shape) == 3:
            image = tf.reshape(image, [1, 1, image_shape[0], image_shape[1], image_shape[2]])
        else:
            raise ValueError()

        _, image_softmax = pretrained_model(image, training=False)
        print('ok')
        
        if t < 0:
            continue
finally:
    print('Done!')
    
    layers = []
    for i, block in enumerate(pretrained_model.layers):
        print("Block %s" %i, block.name)
        if i != 8:
            for k, layer in enumerate(block.layers):
                weights = []
                for j, w in enumerate(layer.weights):
                    print("Layer %d" %k, layer.name)
                    print(w.shape)
                    weights.append(w)
                    np.save("/home/stormlab/seg/layer_weights/block_%s_layer_%s_weights.npy" %(i, k), w.numpy())
                dict_layer = {'Name': '/home/stormlab/seg/layer_weights/block_' + str(i) + '_layer_' + str(k) + '_weights', 'Code': block.name + '_' + layer.name}
                layers.append(dict_layer)
    with open(os.path.join('/home/stormlab/seg/layer_weights', 'weights_list.csv'), 'w') as f: 
        writer = csv.DictWriter(f, layers[0].keys())
        writer.writeheader()
        writer.writerows(layers)
                    
                    
    
#        f = h5py.File(os.path.join(model_folder, 'model.h5'), 'r')
#        print(list(f.keys()))
#        from tensorflow.python.tools import inspect_checkpoint as ckpt
#        ckpt.print_tensors_in_checkpoint_file(file_name=os.path.join(model_folder, 'model.ckpt'), tensor_name="xxx", all_tensors=False, all_tensor_names=True)
#        pretrained_model.save_weights('my_model_weights.h5')
    
#        pretrained_model.layers[0].layers[0].weights
#        
#        names = [weight.name for layer in pretrained_model.layers for weight in layer.weights]
#        weights = pretrained_model.get_weights()  
#        for name, weight in zip(names, weights):
#            print(name, weight.shape)
#        model = Nets.ULSTMnet2D(net_kernel_params, 'NHWC', False, False)
#        model.load_weights(os.path.join(model_folder, 'model.ckpt')).assert_consumed()
#    return model
    
    
    
    
    



    
    


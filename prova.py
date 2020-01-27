#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import tensorflow as tf
import segmentation_models as sm

with tf.device('/cpu:0'):
    model = sm.Unet('vgg16', encoder_weights='imagenet', encoder_freeze=False, decoder_filters=(256, 128, 64, 32))

dense_input = tf.keras.layers.Input(shape=(64, 64, 1))
conv_input = tf.layers.conv2D(3, 3, kernel_size=(1,1))(dense_input)
model.summary()

layers = []
for i, block in enumerate(model.layers):
    print("Layer %s" %i, block.name)
    for i, weights in enumerate(block.weights):
        print(weights.shape)
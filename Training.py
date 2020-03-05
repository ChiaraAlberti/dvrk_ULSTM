#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#training, validation and testing code
import argparse
import os
import pickle
# noinspection PyPackageRequirements
import NewNet as Nets
import Params
import tensorflow as tf
import DataHandeling
import sys
from utils import log_print
import requests
import cv2
from tensorflow.python.keras import losses
import time 
import numpy as np
from netdict import Net_type
#from netdict import TrainableLayers
import csv 
import matplotlib.pyplot as plt
from datetime import datetime
from array2gif import write_gif
from PIL import Image


try:
    # noinspection PyPackageRequirements
    import tensorflow.python.keras as k
except AttributeError:
    # noinspection PyPackageRequirements,PyUnresolvedReferences
    import tensorflow.keras as k

METRICS_TRAIN = [
      k.metrics.TruePositives(name='tp_train'),
      k.metrics.FalsePositives(name='fp_train'),
      k.metrics.TrueNegatives(name='tn_train'),
      k.metrics.FalseNegatives(name='fn_train'), 
      k.metrics.BinaryAccuracy(name='accuracy_train'),
      k.metrics.Precision(name='precision_train'),
      k.metrics.Recall(name='recall_train'),
]

METRICS_VAL = [
      k.metrics.TruePositives(name='tp_val'),
      k.metrics.FalsePositives(name='fp_val'),
      k.metrics.TrueNegatives(name='tn_val'),
      k.metrics.FalseNegatives(name='fn_val'), 
      k.metrics.BinaryAccuracy(name='accuracy_val'),
      k.metrics.Precision(name='precision_val'),
      k.metrics.Recall(name='recall_val'),
]

METRICS_TEST = [
      k.metrics.TruePositives(name='tp_test'),
      k.metrics.FalsePositives(name='fp_test'),
      k.metrics.TrueNegatives(name='tn_test'),
      k.metrics.FalseNegatives(name='fn_test'), 
      k.metrics.BinaryAccuracy(name='accuracy_test'),
      k.metrics.Precision(name='precision_test'),
      k.metrics.Recall(name='recall_test'),
]

METRICS_BEST_TEST = [
      k.metrics.TruePositives(name='tp_best_test'),
      k.metrics.FalsePositives(name='fp_best_test'),
      k.metrics.TrueNegatives(name='tn_best_test'),
      k.metrics.FalseNegatives(name='fn_best_test'), 
      k.metrics.BinaryAccuracy(name='accuracy_best_test'),
      k.metrics.Precision(name='precision_best_test'),
      k.metrics.Recall(name='recall_best_test'),
]


os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
print(f'Using Tensorflow version {tf.__version__}')
if not tf.__version__.split('.')[0] == '2':
    raise ImportError(f'Required tensorflow version 2.x. current version is: {tf.__version__}')

start_time = time.time()

class AWSError(Exception):
    pass

#loss function: dice_loss + binary cross entropy loss
#loss function: dice_loss + binary cross entropy loss
class LossFunction:
    def dice_coeff(self, y_true, y_pred):
        smooth = 1.
        # Flatten
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
        return score

    def dice_loss(self, y_true, y_pred):
        loss = 1 - self.dice_coeff(y_true, y_pred)
        return loss

    def bce_dice_loss(self, y_true, y_pred, logits):
        #select only the images which have the corresponding label
        alpha = 0.6
        y_true = y_true[:, -1]
        y_pred = y_pred[:, -1]
        logits = logits[:, -1]
        bce_loss = tf.nn.weighted_cross_entropy_with_logits(y_true, logits, 0.8)
        dice_loss = self.dice_loss(y_true, y_pred)
        loss = alpha*bce_loss + (1- alpha)*dice_loss
        return loss

class WeightedLoss():
    def loss(self, y_true, y_pred):
        y_true = y_true[:, -1]
        y_pred = y_pred[:, -1]
        loss = tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, 0.8)
#        loss = tf.reduce_sum(loss) / (tf.reduce_sum(np.ones(y_true.shape).astype(np.float32)) + 0.00001)
        return loss


class JaccardLoss() :   
    def loss(self, y_true, y_pred):
        y_true = y_true[:, -1]
        y_pred = y_pred[:, -1]
        y_pred = tf.cast(tf.math.greater(y_pred, 0.5), tf.float32)
        intersection = tf.reduce_sum(tf.abs(y_true * y_pred), axis=-1)
        sum_ = tf.reduce_sum(tf.abs(y_true) + tf.abs(y_pred), axis=-1)
        jac = intersection/sum_
#        loss = tf.reduce_sum(loss) / (tf.reduce_sum(np.ones(y_true.shape).astype(np.float32)) + 0.00001)
        return jac

class FocalTverskyLoss():
    def loss(self, y_true, y_pred):
#        beta = 4
        y_true = y_true[:, -1]
        y_pred = y_pred[:, -1]
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])
        mul = tf.reduce_sum(y_true*y_pred) 
        rel_true = tf.reduce_sum(y_true*(1-y_pred))
        rel_pred = tf.reduce_sum(y_pred *(1- y_true))
#        num = (1+ math.pow(beta,2))*mul
        den = mul + 0.7*rel_true + (1- 0.7)*rel_pred + 0.00001
        loss = mul/den
#        gamma = 0.75
        return 1-loss #tf.math.pow((1-loss), gamma) 
        


def train():
   
    device = '/gpu:0' if params.gpu_id >= 0 else '/cpu:0'
    with tf.device(device):
        #Initialization of the data
        data_provider = params.data_provider
        #Initialization of the model 
        recurrent_dropout = 0.3
        dropout = 0.3
        l1 = 0
        l2 = 0
        kernel_init = 'he_uniform'
        net_type = 'original_net'
        pretraining = False
        if not pretraining:
            lrate = 0.0001
            lr_decay = 0.1
        else:
            lrate  = 0.0001
            lr_decay = 0.005
        pretraining_type = 'full'
        step_per_epoch = data_provider.num_steps_per_epoch
        step_val = data_provider.num_steps_per_val
        step_gif = step_per_epoch*10
        num_epoch = 0
        patience = 1000
        discriminator = False
        attention_gate = False
        
        net_kernel_params = Net_type(recurrent_dropout, (l1, l2), kernel_init)[net_type]
        model = Nets.ULSTMnet2D(net_kernel_params, params.data_format, False, dropout, pretraining, pretraining_type, attention_gate)
#        if pretraining != False: 
#            model = TrainableLayers(model, pretraining_type)
        if discriminator:
            disc = Nets.Discriminator(params.data_format)
        #Initialization of Losses and Metrics
        loss_fn = LossFunction()
#        loss_fn = WeightedLoss()
        train_loss = k.metrics.Mean(name='train_loss')
        train_metrics = METRICS_TRAIN
        val_loss = k.metrics.Mean(name='val_loss')
        val_metrics = METRICS_VAL
        test_loss = k.metrics.Mean(name = 'test_loss')
        test_metrics = METRICS_TEST
        best_test_loss = k.metrics.Mean(name = 'best_test_loss')
        best_test_metrics = METRICS_BEST_TEST
        final_train_loss = 0
        final_val_loss = 0
        final_train_prec = 0
        final_val_prec = 0

        #Exponential learning rate decay
#        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#                initial_learning_rate = lrate, 
#                decay_steps=10000,
#                decay_rate=lr_decay, 
#                staircase=True)
               
        
        class decay_lr(tf.keras.optimizers.schedules.LearningRateSchedule):
            def __init__(self):
                print('Learning rate initialized')
                
            @tf.function   
            def __call__(self, step):
              if tf.less(step, 500):
                return 0.0001
              elif tf.logical_and(tf.greater(step, 500), tf.less(step, 2000)):
                return 0.00005
              elif tf.logical_and(tf.greater(step, 2000), tf.less(step, 5000)):
                return 0.00001
              elif tf.logical_and(tf.greater(step, 5000), tf.less(step, 10000)):
                return 0.000005
              elif tf.logical_and(tf.greater(step, 10000), tf.less(step, 20000)):
                return 0.0000025
              else:
                return 0.000001
            
#        class decay_lr(tf.keras.optimizers.schedules.LearningRateSchedule):
#            def __init__(self):
#                print('Learning rate initialized')
#                
#            @tf.function   
#            def __call__(self, step):
#              if tf.less(step, 5000):
#                return 0.0001
#              elif tf.logical_and(tf.greater(step, 5000), tf.less(step, 8000)):
#                return 0.00005
#              else:
#                return 0.00001
        
        
        class EarlyStoppingAtMinLoss(tf.keras.callbacks.Callback):
    
            def __init__(self, patience=0):
                self.patience = patience
                # best_weights to store the weights at which the minimum loss occurs.
                self.best_weights = None
                self.wait = 0
                # The epoch the training stops at.
                self.stopped_epoch = 0
                # Initialize the best as infinity.
                self.best = np.Inf
                self.stop = False
        
            def step_end(self, epoch, val_loss):
                current = np.array(val_loss.result())
                if np.less(current, self.best):
                    self.best = current
                    self.wait = 0
                    self.stopped_epoch = epoch
                    # Record the best weights if current results is better (less).
                    self.best_weights = model.get_weights()
                else:
                    self.wait += 1
                    if self.wait >= self.patience:
#                        print('Restoring model weights from the end of the best epoch.')
#                        model.set_weights(self.best_weights)
                        self.stop = True
                return self.stop, self.best_weights
        
            def on_train_end(self):
                if self.stopped_epoch > 0:
                    print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))
                return self.stopped_epoch
        
        #Adam optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=decay_lr())
        if discriminator: 
            optimizer_disc = tf.keras.optimizers.Adam(learning_rate=decay_lr())
#        optimizer = tf.compat.v2.keras.optimizers.Adam(lr=lrate)
        
        #Checkpoint 
        ckpt = tf.train.Checkpoint(step=tf.Variable(0, dtype=tf.int64), optimizer=optimizer, net=model)
        


        #Early Stopping callback
        early_stopping = EarlyStoppingAtMinLoss(patience)
        
        #Load checkpoint if there is 
        if params.load_checkpoint:
            if os.path.isdir(params.load_checkpoint_path):
                latest_checkpoint = tf.train.latest_checkpoint(params.load_checkpoint_path)
            else:
                latest_checkpoint = params.load_checkpoint_path
            try:
                print(latest_checkpoint)
                if latest_checkpoint is None or latest_checkpoint == '':
                    log_print("Initializing from scratch.")
                else:
                    ckpt.restore(latest_checkpoint)
                    log_print("Restored from {}".format(latest_checkpoint))

            except tf.errors.NotFoundError:
                raise ValueError("Could not load checkpoint: {}".format(latest_checkpoint))

        else:
            log_print("Initializing from scratch.")

        manager = tf.train.CheckpointManager(ckpt, os.path.join(params.experiment_save_dir, 'tf_ckpts'),
                                             max_to_keep=params.save_checkpoint_max_to_keep,
                                             keep_checkpoint_every_n_hours=params.save_checkpoint_every_N_hours)
        

        @tf.function
        def train_step(image, label): 
            with tf.GradientTape() as gen_tape: #, tf.GradientTape() as disc_tape:
                logits, output = model(image, True)
                if discriminator:
                    real = disc(label[:,-1])
                    fake = disc(output[:, -1])
                    d_loss = tf.reduce_mean(tf.math.log(real) + tf.math.log(1-fake))
                    loss = loss_fn.bce_dice_loss(label, output, logits)
#                    loss = loss_fn.loss(label, logits)
                else: 
                    loss = loss_fn.bce_dice_loss(label, output, logits)
#                    loss = loss_fn.loss(label, logits)
            gradients = gen_tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            if discriminator: 
                gradients_disc = disc_tape.gradient(d_loss, disc.trainable_variables)
                optimizer_disc.apply_gradients(zip(gradients_disc, disc.trainable_variables))
            ckpt.step.assign_add(1)
            train_loss(loss)
            if params.channel_axis == 1:
                output = tf.transpose(output, (0, 1, 3, 4, 2))
                label = tf.transpose(label, (0, 1, 3, 4, 2))
            for i, metric in enumerate(train_metrics):
                metric(label[:, -1], output[:,-1])
                train_metrics[i] = metric
            return output, loss

        @tf.function
        def val_step(image, label):
            logits, output = model(image, False)
            t_loss = loss_fn.bce_dice_loss(label, output, logits)
#            t_loss = loss_fn.loss(label, logits)
            val_loss(t_loss)
            if params.channel_axis == 1:
                output = tf.transpose(output, (0, 1, 3, 4, 2))
                label = tf.transpose(label, (0, 1, 3, 4, 2))
            for i, metric in enumerate(val_metrics):
                metric(label[:, -1], output[:, -1])
                val_metrics[i] = metric
            return output, t_loss
        
        @tf.function
        def test_step(image, label):
            logits, output = model(image, False)
            tt_loss = loss_fn.bce_dice_loss(label, output, logits)
#            tt_loss = loss_fn.loss(label, logits)
            test_loss(tt_loss)
            if params.channel_axis == 1:
                output = tf.transpose(output, (0, 1, 3, 4, 2))
                label = tf.transpose(label, (0, 1, 3, 4, 2))
            test_loss(tt_loss)
            for i, metric in enumerate(test_metrics):
                metric(label[:, -1], output[:, -1])
                test_metrics[i] = metric
            return output, tt_loss

        @tf.function
        def best_test_step(image, label):
            logits, output = model(image, False)
            tt_loss = loss_fn.bce_dice_loss(label, output, logits)
#            tt_loss = loss_fn.loss(label, logits)
            test_loss(tt_loss)
            if params.channel_axis == 1:
                output = tf.transpose(output, (0, 1, 3, 4, 2))
                label = tf.transpose(label, (0, 1, 3, 4, 2))
            best_test_loss(tt_loss)
            for i, metric in enumerate(best_test_metrics):
                metric(label[:, -1], output[:, -1])
                best_test_metrics[i] = metric
            return output, tt_loss
        
        @tf.function
        def test_epoch(image):
            logits, output = model(image, False)
            return output

        #inizialize directories and dictionaries to use on tensorboard
        train_summary_writer = val_summary_writer = test_summary_writer = best_test_summary_writer = None
        train_scalars_dict = val_scalars_dict  = best_test_scalars_dict = test_scalars_dict = None
        
        if not params.dry_run: 
            #Initialization of tensorboard's writers and dictionaries
            train_log_dir = os.path.join(params.experiment_log_dir,  'train')
            val_log_dir = os.path.join(params.experiment_log_dir, 'val')
            test_log_dir = os.path.join(params.experiment_log_dir, 'test')
            best_test_log_dir = os.path.join(params.experiment_log_dir, 'best_test')
            train_summary_writer = tf.summary.create_file_writer(train_log_dir)
            val_summary_writer = tf.summary.create_file_writer(val_log_dir)
            test_summary_writer = tf.summary.create_file_writer(test_log_dir)
            best_test_summary_writer = tf.summary.create_file_writer(best_test_log_dir)
            
            train_scalars_dict = {'Loss': train_loss,'LUT values': train_metrics[0:4], 'Model evaluation': train_metrics[4:7]}
            val_scalars_dict = {'Loss': val_loss, 'LUT values': val_metrics[0:4], 'Model evaluation': val_metrics[4:7]}
            test_scalars_dict = {'Loss': test_loss, 'LUT values': test_metrics[0:4], 'Model evaluation': test_metrics[4:7]}
            best_test_scalars_dict = {'Loss': best_test_loss, 'LUT values': best_test_metrics[0:4], 'Model evaluation': best_test_metrics[4:7]}
#            
        #write the values in tensorboard
        def tboard(writer, log_dir, step, scalar_loss_dict, images_dict, factor):
            with tf.device('/cpu:0'):
                with writer.as_default():
                    for scalar_loss_name, scalar_loss in scalar_loss_dict.items():
                        if (scalar_loss_name == 'LUT values'):
                            with tf.summary.create_file_writer(os.path.join(log_dir, 'TruePositive')).as_default():
                               tf.summary.scalar(scalar_loss_name, scalar_loss[0].result().numpy()/step_per_epoch*factor, step=step)
                            with tf.summary.create_file_writer(os.path.join(log_dir, 'FalsePositive')).as_default():
                               tf.summary.scalar(scalar_loss_name, scalar_loss[1].result().numpy()/step_per_epoch*factor, step=step)
                            with tf.summary.create_file_writer(os.path.join(log_dir, 'TrueNegative')).as_default():
                               tf.summary.scalar(scalar_loss_name, scalar_loss[2].result().numpy()/step_per_epoch*factor, step=step)
                            with tf.summary.create_file_writer(os.path.join(log_dir, 'FalseNegative')).as_default():
                               tf.summary.scalar(scalar_loss_name, scalar_loss[3].result().numpy()/step_per_epoch*factor, step=step)
                        elif (scalar_loss_name == 'Model evaluation'):
                            with tf.summary.create_file_writer(os.path.join(log_dir, 'Accuracy')).as_default():
                               tf.summary.scalar(scalar_loss_name, scalar_loss[0].result()*100, step=step)
                            with tf.summary.create_file_writer(os.path.join(log_dir, 'Precision')).as_default():
                               tf.summary.scalar(scalar_loss_name, scalar_loss[1].result()*100, step=step)
                            with tf.summary.create_file_writer(os.path.join(log_dir, 'Recall')).as_default():
                               tf.summary.scalar(scalar_loss_name, scalar_loss[2].result()*100, step=step)                   
                        else:
                            tf.summary.scalar(scalar_loss_name, scalar_loss.result(), step=step)
                    for image_name, image in images_dict.items():
                        if params.channel_axis == 1:
                            image = tf.transpose(image, (0, 2, 3, 1))
                        tf.summary.image(image_name, image, max_outputs=1, step=step)
        
        #binarization of the output                  
        def post_processing(images):
            images_shape = images.shape
            im_reshaped = np.reshape(images, (images.shape[0], images.shape[1], images.shape[2]))
            bw_predictions = np.zeros((images.shape[0], images.shape[1], images.shape[2])).astype(np.float32)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
            for i in range(0, images.shape[0]):
                ret, bw_predictions[i] = cv2.threshold(im_reshaped[i],0.8, 1 ,cv2.THRESH_BINARY)
                bw_predictions[i] = cv2.morphologyEx(bw_predictions[i], cv2.MORPH_OPEN, kernel)
                bw_predictions[i] = cv2.morphologyEx(bw_predictions[i], cv2.MORPH_CLOSE, kernel)
            bw_predictions = np.reshape(bw_predictions, images_shape)
            return bw_predictions
        
        #visualize images and labels of a batch (can be also used to visualize predictions and labels )
        def show_dataset_labels(x_train, y_train):
            num_train = x_train.shape[0]*x_train.shape[1]
            x_train = np.reshape(x_train, (x_train.shape[0]*x_train.shape[1], x_train.shape[2], x_train.shape[3]))
            y_train = np.reshape(y_train, (y_train.shape[0]*y_train.shape[1], y_train.shape[2], y_train.shape[3]))
            plt.figure(figsize=(15, 15))
            for i in range(0, num_train):
                plt.subplot(num_train/2, 2, i + 1)
                plt.imshow(x_train[i, :,:], cmap = 'gray')
                plt.title("Original Image")
            plt.show()
            for j in range(0, num_train):
                plt.subplot(num_train/2, 2, j + 1)
                plt.imshow(y_train[j, :,:], cmap = 'gray')
                plt.title("Masked Image")
            plt.suptitle("Examples of Images and their Masks")
            plt.show()
       
        template = '{}: Step {}, Loss: {}, Accuracy: {}, Precision: {}, Recall: {}'
        log_print('Start of training')
        try:
            # if True:
            train_imgs_dict = {}
            val_imgs_dict = {}
            best_test_imgs_dict = {}
            test_imgs_dict = {}
            minimum_found = False
            stopped_epoch = None
            stop = False
            output_batch_list = {}
            for i in range(0, params.batch_size):
                output_batch_list[str(i)] = []
            first = True
            
            log_print('Starting of epoch: {}'.format(int(num_epoch)))
            progbar = tf.keras.utils.Progbar(step_per_epoch)

            #iterate along the number of iterations
            for _ in range(int(ckpt.step), params.num_iterations + 1):
                
                if params.aws:
                    r = requests.get('http://169.254.169.254/latest/meta-data/spot/instance-action')
                    if not r.status_code == 404:
                        raise AWSError('Quitting Spot Instance Gracefully')

                image_sequence, seg_sequence, is_last_batch = data_provider.read_batch('train', False, None)
#                show_dataset_labels(image_sequence, seg_sequence)
                train_output_sequence, train_loss_value= train_step(image_sequence, seg_sequence)    
                train_bw_predictions = post_processing(train_output_sequence[:, -1])
                
                progbar.update(int(ckpt.step)- num_epoch*step_per_epoch)
                
                #model.reset_states_per_batch(is_last_batch)  # reset states for sequences that ended

#                if not int(ckpt.step) % params.validation_interval:
#                    #validation
#                    (val_image_sequence, val_seg_sequence, is_last_batch) = data_provider.read_batch('val', False, None, None)
#                        
#                    #if profile is true, write on tensorboard the network graph
#                    if params.profile:
#                        graph_dir = os.path.join(params.experiment_log_dir, 'graph/') + datetime.now().strftime("%Y%m%d-%H%M%S")
#                        tf.summary.trace_on(graph=True, profiler=True)
#                        graph_summary_writer = tf.summary.create_file_writer(graph_dir)
#                    
#                    val_output_sequence, val_loss_value= val_step(val_image_sequence,
#                                                                   val_seg_sequence)
#                    
#                    if not minimum_found:
#                        stop, best_weights = early_stopping.step_end(num_epoch, val_loss)
#                    
#                    if params.profile:
#                        with graph_summary_writer.as_default():
#                            tf.summary.trace_export('train_step', step=int(ckpt.step),
#                                                    profiler_outdir=params.experiment_log_dir)
#                    
#                    val_bw_predictions = post_processing(val_output_sequence[:, -1])
#                    #model.reset_states_per_batch(is_last_batch)  # reset states for sequences that ended   
              
               
                if not int(ckpt.step) % step_per_epoch:
                    
                    #validation
                    for i in range(0, step_val):
                        (val_image_sequence, val_seg_sequence, is_last_batch) = data_provider.read_batch('val', False, None)                      
                        #if profile is true, write on tensorboard the network graph
                        if params.profile:
                            graph_dir = os.path.join(params.experiment_log_dir, 'graph/') + datetime.now().strftime("%Y%m%d-%H%M%S")
                            tf.summary.trace_on(graph=True, profiler=True)
                            graph_summary_writer = tf.summary.create_file_writer(graph_dir)
                    
                        val_output_sequence, val_loss_value= val_step(val_image_sequence,
                                                                      val_seg_sequence)
                        if params.profile:
                            with graph_summary_writer.as_default():
                                tf.summary.trace_export('train_step', step=int(ckpt.step),
                                                        profiler_outdir=params.experiment_log_dir)
                        val_bw_predictions = post_processing(val_output_sequence[:, -1])
                    
                    if not minimum_found:
                        stop, best_weights = early_stopping.step_end(num_epoch, val_loss)

                    #model.reset_states_per_batch(is_last_batch)  # reset states for sequences that ended                      
                    
                    
                    #print training values to console 
                    log_print(template.format('Training', int(ckpt.step),
                          train_loss.result(),
                          train_metrics[4].result() * 100, train_metrics[5].result() * 100, 
                          train_metrics[6].result() * 100))

                     #calling the function that writes the training dictionaries on tensorboard
                    if not params.dry_run:
                        display_image = image_sequence[:, -1]
                        display_image = display_image - tf.reduce_min(display_image, axis=(1, 2, 3), keepdims=True)
                        display_image = display_image / tf.reduce_max(display_image, axis=(1, 2, 3), keepdims=True)
                        train_imgs_dict['Image'] = display_image
                        train_imgs_dict['GT'] = seg_sequence[:, -1]
                        train_imgs_dict['Output'] = train_output_sequence[:, -1]
                        train_imgs_dict['Output_bw'] = train_bw_predictions
                        tboard(train_summary_writer, train_log_dir, int(ckpt.step), train_scalars_dict, train_imgs_dict, 1)
                        
                        log_print('Printed Training Step: {} to Tensorboard'.format(int(ckpt.step)))
                    else:
                        log_print("WARNING: dry_run flag is ON! Not saving checkpoints or tensorboard data")
                        
                    #reset metrics
                    for i in range(0, 7):
                        train_metrics[i].reset_states()
                    train_loss.reset_states()
            
                    #print validation values to console
                    log_print(template.format('Validation', int(ckpt.step),
                                              val_loss.result(),
                                              val_metrics[4].result() * 100, val_metrics[5].result() * 100, 
                                              val_metrics[6].result() * 100))
            
                    #calling the function that writes the validation dictionaries on tensorboard
                    if not params.dry_run:
                        display_image = val_image_sequence[:, -1]
                        display_image = display_image - tf.reduce_min(display_image, axis=(1, 2, 3), keepdims=True)
                        display_image = display_image / tf.reduce_max(display_image, axis=(1, 2, 3), keepdims=True)
                        val_imgs_dict['Image'] = display_image
                        val_imgs_dict['GT'] = val_seg_sequence[:, -1]
                        val_imgs_dict['Output'] = val_output_sequence[:, -1]
                        val_imgs_dict['Output_bw'] = val_bw_predictions
                        tboard(val_summary_writer, val_log_dir, int(ckpt.step), val_scalars_dict, val_imgs_dict, step_per_epoch/step_val)
                    
                        log_print('Printed Validation Step: {} to Tensorboard'.format(int(ckpt.step)))
                    else:
                        log_print("WARNING: dry_run flag is ON! Not saving checkpoints or tensorboard data")
                    
                    #reset metrics                     
                    for i in range(0, 7):
                        val_metrics[i].reset_states()
                    val_loss.reset_states()
                    
                    num_epoch = num_epoch +1
                    log_print('Starting of epoch: {}'.format(int(num_epoch)))
                    
                    progbar = tf.keras.utils.Progbar(step_per_epoch)
                
                if not int(ckpt.step) % step_gif:
                    image_seq, seg_seq = data_provider.read_new_image('epoch_test')
                    if first:
                        for i in range(0, image_seq.shape[0]):
                            image = np.squeeze(np.array(image_seq[i, -1]))
                            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
                            img = Image.fromarray(image.astype(np.uint8))
                            img.save(params.experiment_save_dir + '/image' + str(i) + '.png')
                            seg = np.squeeze(np.array(seg_seq[i, -1]))
                            seg = cv2.normalize(seg, None, 0, 255, cv2.NORM_MINMAX)
                            seg = Image.fromarray(seg.astype(np.uint8))
                            seg.save(params.experiment_save_dir + '/label' + str(i) + '.png')
                        first = False
                            
                    output = test_epoch(image_seq)
                    
                    for i in range(0, image_seq.shape[0]):
                        image = cv2.normalize(np.array(output[i, -1]), None, 0, 255, cv2.NORM_MINMAX)
#                        image = cv2.resize(image, (300, 300), interpolation = cv2.INTER_AREA)
                        image = image.astype(np.uint8)
                        image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
                        image = np.moveaxis(image, -1, 0)
                        output_batch_list[str(i)].append(image)                   
                        write_gif(output_batch_list[str(i)], params.experiment_save_dir + '/prediction' + str(i) + '.gif', fps=5)
                    if num_epoch == 200:
                        step_gif = step_per_epoch*50

                #save checkpoints
                if int(ckpt.step) % params.save_checkpoint_iteration == 0 or int(ckpt.step) == params.num_iterations:
                    if not params.dry_run:
                        save_path = manager.save(int(ckpt.step))
                        log_print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
                    else:
                        log_print("WARNING: dry_run flag is ON! Mot saving checkpoints or tensorboard data")                 
#                
                if stop:
                    actual_weights = model.get_weights()
                    model.set_weights(best_weights)
                    stopped_epoch = early_stopping.on_train_end()
                    log_print('Saving Best Model of inference:')
                    model_fname = os.path.join(params.experiment_save_dir, 'best_model.ckpt')
                    model.save_weights(model_fname, save_format='tf')
                    with open(os.path.join(params.experiment_save_dir, 'best_model_params.pickle'), 'wb') as fobj:
                        pickle.dump({'name': model.__class__.__name__, 'params': (net_kernel_params,)},
                                     fobj, protocol=pickle.HIGHEST_PROTOCOL)
                    log_print('Saved. Continue training')
                    stop = False
                    minimum_found = True
                    #create the dataset
                    num_testing = data_provider.num_test()
                    data_provider.enqueue_index('best_test', None)
                    for i in range(0, num_testing):
                        image_seq, seg_seq = data_provider.read_new_image('best_test')
                        best_test_output_sequence, best_test_loss_value= best_test_step(image_seq, seg_seq)
                        log_print(template.format('Testing', int(i),
                                                  best_test_loss.result(),
                                                  best_test_metrics[4].result() * 100, best_test_metrics[5].result() * 100, 
                                                  best_test_metrics[6].result() * 100))
                        display_image = image_seq[:, -1]
                        display_image = display_image - tf.reduce_min(display_image, axis=(1, 2, 3), keepdims=True)
                        display_image = display_image / tf.reduce_max(display_image, axis=(1, 2, 3), keepdims=True)
                        best_test_imgs_dict['Image'] = display_image
                        best_test_imgs_dict['GT'] = seg_seq[:, -1]
                        best_test_imgs_dict['Output'] = best_test_output_sequence[:, -1]
                        tboard(best_test_summary_writer, best_test_log_dir, i, best_test_scalars_dict, best_test_imgs_dict, step_per_epoch/1)
                        log_print('Printed Testing Step: {} to Tensorboard'.format(i))
                        for i in range(0, 7):
                            best_test_metrics[i].reset_states()
                        best_test_loss.reset_states()
                    
                    model.set_weights(actual_weights)
                    
                #when it comes to the end save the final precisions ans losses AND PERFORM PREDICTION ON NEW SAMPLES
                if ckpt.step == params.num_iterations:
                    final_train_loss = train_loss.result()
                    final_val_loss = val_loss.result()
                    final_train_prec = train_metrics[5].result() * 100
                    final_val_prec = train_metrics[5].result() * 100
                    #create the dataset
                    num_testing = data_provider.num_test()
                    data_provider.enqueue_index('test', None)
                    for i in range(0, num_testing):
                        image_seq, seg_seq = data_provider.read_new_image('test')
                        test_output_sequence, test_loss_value= test_step(image_seq, seg_seq)
                        log_print(template.format('Testing', int(i),
                                                  test_loss.result(),
                                                  test_metrics[4].result() * 100, test_metrics[5].result() * 100, 
                                                  test_metrics[6].result() * 100))
                        display_image = image_seq[:, -1]
                        display_image = display_image - tf.reduce_min(display_image, axis=(1, 2, 3), keepdims=True)
                        display_image = display_image / tf.reduce_max(display_image, axis=(1, 2, 3), keepdims=True)
                        test_imgs_dict['Image'] = display_image
                        test_imgs_dict['GT'] = seg_seq[:, -1]
                        test_imgs_dict['Output'] = test_output_sequence[:, -1]
                        tboard(test_summary_writer, test_log_dir, i, test_scalars_dict, test_imgs_dict, step_per_epoch/1)
                        log_print('Printed Testing Step: {} to Tensorboard'.format(i))
                        for i in range(0, 7):
                            test_metrics[i].reset_states()
                        test_loss.reset_states()

        except (KeyboardInterrupt, ValueError, AWSError) as err:
            if not params.dry_run:
                log_print('Saving Model Before closing due to error: {}'.format(str(err)))
                save_path = manager.save(int(ckpt.step))
                log_print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
                # raise err

        except Exception as err:
            #
            raise err
        finally:
            if not params.dry_run:
                #save model's weights
                log_print('Saving Model of inference:')
                model_fname = os.path.join(params.experiment_save_dir, 'model.ckpt')
                model.save_weights(model_fname, save_format='tf')
                with open(os.path.join(params.experiment_save_dir, 'model_params.pickle'), 'wb') as fobj:
                    pickle.dump({'name': model.__class__.__name__, 'params': (net_kernel_params,)},
                                fobj, protocol=pickle.HIGHEST_PROTOCOL)
                #save parameters values and final loss and precision values 
                with open(os.path.join(params.experiment_save_dir, 'params_list.csv'), 'w') as fobj:
                    writer = csv.writer(fobj)
                    model_dict = {'Pretraining': pretraining, 'Mode': pretraining_type, 'Stopping_epoch': stopped_epoch, 'Attention gate': attention_gate,
                                  'Dropout': dropout, 'Recurrent dropout': recurrent_dropout, 'L1': l1, 'L2': l2, 
                                  'Kernel init': kernel_init, 'Net type': net_type, 'Learning rate': lrate, 
                                  'Lr decay': lr_decay}
                    model_dict.update({'Train_loss': final_train_loss, 'Train_precision': final_train_prec,
                                      'Val_loss': final_val_loss, 'Val_precision': final_val_prec})
                    for key, value in model_dict.items():
                       writer.writerow([key, value])
                log_print('Saved Model to file: {}'.format(model_fname))
                end_time = time.time()
                log_print('Program execution time:', end_time - start_time)                
            else:
                log_print('WARNING: dry_run flag is ON! Not Saving Model')
            log_print('Closing gracefully')
            log_print('Done')


if __name__ == '__main__':

    class AddNets(argparse.Action):
        import NewNet as Nets

        def __init__(self, option_strings, dest, **kwargs):
            super(AddNets, self).__init__(option_strings, dest, **kwargs)

        def __call__(self, parser, namespace, values, option_string=None):
            nets = [getattr(Nets, v) for v in values]
            setattr(namespace, self.dest, nets)


    class AddReader(argparse.Action):

        def __init__(self, option_strings, dest, **kwargs):
            super(AddReader, self).__init__(option_strings, dest, **kwargs)

        def __call__(self, parser, namespace, values, option_string=None):
            reader = getattr(DataHandeling, values)
            setattr(namespace, self.dest, reader)


    class AddDatasets(argparse.Action):

        def __init__(self, option_strings, dest, *args, **kwargs):

            super(AddDatasets, self).__init__(option_strings, dest, *args, **kwargs)

        def __call__(self, parser, namespace, values, option_string=None):

            if len(values) % 2:
                raise ValueError("dataset values should be of length 2*N where N is the number of datasets")
            datastets = []
            for i in range(0, len(values), 2):
                datastets.append((values[i], (values[i + 1])))
            setattr(namespace, self.dest, datastets)


    arg_parser = argparse.ArgumentParser(description='Run Train LSTMUnet Segmentation')
    arg_parser.add_argument('-n', '--experiment_name', dest='experiment_name', type=str,
                            help="Name of experiment")
    arg_parser.add_argument('--gpu_id', dest='gpu_id', type=str,
                            help="Visible GPUs: example, '0,2,3'")
    arg_parser.add_argument('--dry_run', dest='dry_run', action='store_const', const=True,
                            help="Do not write any outputs: for debugging only")
    arg_parser.add_argument('--profile', dest='profile', type=bool,
                            help="Write profiling data to tensorboard. For debugging only")
    arg_parser.add_argument('--root_data_dir', dest='root_data_dir', type=str,
                            help="Root folder containing training data")
    arg_parser.add_argument('--data_provider_class', dest='data_provider_class', type=str, action=AddReader,
                            help="Type of data provider")
    arg_parser.add_argument('--dataset', dest='train_sequence_list', type=str, action=AddDatasets, nargs='+',
                            help="Datasets to run. string of pairs: DatasetName, SequenceNumber")
    arg_parser.add_argument('--val_dataset', dest='val_sequence_list', type=str, action=AddDatasets, nargs='+',
                            help="Datasets to run. string of pairs DatasetName, SequenceNumber")
    arg_parser.add_argument('--net_gpus', dest='net_gpus', type=int, nargs='+',
                            help="gpus for each net: example: 0 0 1")
    arg_parser.add_argument('--net_types', dest='net_types', type=int, nargs='+', action=AddNets,
                            help="Type of nets")
    arg_parser.add_argument('--crop_size', dest='crop_size', type=int, nargs=2,
                            help="crop size for y and x dimensions: example: 160 160")
    arg_parser.add_argument('--train_q_capacity', dest='train_q_capacity', type=int,
                            help="Capacity of training queue")
    arg_parser.add_argument('--val_q_capacity', dest='val_q_capacity', type=int,
                            help="Capacity of validation queue")
    arg_parser.add_argument('--num_train_threads', dest='num_train_threads', type=int,
                            help="Number of train data threads")
    arg_parser.add_argument('--num_val_threads', dest='num_val_threads', type=int,
                            help="Number of validation data threads")
    arg_parser.add_argument('--data_format', dest='data_format', type=str, choices=['NCHW', 'NWHC'],
                            help="Data format NCHW or NHWC")
    arg_parser.add_argument('--batch_size', dest='batch_size', type=int,
                            help="Batch size")
    arg_parser.add_argument('--unroll_len', dest='unroll_len', type=int,
                            help="LSTM unroll length")
    arg_parser.add_argument('--num_iterations', dest='num_iterations', type=int,
                            help="Maximum number of training iterations")
    arg_parser.add_argument('--validation_interval', dest='validation_interval', type=int,
                            help="Number of iterations between validation iteration")
    arg_parser.add_argument('--load_checkpoint', dest='load_checkpoint', action='store_const', const=True,
                            help="Load from checkpoint")
    arg_parser.add_argument('--load_checkpoint_path', dest='load_checkpoint_path', type=str,
                            help="path to checkpoint, used only with --load_checkpoint")
    arg_parser.add_argument('--continue_run', dest='continue_run', action='store_const', const=True,
                            help="Continue run in existing directory")
    arg_parser.add_argument('--learning_rate', dest='learning_rate', type=float,
                            help="Learning rate")
    arg_parser.add_argument('--decay_rate', dest='decay_rate', type=float,
                            help="Decay rate")
    arg_parser.add_argument('--class_weights', dest='class_weights', type=float, nargs=3,
                            help="class weights for background, foreground and edge classes")
    arg_parser.add_argument('--save_checkpoint_dir', dest='save_checkpoint_dir', type=str,
                            help="root directory to save checkpoints")
    arg_parser.add_argument('--save_log_dir', dest='save_log_dir', type=str,
                            help="root directory to save tensorboard outputs")
    arg_parser.add_argument('--tb_sub_folder', dest='tb_sub_folder', type=str,
                            help="sub-folder to save outputs")
    arg_parser.add_argument('--save_checkpoint_iteration', dest='save_checkpoint_iteration', type=int,
                            help="number of iterations between save checkpoint")
    arg_parser.add_argument('--save_checkpoint_max_to_keep', dest='save_checkpoint_max_to_keep', type=int,
                            help="max recent checkpoints to keep")
    arg_parser.add_argument('--save_checkpoint_every_N_hours', dest='save_checkpoint_every_N_hours', type=int,
                            help="keep checkpoint every N hours")
    arg_parser.add_argument('--write_to_tb_interval', dest='write_to_tb_interval', type=int,
                            help="Interval between writes to tensorboard")
    sys_args = sys.argv

    input_args = arg_parser.parse_args()
    args_dict = {key: val for key, val in vars(input_args).items() if not (val is None)}
    print(args_dict)
    # params = Params.CTCParamsNoLSTM(args_dict)

    # try:
    #     train()
    # finally:
    #     log_print('Done')
    
    
#    changes = [[False, False], [False, True], [True, False]]
#    for i, comb in enumerate(changes):
    params = Params.CTCParams(args_dict)
    train()
    

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
import pickle
import Networks as Nets
import Params
import tensorflow as tf
#from Sequence_generator import DataGenerator
from Data_generator import DataSet
import sys
from utils import log_print
import cv2
from tensorflow.python.keras import losses
import time 
import numpy as np
from netdict import Net_type
import csv 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#from keras.preprocessing.sequence import TimeseriesGenerator


try:
    # noinspection PyPackageRequirements
    import tensorflow.python.keras as k
except AttributeError:
    # noinspection PyPackageRequirements,PyUnresolvedReferences
    import tensorflow.keras as k


METRICS = [
      k.metrics.TruePositives(name='tp'),
      k.metrics.FalsePositives(name='fp'),
      k.metrics.TrueNegatives(name='tn'),
      k.metrics.FalseNegatives(name='fn'), 
      k.metrics.BinaryAccuracy(name='accuracy'),
      k.metrics.Precision(name='precision'),
      k.metrics.Recall(name='recall'),
]


os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
print(f'Using Tensorflow version {tf.__version__}')
if not tf.__version__.split('.')[0] == '2':
    raise ImportError(f'Required tensorflow version 2.x. current version is: {tf.__version__}')


start_time = time.time()

class AWSError(Exception):
    pass

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

    def bce_dice_loss(self, y_true, y_pred):
        y_true = y_true[:, -1]
        y_pred = y_pred[:, -1]
        bce_loss = losses.binary_crossentropy(y_true, y_pred)
        dice_loss = self.dice_loss(y_true, y_pred)
        loss = bce_loss + dice_loss
        return loss
    

def train():
   
    device = '/gpu:0' if params.gpu_id >= 0 else '/cpu:0'
    with tf.device(device):
        
        dropout = 0
        drop_input = False
        l1 = 0
        l2 = 0
        kernel_init = 'he_normal'
        net_type = 'original_net'
        lr = 0.0005
        lr_decay = 0.96

        # Model
        net_kernel_params = Net_type(dropout, (l1, l2), kernel_init)[net_type]
        model = Nets.ULSTMnet2D(net_kernel_params, params.data_format, False, drop_input)
        
        inputs = tf.keras.Input(shape=[params.unroll_len, params.reshape_size[0], params.reshape_size[1], 1], batch_size=params.batch_size)
        sigmoid = model(inputs)
        model_keras = tf.keras.Model(inputs=inputs, outputs=sigmoid)

        # Losses and Metrics
        loss_fn = LossFunction()
        final_train_loss = 0
        final_val_loss = 0
        final_train_prec = 0
        final_val_prec = 0

        parameter = {
          'shuffle': False,
          'image_crop_size': params.crop_size,
          'image_reshape_size': params.reshape_size,
          'unroll_len': params.unroll_len,
          'batch_size': params.batch_size,
          'data_format': params.data_format,
          'randomize': True,
          }
        
        # Save Checkpoints

        lr_schedule = tf.optimizers.schedules.ExponentialDecay(
                initial_learning_rate = lr, 
                decay_steps=100000,
                decay_rate=lr_decay, 
                staircase=True)
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        
        cp = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(params.experiment_save_dir, 'tf_ckpts'),
                                        monitor='val_loss',
                                        save_best_only=True,
                                        verbose=1)

        tb = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(params.experiment_log_dir), 
                                            write_graph=True, write_images=True, update_freq='epoch')

                        
        def post_processing(images):
            images_shape = images.shape
            im_reshaped = np.reshape(images, (images_shape[0]*images_shape[1], images_shape[2], images_shape[3]))
            bw_predictions = np.zeros((im_reshaped.shape[0], im_reshaped.shape[1], im_reshaped.shape[2])).astype(np.float32)
            for i in range(0, images.shape[0]):
                ret, bw_predictions[i] = cv2.threshold(im_reshaped[i],0.8, 1 ,cv2.THRESH_BINARY)
            bw_predictions = np.reshape(bw_predictions, images_shape)
            return bw_predictions
        
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
            
        log_print('Start of training')
        try:
            # if True:
            sequence_folder = params.root_data_dir
            with open(os.path.join(sequence_folder, 'full_csv_prova.pkl'), 'rb') as fobj:
                metadata = pickle.load(fobj)
            filename_list = metadata['filelist']
            valid_masks = [i for i, x in enumerate(filename_list) if x[1] != 'None']
            valid_list_train, valid_list_val = train_test_split(valid_masks, test_size= 0.2)
            partition = {'train': valid_list_train, 'validation': valid_list_val}
#            data_training = TimeseriesGenerator(partition['train'], np.linspace(0, len(partition['train']), len(partition['train'])), length=params.unroll_len, batch_size=params.batch_size)
#            data_validation = TimeseriesGenerator(partition['validation'], np.linspace(0, len(partition['validation']), len(partition['validation'])), length=params.unroll_len, batch_size=params.batch_size)
#            data_training = DataGenerator(partition['train'], filename_list, sequence_folder, **parameter)
#            data_validation = DataGenerator(partition['validation'], filename_list, sequence_folder, **parameter)
            data_training = DataSet(partition['train'], filename_list, sequence_folder, **parameter)
            data_validation = DataSet(partition['validation'], filename_list, sequence_folder, **parameter)
            train_generator = data_training.frame_generator()
            val_generator = data_validation.frame_generator()
            steps = int(np.floor(len(partition['train'])/params.batch_size))
            val_steps = int(np.floor(len(partition['validation'])/params.batch_size))
            model_keras.compile(optimizer=optimizer, loss=loss_fn.bce_dice_loss, metrics= METRICS)
            history = model_keras.fit(x = train_generator,verbose=1, callbacks=[tb, cp], validation_data=val_generator, 
                                steps_per_epoch=steps, validation_steps = val_steps, epochs= 200)
#                show_dataset_labels(image_sequence, seg_sequence)
            
            
            final_train_loss = history.history['loss'][-1]
            final_val_loss = history.history['val_loss'][-1]
            final_train_prec = history.history['acc'][-1]
            final_val_prec = history.history['val_acc'][-1]

        except Exception as err:
            #
            raise err
        finally:
            if not params.dry_run:
                log_print('Saving Model of inference:')
                model_fname = os.path.join(params.experiment_save_dir, 'model')
                model.save_weights(model_fname, save_format='tf')
                with open(os.path.join(params.experiment_save_dir, 'model_params.pickle'), 'wb') as fobj:
                    pickle.dump({'name': model.__class__.__name__, 'params': (net_kernel_params,)},
                                fobj, protocol=pickle.HIGHEST_PROTOCOL)
                with open(os.path.join(params.experiment_save_dir, 'params_list.csv'), 'w') as fobj:
                    writer = csv.writer(fobj)
                    model_dict = {'Dropout': dropout, 'Drop_input': drop_input, 'L1': l1, 'L2': l2, 
                                  'Kernel init': kernel_init, 'Net type': net_type, 'Learning rate': lr, 
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
        import Networks as Nets

        def __init__(self, option_strings, dest, **kwargs):
            super(AddNets, self).__init__(option_strings, dest, **kwargs)

        def __call__(self, parser, namespace, values, option_string=None):
            nets = [getattr(Nets, v) for v in values]
            setattr(namespace, self.dest, nets)


    class AddReader(argparse.Action):

        def __init__(self, option_strings, dest, **kwargs):
            super(AddReader, self).__init__(option_strings, dest, **kwargs)

        def __call__(self, parser, namespace, values, option_string=None):
            reader = getattr(DataGenerator, values)
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
    arg_parser.add_argument('--reshape_size', dest='reshape_size', type=int, nargs=2,
                            help="reshape size for y and x dimensions: example: 160 160")
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
    params = Params.CTCParams(args_dict)
    # params = Params.CTCParamsNoLSTM(args_dict)

    # try:
    #     train()
    # finally:
    #     log_print('Done')
    

    train()


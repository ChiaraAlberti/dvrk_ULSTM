#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import pickle
from utils import get_model, log_print
import tensorflow as tf
import Params
import argparse
import DataHandeling
import NewNet as Nets
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt


try:
    # noinspection PyPackageRequirements
    import tensorflow.python.keras as k
except AttributeError:
    # noinspection PyPackageRequirements,PyUnresolvedReferences
    import tensorflow.keras as k


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

METRICS_TEST = [
      k.metrics.TruePositives(name='tp_test'),
      k.metrics.FalsePositives(name='fp_test'),
      k.metrics.TrueNegatives(name='tn_test'),
      k.metrics.FalseNegatives(name='fn_test'), 
      k.metrics.BinaryAccuracy(name='accuracy_test'),
      k.metrics.Precision(name='precision_test'),
      k.metrics.Recall(name='recall_test'),
]


def test():
    model_folder = '/home/stormlab/seg/LSTM-UNet-Outputs/Retrained/LSTMUNet/MyRun_SIM/2020-03-03_191130'
    with open(os.path.join(model_folder, 'model_params.pickle'), 'rb') as fobj:
        model_dict = pickle.load(fobj)
    model_cls = get_model(model_dict['name'])
    
    device = '/gpu:0'
    with tf.device(device):
        model = model_cls(*model_dict['params'], data_format='NHWC', pad_image=False)
        model.load_weights(os.path.join(model_folder, 'model.ckpt'))
        log_print("Restored from {}".format(os.path.join(model_folder, 'model')))
        
    image = cv2.imread('/home/stormlab/seg/LSTM-UNet-Outputs/Retrained/LSTMUNet/MyRun_SIM/2020-03-03_191130/image7.png', -1)
    plt.imshow(image, cmap = 'gray')
    img = cv2.resize(image, (64, 64), interpolation = cv2.INTER_AREA)
    image = cv2.normalize(img.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX)
    np_image = np.expand_dims(image, axis=0)  # Add another dimension for tensorflow
    np_image = np.expand_dims(np_image, axis=0)
    np_image = np.expand_dims(np_image, axis=-1)

    
    logits, pred = model(np_image, False)
    pred = np.squeeze(pred, (0, 1, 4))
    plt.imshow(pred, cmap = 'gray')
    
#    data_provider = params.data_provider
#    num_testing = data_provider.num_test()
#    data_provider.enqueue_index('test')
#    template = '{}: Step {}, Loss: {}, Accuracy: {}, Precision: {}, Recall: {}'
#    for i in range(0, num_testing):
#        image_seq, seg_seq = data_provider.read_new_image('test')
#        test_output_sequence, test_loss_value= test_step(image_seq, seg_seq)
#        log_print(template.format('Testing', int(i),
#                                  test_loss.result(),
#                                  test_metrics[4].result() * 100, test_metrics[5].result() * 100, 
#                                  test_metrics[6].result() * 100))
##            display_image = image_seq[:, -1]
##            display_image = display_image - tf.reduce_min(display_image, axis=(1, 2, 3), keepdims=True)
##            display_image = display_image / tf.reduce_max(display_image, axis=(1, 2, 3), keepdims=True)
##            test_imgs_dict['Image'] = display_image
##            test_imgs_dict['GT'] = seg_seq[:, -1]
##            test_imgs_dict['Output'] = test_output_sequence[:, -1]
##            tboard(test_summary_writer, test_log_dir, i, test_scalars_dict, test_imgs_dict, step_per_epoch/1)
##            log_print('Printed Testing Step: {} to Tensorboard'.format(i))
#        for i in range(0, 7):
#            test_metrics[i].reset_states()
#        test_loss.reset_states()
#    for i in range(params.batch_size):
#        label_name = '/home/stormlab/seg/Output/Test_images/' + 'label' + str(i) + '.png'
#        image_name = '/home/stormlab/seg/Output/Test_images/' + 'image' + str(i) + '.png'
#        pred_name = '/home/stormlab/seg/Output/Test_images/' + 'pred' + str(i) + '.png'
#        pred = cv2.normalize(np.squeeze(np.array(test_output_sequence[i, -1])).astype(np.uint8), None, 0.0, 255, cv2.NORM_MINMAX)
#        image = cv2.normalize(np.squeeze(np.array(image_seq[i, -1])).astype(np.uint8), None, 0.0, 255, cv2.NORM_MINMAX)
#        label = cv2.normalize(np.squeeze(np.array(seg_seq[i, -1])).astype(np.uint8), None, 0.0, 255, cv2.NORM_MINMAX)
#        cv2.imwrite(filename=pred_name, img=np.squeeze(np.array(test_output_sequence[i, -1])).astype(np.uint8))
#        cv2.imwrite(filename=label_name, img=np.squeeze(np.array(seg_seq[i, -1])).astype(np.uint8))
#        cv2.imwrite(filename=image_name, img=np.squeeze(np.array(image_seq[i, -1])).astype(np.uint8))
#        print('Saved files', i)
                
    
    
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
    
    params = Params.CTCParams(args_dict)
    
    test()
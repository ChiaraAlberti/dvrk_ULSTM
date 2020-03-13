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
import time 
import numpy as np
from netdict import Net_type
import csv 
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

try:
    # noinspection PyPackageRequirements
    import tensorflow.python.keras as k
except AttributeError:
    # noinspection PyPackageRequirements,PyUnresolvedReferences
    import tensorflow.keras as k


#defining metrics for training, validation and test
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


os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
print(f'Using Tensorflow version {tf.__version__}')
if not tf.__version__.split('.')[0] == '2':
    raise ImportError(f'Required tensorflow version 2.x. current version is: {tf.__version__}')


start_time = time.time()

class AWSError(Exception):
    pass

#loss function: dice_loss + weighted binary cross entropy loss
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
        bce_loss = tf.nn.weighted_cross_entropy_with_logits(y_true, logits, 0.6)
        dice_loss = self.dice_loss(y_true, y_pred)
        loss = alpha*bce_loss + (1- alpha)*dice_loss
        return loss

#loss weighted cross entropy
class WeightedLoss():
    def loss(self, y_true, y_pred):
        y_true = y_true[:, -1]
        y_pred = y_pred[:, -1]
        loss = tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, 1.2)
#        loss = tf.reduce_sum(loss) / (tf.reduce_sum(np.ones(y_true.shape).astype(np.float32)) + 0.00001)
        return loss


class JaccardIndex() :   
    def loss(self, y_true, y_pred):
        y_true = y_true[:, -1]
        y_pred = y_pred[:, -1]
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])
        y_pred = tf.cast(tf.math.greater(y_pred, 0.5), tf.float32)
        intersection = tf.reduce_sum(tf.abs(y_true * y_pred), axis=-1)
        sum_ = tf.reduce_sum(tf.abs(y_true) + tf.abs(y_pred), axis=-1)
        jac = intersection/sum_
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


def train(train_index, test_index, kfold_dir):
   
    device = '/gpu:0' if params.gpu_id >= 0 else '/cpu:0'
    with tf.device(device):
        #Initialization of the data
        data_provider = params.data_provider
        #Initialization of the model and paramenters
        recurrent_dropout = 0.3
        dropout = 0.3
        l1 = 0
        l2 = 0
        kernel_init = 'he_uniform'
        net_type = 'original_net'
        pretraining = False
        lstm_type = 'full'
        num_epoch = 0
        discriminator = False
        attention_gate = False
        
        #initialization neural network
        net_kernel_params = Net_type(recurrent_dropout, (l1, l2), kernel_init)[net_type]
        model = Nets.ULSTMnet2D(net_kernel_params, params.data_format, False, dropout, 
                                pretraining, lstm_type, attention_gate)
        
        if discriminator:
            disc = Nets.Discriminator(params.data_format)
            
        #Initialization of Losses and Metrics
        loss_fn = LossFunction()
        jaccard = JaccardIndex()
#        loss_fn = WeightedLoss()
        train_loss = k.metrics.Mean(name='train_loss')
        train_metrics = METRICS_TRAIN
        val_loss = k.metrics.Mean(name='val_loss')
        val_metrics = METRICS_VAL
        test_loss = k.metrics.Mean(name = 'test_loss')
        test_jindx = k.metrics.Mean(name = 'jaccard_index')
        test_metrics = METRICS_TEST
        final_test_loss = 0
        final_test_prec = 0
        final_test_acc = 0
        final_test_rec = 0
        final_test_jac = 0
    
        #define learning rate step decay
        if lstm_type == 'enc':
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
                  elif tf.logical_and(tf.greater(step, 5000), tf.less(step, 8000)):
                    return 0.000005
                  else:
                    return 0.000001            
        else: 
            class decay_lr(tf.keras.optimizers.schedules.LearningRateSchedule):
                def __init__(self):
                    print('Learning rate initialized')
                    
                @tf.function   
                def __call__(self, step):
                  if tf.less(step, 700):
                    return 0.0001
                  elif tf.logical_and(tf.greater(step, 700), tf.less(step, 2800)):
                    return 0.00005
                  elif tf.logical_and(tf.greater(step, 2800), tf.less(step, 7000)):
                    return 0.00001
                  elif tf.logical_and(tf.greater(step, 7000), tf.less(step, 11200)):
                    return 0.000005
                  else:
                    return 0.000001          
        
        
        #Set optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=decay_lr())
        if discriminator: 
            optimizer_disc = tf.keras.optimizers.Adam(learning_rate=decay_lr())
        
        #Checkpoint
        ckpt = tf.train.Checkpoint(step=tf.Variable(0, dtype=tf.int64), optimizer=optimizer, net=model)
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

        manager = tf.train.CheckpointManager(ckpt, os.path.join(params.experiment_k_fold_dir, 'NN_' + str(kfold_dir), 'tf_ckpts'),
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
            jaccard_ind = jaccard.loss(label, output)
            test_jindx(jaccard_ind)
            return output, tt_loss
                       
        template = '{}: Step {}, Loss: {}, Accuracy: {}, Precision: {}, Recall: {}'
        
        log_print('Start of training')
        try:
            log_print('Starting of epoch: {}'.format(int(num_epoch)))
            #divide in train and val set
            training_index, val_index = train_test_split(train_index, test_size = 0.2)
            #define number of train and val step per epoch
            step_per_epoch =  int(np.floor(len(training_index)/params.batch_size))
            step_val = int(np.floor(len(val_index)/params.batch_size))
            progbar = tf.keras.utils.Progbar(step_per_epoch)
            val_history = []

            for _ in range(int(ckpt.step), params.num_iterations + 1):
                if params.aws:
                    r = requests.get('http://169.254.169.254/latest/meta-data/spot/instance-action')
                    if not r.status_code == 404:
                        raise AWSError('Quitting Spot Instance Gracefully')
                #read batch
                image_sequence, seg_sequence, is_last_batch = data_provider.read_batch('train', True, training_index)
                #train batch
                train_output_sequence, train_loss_value= train_step(image_sequence, seg_sequence)
                
                progbar.update(int(ckpt.step)- num_epoch*step_per_epoch)
                
#                model.reset_states_per_batch(is_last_batch)  # reset states for sequences that ended

                if int(ckpt.step) % params.save_checkpoint_iteration == 0 or int(ckpt.step) == params.num_iterations:
                    if not params.dry_run:
                        save_path = manager.save(int(ckpt.step))
                        log_print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
                    else:
                        log_print("WARNING: dry_run flag is ON! Mot saving checkpoints or tensorboard data")

                #if epoch is finished
                if not int(ckpt.step) % step_per_epoch:
                 
                    #validation steps are performed
                    for i in range(0, step_val):
                        (val_image_sequence, val_seg_sequence, is_last_batch) = data_provider.read_batch('val', True, val_index)                      
                        val_output_sequence, val_loss_value= val_step(val_image_sequence,
                                                                      val_seg_sequence)                                                        
                    #validation metrics are stored in a dictionary
                    val_dict = {'Accuracy ': np.array(val_metrics[4].result()) * 100, 'Precision':  np.array(val_metrics[5].result()) * 100, 'Recall':  np.array(val_metrics[6].result()) * 100}
                    val_history.append(val_dict)
                    #print training values to console 
                    log_print(template.format('Training', int(ckpt.step),
                          train_loss.result(),
                          train_metrics[4].result() * 100, train_metrics[5].result() * 100, 
                          train_metrics[6].result() * 100))
                        
                    #reset train metrics 
                    for i in range(0, 7):
                        train_metrics[i].reset_states()
                    train_loss.reset_states()
            
                    #print validation values to console
                    log_print(template.format('Validation', int(ckpt.step),
                                              val_loss.result(),
                                              val_metrics[4].result() * 100, val_metrics[5].result() * 100, 
                                              val_metrics[6].result() * 100))
                    
                    #reset val metrics                     
                    for i in range(0, 7):
                        val_metrics[i].reset_states()
                    val_loss.reset_states()
                    
                    num_epoch = num_epoch +1
                    log_print('Starting of epoch: {}'.format(int(num_epoch)))
                    
                    progbar = tf.keras.utils.Progbar(step_per_epoch)
                    
                #when training is finished
                if ckpt.step == params.num_iterations:
                    #define number of test steps 
                    num_testing = int(np.floor(len(test_index) / params.batch_size))
                    #create the dataset for test
                    data_provider.enqueue_index('test', test_index)
                    #perform testing on new data
#                    loss_list = []
#                    acc_list = []
#                    rec_list = []
#                    prec_list = []
#                    jacc_list = []
                    
                    for i in range(0, num_testing):
                        image_seq, seg_seq = data_provider.read_new_image('test')
                        test_output_sequence, test_loss_value= test_step(image_seq, seg_seq)
                        log_print(template.format('Testing', int(i),
                                                  test_loss.result(),
                                                  test_metrics[4].result() * 100, test_metrics[5].result() * 100, 
                                                  test_metrics[6].result() * 100))
#                        loss_list.append(test_loss.result())
#                        acc_list.append(test_metrics[4].result() * 100)
#                        rec_list.append(test_metrics[6].result() * 100)
#                        prec_list.append(test_metrics[5].result() * 100)
#                        jacc_list.append(test_jindx.result())
                        
#                        for i in range(0, 7):
#                            test_metrics[i].reset_states()
#                        test_loss.reset_states()
#                        test_jindx.reset_states()
  
                    #save test values
                    final_test_loss = test_loss.result()
                    final_test_acc = test_metrics[4].result() * 100
                    final_test_prec = test_metrics[5].result() * 100
                    final_test_rec = test_metrics[6].result() * 100
                    final_test_jac = test_jindx.result()
#                    final_test_loss = np.mean(loss_list)
#                    final_test_acc = np.mean(acc_list)
#                    final_test_prec = np.mean(prec_list)
#                    final_test_rec = np.mean(rec_list)
#                    final_test_jac = np.mean(jacc_list)
                    

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
                #save the model
                log_print('Saving Model of inference:')
                model_fname = os.path.join(params.experiment_k_fold_dir, 'NN_'+ str(kfold_dir), 'model.ckpt'.format(int(ckpt.step)))
                model.save_weights(model_fname, save_format='tf')
                with open(os.path.join(params.experiment_k_fold_dir, 'NN_'+ str(kfold_dir), 'model_params.pickle'), 'wb') as fobj:
                    pickle.dump({'name': model.__class__.__name__, 'params': (net_kernel_params,)},
                                fobj, protocol=pickle.HIGHEST_PROTOCOL)
                    
                #save parameters values and final loss and precision values 
                with open(os.path.join(params.experiment_k_fold_dir, 'NN_'+ str(kfold_dir), 'metrics_list.csv'), 'w') as fobj:
                    writer = csv.writer(fobj)
                    model_dict = {'Loss': np.array(final_test_loss), 'Accuracy': np.array(final_test_acc), 
                                  'Precision': np.array(final_test_prec), 'Recall': np.array(final_test_rec), 
                                  'Jaccard': np.array(final_test_jac)}
                    for key, value in model_dict.items():
                       writer.writerow([key, value])
                       
                with open(os.path.join(params.experiment_k_fold_dir, 'NN_'+ str(kfold_dir), 'val_history.csv'), 'w') as f: 
                    writer = csv.DictWriter(f, val_history[0].keys())
                    writer.writeheader()
                    writer.writerows(val_history)
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
    params = Params.CTCParams(args_dict)

    complete_dict = []
    #define the number of k folds 
    kf = KFold(n_splits=10)
    #create the k train-test pairs
    train_seq, test_seq = params.data_provider.split_k_fold(kf)
    kfold_ind = 0
    
    for train_index, test_index in zip(train_seq, test_seq):
        train(train_index, test_index, kfold_ind)
        with open(os.path.join(params.experiment_k_fold_dir, 'NN_'+ str(kfold_ind), 'metrics_list.csv')) as csv_file:
            reader = csv.reader(csv_file)
            model_dict = dict(reader)
            complete_dict.append(model_dict)
        with open(os.path.join(params.experiment_k_fold_dir, 'kfold_metrics_list.csv'), 'w') as f: 
            writer = csv.DictWriter(f, complete_dict[0].keys())
            writer.writeheader()
            writer.writerows(complete_dict)

        kfold_ind = kfold_ind + 1
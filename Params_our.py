import DataHandeling_our as DataHandeling
import os
from datetime import datetime
import Networks_our as Nets

ROOT_DATA_DIR = '/home/stormlab/seg/lstm_dataset'
ROOT_TEST_DATA_DIR = '/home/stormlab/seg/Test'
ROOT_SAVE_DIR = '/home/stormlab/seg/LSTM-UNet-Outputs/Retrained'

class ParamsBase(object):
    aws = False

    def _override_params_(self, params_dict: dict):
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        this_dict = self.__class__.__dict__.keys()
        for key, val in params_dict.items():
            if key not in this_dict:
                print('Warning!: Parameter:{} not in default parameters'.format(key))
            setattr(self, key, val)

    pass


class CTCParams(ParamsBase):
    # --------General-------------
    experiment_name = 'MyRun_SIM'
    gpu_id = 0  # set -1 for CPU or GPU index for GPU.
    trial = 1

    #  ------- Data -------
    data_provider_class = DataHandeling.CTCRAMReaderSequence2D
    root_data_dir = ROOT_DATA_DIR
    crop_size = (64, 64)  # (height, width) preferably height=width 
    reshape_size = (64, 64)
    batch_size = 4
    unroll_len = 4
    data_format = 'NHWC' # either 'NCHW' or 'NHWC'
    train_q_capacity = 200
    val_q_capacity = 200
    num_val_threads = 2
    num_train_threads = 8

    # -------- Network Architecture ----------
#    
#    if gpu_id ==-1:
#        net_kernel_params = {
#            'down_conv_kernels': [
#                [(5, 128), (5, 128)],  # [(kernel_size, num_filters), (kernel_size, num_filters), ...] As many convolustoins in each layer
#                [(5, 256), (5, 256)],
#                [(5, 256), (5, 256)],
#                [(5, 512), (5, 512)],
#            ],
#            'lstm_kernels': [
#                [(5, 128)],  # [(kernel_size, num_filters), (kernel_size, num_filters), ...] As many C-LSTMs in each layer
#                [(5, 256)],
#                [(5, 256)],
#                [(5, 512)],
#            ],
#            'up_conv_kernels': [
#                [(5, 256), (5, 256)],   # [(kernel_size, num_filters), (kernel_size, num_filters), ...] As many convolustoins in each layer
#                [(5, 128), (5, 128)],
#                [(5, 64), (5, 64)],
#                [(5, 32), (5, 32), (1, 1)],
#            ],
#    
#        }
#    elif trial==1:   
#        net_kernel_params = {
#            'down_conv_kernels': [
#                [(5, 50), (5, 50)],
#                [(5, 100), (5, 100)],
#                [(5, 100), (5, 100)],
#                [(5, 200), (5, 200)],
#                [(5, 400), (5, 400)],
#            ],
#            'lstm_kernels': [
#                [(5, 50)],
#                [(5, 100)],
#                [(5, 100)],
#                [(5, 200)],
#                [(5, 400)],                
#            ],
#            'up_conv_kernels': [
#                [(5, 100), (5, 100)],
#                [(5, 100), (5, 100)],
#                [(5, 50), (5, 50)],
#                [(5, 50), (5, 50)],
#                [(5, 20), (5, 20), (1, 1)],
#            ],
#        }
#    else:
#        net_kernel_params = {
#            'down_conv_kernels': [
#                [(5, 100), (5, 100)],
#                [(5, 200), (5, 200)],
#                [(5, 200), (5, 200)],
#                [(5, 400), (5, 400)],
#            ],
#            'lstm_kernels': [
#                [(5, 100)],
#                [(5, 200)],
#                [(5, 200)],
#                [(5, 400)],
#            ],
#            'up_conv_kernels': [
#                [(5, 200), (5, 200)],
#                [(5, 100), (5, 100)],
#                [(5, 50), (5, 50)],
#                [(5, 20), (5, 20), (1, 1)],
#            ],
#        }  


    # -------- Training ----------
#    learning_rate = 0.0001
#    decay_rate=0.96
#    drop_input = True
    num_iterations = 1
    validation_interval = 10
    print_to_console_interval = 10

    # ---------Save and Restore ----------
    load_checkpoint = False
    load_checkpoint_path = ''  # Used only if load_checkpoint is True
    continue_run = False
    save_checkpoint_dir = ROOT_SAVE_DIR
    save_checkpoint_iteration = 100
    save_checkpoint_every_N_hours = 24
    save_checkpoint_max_to_keep = 5

    # ---------Tensorboard-------------
    tb_sub_folder = 'LSTMUNet'
    write_to_tb_interval = 10
    save_log_dir = ROOT_SAVE_DIR

    # ---------Debugging-------------
    dry_run = False  # Default False! Used for testing, when True no checkpoints or tensorboard outputs will be saved
    profile = False

    def __init__(self, params_dict):
        self._override_params_(params_dict)
        if self.gpu_id >= 0:
            if isinstance(self.gpu_id, list):
                os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)[1:-1]
            else:
                os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = "-1"    

        self.train_data_provider = self.data_provider_class(sequence_folder_list=self.root_data_dir,
                                                            image_crop_size=self.crop_size,
                                                            image_reshape_size = self.reshape_size,
                                                            unroll_len=self.unroll_len,
                                                            deal_with_end=0,
                                                            batch_size=self.batch_size,
                                                            queue_capacity=self.train_q_capacity,
                                                            data_format=self.data_format,
                                                            randomize=True,
                                                            num_threads=self.num_train_threads
                                                            )
        self.val_data_provider = self.data_provider_class(sequence_folder_list=self.root_data_dir,
                                                          image_crop_size=self.crop_size,
                                                          image_reshape_size = self.reshape_size, 
                                                          unroll_len=self.unroll_len,
                                                          deal_with_end=0,
                                                          batch_size=self.batch_size,
                                                          queue_capacity=self.train_q_capacity,
                                                          data_format=self.data_format,
                                                          randomize=True,
                                                          num_threads=self.num_val_threads
                                                          )

        now_string = datetime.now().strftime('%Y-%m-%d_%H%M%S')
        if self.load_checkpoint and self.continue_run:
            if os.path.isdir(self.load_checkpoint_path):
                if self.load_checkpoint_path.endswith('tf-ckpt') or self.load_checkpoint_path.endswith('tf-ckpt/'):
                    self.experiment_log_dir = self.experiment_save_dir = os.path.dirname(self.load_checkpoint_path)
                else:
                    self.experiment_log_dir = self.experiment_save_dir = self.load_checkpoint_path
            else:

                save_dir = os.path.dirname(os.path.dirname(self.load_checkpoint_path))
                self.experiment_log_dir = self.experiment_save_dir = save_dir
                self.load_checkpoint_path = os.path.join(self.load_checkpoint_path, 'tf-ckpt')
        else:
            self.experiment_log_dir = os.path.join(self.save_log_dir, self.tb_sub_folder, self.experiment_name,
                                                   now_string)
            self.experiment_save_dir = os.path.join(self.save_checkpoint_dir, self.tb_sub_folder, self.experiment_name,
                                                    now_string)
        if not self.dry_run:
            os.makedirs(self.experiment_log_dir, exist_ok=True)
            os.makedirs(self.experiment_save_dir, exist_ok=True)
#            os.makedirs(os.path.join(self.experiment_log_dir, 'train'), exist_ok=True)
#            os.makedirs(os.path.join(self.experiment_log_dir, 'val'), exist_ok=True)
        self.channel_axis = 1 if self.data_format == 'NCHW' else 3
        if self.profile:
            if 'CUPTI' not in os.environ['LD_LIBRARY_PATH']:
                os.environ['LD_LIBRARY_PATH'] += ':/usr/local/cuda/extras/CUPTI/lib64/'


class CTCInferenceParams(ParamsBase):

    gpu_id = 0  # for CPU ise -1 otherwise gpu id
    model_path = '/home/stormlab/seg/LSTM-UNet-Outputs/Retrained/LSTMUNet/MyRun_SIM/2019-12-03_104028' # download from https://drive.google.com/file/d/1uQOdelJoXrffmW_1OCu417nHKtQcH3DJ/view?usp=sharing
    output_path = '/home/stormlab/seg/Output/sample'
    sequence_path = os.path.join(ROOT_TEST_DATA_DIR, 'tissues')
    filename_format = '*.tif'  # default format for CTC
    reshape_size = (64, 64)

    data_reader = DataHandeling.CTCInferenceReader
    data_format = 'NHWC'  # 'NCHW' or 'NHWC'
    pre_sequence_frames = 4  # Initialize the sequence with first pre_sequence_frames played in reverse

    # ---------Debugging---------

    dry_run = False
    save_intermediate = True
    save_intermediate_path = output_path

    def __init__(self, params_dict: dict = None):
        if params_dict is not None:
            self._override_params_(params_dict)
        self.channel_axis = 1 if self.data_format == 'NCHW' else 3
        if not self.dry_run:
            os.makedirs(self.output_path, exist_ok=True)
            if self.save_intermediate:
                now_string = datetime.now().strftime('%Y-%m-%d_%H%M%S')
                self.save_intermediate_path = os.path.join(self.save_intermediate_path,'IntermediateImages', now_string)
                self.save_intermediate_vis_path = os.path.join(self.save_intermediate_path, 'Softmax')
                self.save_intermediate_label_path = os.path.join(self.save_intermediate_path, 'Labels')
                os.makedirs(self.save_intermediate_path, exist_ok=True)
                os.makedirs(self.save_intermediate_vis_path, exist_ok=True)
                os.makedirs(self.save_intermediate_label_path, exist_ok=True)





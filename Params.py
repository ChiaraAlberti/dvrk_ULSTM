#settings of the parameters
import DataHandeling
import os
from datetime import datetime

ROOT_DATA_DIR = '/home/stormlab/seg/lstm_dataset_complete'
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

    #  ------- Data -------
    data_provider_class = DataHandeling.CTCRAMReaderSequence2D
    root_data_dir = ROOT_DATA_DIR
    crop_size = (64, 64)  # (height, width) preferably height=width 
    reshape_size = (64, 64)
    batch_size = 5
    unroll_len = 5
    data_format = 'NHWC' # either 'NCHW' or 'NHWC'


    # -------- Training ----------
    num_iterations = 100000
    validation_interval = 10
    print_to_console_interval = 10

    # ---------Save and Restore ----------
    load_checkpoint = False
    load_checkpoint_path = ''  # Used only if load_checkpoint is True
    continue_run = False
    save_checkpoint_dir = ROOT_SAVE_DIR
    save_checkpoint_iteration = 5000
    save_checkpoint_every_N_hours = 24
    save_checkpoint_max_to_keep = 5

    # ---------Tensorboard-------------
    tb_sub_folder = 'LSTMUNet'
    write_to_tb_interval = 100
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


        #initialization of dataHandeling classs
        self.data_provider = self.data_provider_class(sequence_folder_list=self.root_data_dir,
                                                            image_crop_size=self.crop_size,
                                                            image_reshape_size = self.reshape_size,
                                                            unroll_len=self.unroll_len,
                                                            batch_size=self.batch_size,
                                                            data_format=self.data_format,
                                                            randomize=True,
                                                            )


#        now_string = datetime.now().strftime('%Y-%m-%d_%H%M%S')
        now_string = 'Pretrained_model'
        
        #initialization of the various dir folders
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
                self.experiment_k_fold_dir = os.path.join(self.save_checkpoint_dir, self.tb_sub_folder, 'k_fold')
        else:
            self.experiment_log_dir = os.path.join(self.save_log_dir, self.tb_sub_folder, self.experiment_name,
                                                   now_string)
            self.experiment_save_dir = os.path.join(self.save_checkpoint_dir, self.tb_sub_folder, self.experiment_name,
                                                    now_string)
            self.experiment_k_fold_dir = os.path.join(self.save_checkpoint_dir, self.tb_sub_folder, 'k_fold')
        if not self.dry_run:
            os.makedirs(self.experiment_log_dir, exist_ok=True)
            os.makedirs(self.experiment_save_dir, exist_ok=True)

        self.channel_axis = 1 if self.data_format == 'NCHW' else 3
        if self.profile:
            if 'CUPTI' not in os.environ['LD_LIBRARY_PATH']:
                os.environ['LD_LIBRARY_PATH'] += ':/usr/local/cuda/extras/CUPTI/lib64/'





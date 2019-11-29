import argparse
import os
import pickle
import scipy.ndimage
import cv2
import numpy as np
import tensorflow as tf
import Networks as Nets
from Params_our import CTCInferenceParams
from distutils.util import strtobool
import DataHandeling_our as DataHandeling
import sys
from utils import log_print, get_model, bbox_crop, bbox_fill

try:
    import tensorflow.python.keras as k
except AttributeError:
    import tensorflow.keras as k
if not tf.__version__.split('.')[0] == '2':
    raise ImportError(f'Required tensorflow version 2.x. current version is: {tf.__version__}')

#
#with open(os.path.join('/home/stormlab/seg/LSTM-UNet-Outputs/Retrained/LSTMUNet/MyRun_SIM/2019-11-19_104459', 'model_params.pickle'), 'rb') as fobj:
#    model_dict = pickle.load(fobj)
#model_cls = getattr(Nets, model_dict['name'])
    
with open(os.path.join('/home/stormlab/seg/LSTM-UNet-Outputs/LSTMUNet/MyRun_SIM/2019-11-12_174731-10000it', 'model_params.pickle'), 'rb') as fobj:
    model_dict = pickle.load(fobj)
model_cls = getattr(Nets, model_dict['name'])



device = '/gpu:0' if CTCInferenceParams.gpu_id >= 0 else '/cpu:0'
with tf.device(device):
    model = model_cls(*model_dict['params'], data_format=CTCInferenceParams.data_format, pad_image=True)
    model.load_weights(os.path.join('/home/stormlab/seg/LSTM-UNet-Outputs/LSTMUNet/MyRun_SIM/2019-11-12_174731-10000it', 'model.ckpt'))
    model.get_weights()
    model.summary()
    
    

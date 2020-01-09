import random
import tensorflow as tf
import os
import glob
import cv2
import numpy as np
import pickle
import utils
import time
import scipy
from PIL import Image

class CTCRAMReaderSequence2D(object):
    def __init__(self, sequence_folder_list, image_crop_size=(128, 128), image_reshape_size=(128,128), unroll_len=7, deal_with_end=0, batch_size=4,
                 queue_capacity=32, num_threads=3, data_format='NCHW', randomize=True, elastic_augmentation=False):
        if not isinstance(image_crop_size, tuple):
            image_crop_size = tuple(image_crop_size)
        self.coord = None
        self.unroll_len = unroll_len
        self.sequence_data = {}
        self.sequence_folder = sequence_folder_list
        self.elastic_augmentation = elastic_augmentation
        self.sub_seq_size = image_crop_size
        self.reshape_size = image_reshape_size
        self.dist_sub_seq_size = (2,) + image_crop_size
        self.deal_with_end = deal_with_end
        self.batch_size = batch_size
        self.queue_capacity = queue_capacity
        self.data_format = data_format
        self.randomize = randomize
        self.num_threads = num_threads
        self.width_shift_range = 0.1
        self.height_shift_range = 0.1
        self.used_masks = []
    
        np.random.seed(1)


    @staticmethod
    def _adjust_brightness_(image, delta):
        out_img = image + delta
        return out_img
    
    @staticmethod
    def _adjust_contrast_(image, factor):
        img_mean = image.mean()
        out_img = (image - img_mean) * factor + img_mean
        return out_img
    
    def read_batch(self):
        sequence_folder = self.sequence_folder
        with open(os.path.join(sequence_folder, 'full_csv_prova.pkl'), 'rb') as fobj:
            metadata = pickle.load(fobj)
    
        filename_list = metadata['filelist']
        
        if len(metadata['shape'])== 3:
            img_size = self.reshape_size + (metadata['shape'][-1],)
#            all_images = np.zeros((self.batch_size, self.unroll_len, img_size[0], img_size[1], img_size[2])).astype(np.float32)
            all_images = []
        else:
            img_size = self.reshape_size      
#            all_images = np.zeros((self.batch_size, self.unroll_len, img_size[0], img_size[1])).astype(np.float32)
            all_images = []
#        all_seg = np.zeros((self.batch_size, self.unroll_len, img_size[0], img_size[1])).astype(np.float32)
        all_seg = []
    
        valid_masks = [i for i, x in enumerate(filename_list) if x[1] != 'None']
        valid_masks = [i for i in valid_masks if i not in self.used_masks]
        batch_index = random.sample(valid_masks, self.batch_size)
        self.used_masks.extend(batch_index)
        valid_future_masks = [i for i in valid_masks if i not in self.used_masks]
        if len(valid_future_masks)<self.batch_size:
            self.used_masks = []
            is_last_batch = np.array([0, 0, 0, 0]).astype(np.float32)
            is_last_batch = is_last_batch.tolist()
        else:
            is_last_batch = np.array([1, 1, 1, 1]).astype(np.float32)
            is_last_batch = is_last_batch.tolist()
        
        for i in range (0, self.batch_size):
            image_seq = []
            seg_seq = []
            if len(metadata['shape'])==3:
                    img_size = self.reshape_size + (metadata['shape'][-1],)
            else:
                img_size = self.reshape_size
            if img_size[0] - self.sub_seq_size[0] > 0:
                crop_y = np.random.randint(0, img_size[0] - self.sub_seq_size[0]) if self.randomize else 0
            else:
                crop_y = 0
            if img_size[1] - self.sub_seq_size[1] > 0:
                crop_x = np.random.randint(0, img_size[1] - self.sub_seq_size[1]) if self.randomize else 0
            else:
                crop_x = 0
            
            crop_y_stop = crop_y + self.sub_seq_size[0]
            crop_x_stop = crop_x + self.sub_seq_size[1]
            flip = np.random.randint(0, 2, 2) if self.randomize else [0, 0]
            rotate = np.random.randint(0, 4) if self.randomize else 0
            
            if self.width_shift_range or self.height_shift_range:
                if self.width_shift_range:
                    width_shift_range = random.uniform(-self.width_shift_range * img_size[1], self.width_shift_range * img_size[1])
                if self.height_shift_range:
                    height_shift_range = random.uniform(-self.height_shift_range * img_size[1], self.height_shift_range * img_size[1])
    
            for j in range (0, self.unroll_len):
                img = cv2.imread(os.path.join(sequence_folder, 'train', filename_list[batch_index[i]- self.unroll_len + j + 1][0]), -1)
                if img is None:
                    raise ValueError('Could not load image: {}'.format(os.path.join(sequence_folder, filename_list[batch_index[i]- self.unroll_len + j +1][0])))
                img = cv2.normalize(img.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX)
                img = cv2.resize(img, self.reshape_size, interpolation = cv2.INTER_AREA)
                if filename_list[batch_index[i]- self.unroll_len + j + 1][1] == 'None':
                    seg = np.ones(img.shape[:2]).astype(np.float32) * (0)
                else:
                    seg =cv2.imread(os.path.join(sequence_folder, 'labels', filename_list[batch_index[i] - self.unroll_len + j + 1][1]), -1)
                    seg = cv2.normalize(seg.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX)
                seg = cv2.resize(seg, self.reshape_size, interpolation = cv2.INTER_AREA)
                
                img_crop = img[crop_y:crop_y_stop, crop_x:crop_x_stop]
                img_max = img_crop.max()
                seg_crop = seg[crop_y:crop_y_stop, crop_x:crop_x_stop]
                
                if self.randomize:
                    # contrast factor between [0.5, 1.5]
                    random_constrast_factor = np.random.rand() + 0.5
                    # random brightness delta plus/minus 10% of maximum value
                    random_brightness_delta = (np.random.rand() - 0.5) * 0.2 * img_max
                    img_crop = self._adjust_contrast_(img_crop, random_constrast_factor)
                    img_crop = self._adjust_brightness_(img_crop, random_brightness_delta)
                    
                if flip[0]:
                    img_crop = cv2.flip(img_crop, 0)
                    seg_crop = cv2.flip(seg_crop, 0)
                if flip[1]:
                    img_crop = cv2.flip(img_crop, 1)
                    seg_crop = cv2.flip(seg_crop, 1)
                if rotate > 0:
                    img_crop = np.rot90(img_crop, rotate)
                    seg_crop = np.rot90(seg_crop, rotate)
                    
                img_crop = scipy.ndimage.shift(img_crop, [width_shift_range, height_shift_range])
                seg_crop = scipy.ndimage.shift(seg_crop, [width_shift_range, height_shift_range])
                
                image_seq.append(img_crop)
                seg_seq.append(seg_crop)
                
            all_images.append(image_seq)
            all_seg.append(seg_seq)
                
        image_batch = tf.stack(all_images)
        seg_batch = tf.stack(all_seg)
        is_last_batch = tf.stack(is_last_batch)
        
        if self.data_format == 'NHWC':
            image_batch = tf.expand_dims(image_batch, 4)
            seg_batch = tf.expand_dims(seg_batch, 4)
        elif self.data_format == 'NCHW':
            image_batch = tf.expand_dims(image_batch, 2)
            seg_batch = tf.expand_dims(seg_batch, 2)
        else:
            raise ValueError()
        return image_batch, seg_batch, is_last_batch



class CTCInferenceReader(object):

    def __init__(self, data_path, filename_format='*.tif', normalize=True, pre_sequence_frames=0):

        file_list = glob.glob(os.path.join(data_path, filename_format))
        if len(file_list) == 0:
            raise ValueError('Could not read images from: {}'.format(os.path.join(data_path, filename_format)))

        def gen():
            file_list.sort()

            file_list_pre = file_list[:pre_sequence_frames].copy()
            file_list_pre.reverse()
            full_file_list = file_list_pre + file_list
            for file in full_file_list:
                img = cv2.imread(file, -1).astype(np.uint16)
                img = img.astype(np.float32)
                if img is None:
                    raise ValueError('Could not read image: {}'.format(file))
                if normalize:
                    img = (img - img.mean())
                    img = img / (img.std())
                yield img

        self.dataset = tf.data.Dataset.from_generator(gen, tf.float32)

if __name__ == "__main__":
    CTCInferenceReader.unit_test()
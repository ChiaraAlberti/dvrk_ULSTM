import numpy as np
import random
import os.path
import cv2
import scipy
import threading
import functools

##data generator that uses threads in order to create continuosly a batch of images with shape(batch_size, unirll_length, reshape_size)


class threadsafe_iterator(object):
    def __init__(self, iterator):
        self.iterator = iterator
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.iterator)

def threadsafe_generator(func):
    @functools.wraps(func)
    def gen(*a, **kw):
        return threadsafe_iterator(func(*a, **kw))
    return gen

class DataSet():

    def __init__(self,  filelist_ind, filename, sequence_folder, shuffle = False, image_crop_size=(128, 128), image_reshape_size=(128,128), unroll_len=7,batch_size=4,
                 data_format='NCHW', randomize=True):
        """Constructor.
        seq_length = (int) the number of frames to consider
        class_limit = (int) number of classes to limit the data to.
            None = no limit.
        """
        self.filelist_ind = filelist_ind
        self.filename = filename
        self.shuffle = shuffle
        self.unroll_len = unroll_len
        self.sub_seq_size = image_crop_size
        self.reshape_size = image_reshape_size
        self.sequence_folder = sequence_folder
        self.batch_size = batch_size
        self.data_format = data_format
        self.randomize = randomize
        self.width_shift_range = 0.1
        self.height_shift_range = 0.1
        np.random.seed(1)# max number of frames a video can have for us to use it


    @threadsafe_generator
    #in this method the sequences are piled into batch 
    def frame_generator(self):
        while 1:
            image_batch, seg_batch = [], []
            for _ in range(0, self.batch_size):
                sample = random.choice(self.filelist_ind)
                sequence_image, sequence_seg = self.build_image_sequence(sample)
                image_batch.append(sequence_image)
                seg_batch.append(sequence_seg)
            print((np.expand_dims(np.array(image_batch), -1)).shape)
            yield (np.expand_dims(np.array(image_batch), -1), np.expand_dims(np.array(seg_batch), -1))
            
    @staticmethod
    def _adjust_brightness_(image, delta):
        out_img = image + delta
        return out_img
    
    @staticmethod
    def _adjust_contrast_(image, factor):
        img_mean = image.mean()
        out_img = (image - img_mean) * factor + img_mean
        return out_img
    
#image augmentation with crop, contrast and brightness adjust, rotation, flip and shift        
    def image_aug(self, img, seg, crop_y, crop_x, crop_y_stop, crop_x_stop, flip, rotate, width_shift_range, height_shift_range):
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
        
#        img_crop = np.expand_dims(img_crop, axis=-1)
#        seg_crop = np.expand_dims(seg_crop, axis=-1)
        return img_crop, seg_crop

#method that build the temporal image sequences and apply data augmentation
    def build_image_sequence(self, sample):
        sequence_image = []
        sequence_seg = []
        if self.reshape_size[0] - self.sub_seq_size[0] > 0:
            crop_y = np.random.randint(0, self.reshape_size[0] - self.sub_seq_size[0]) if self.randomize else 0
        else:
            crop_y = 0
        if self.reshape_size[1] - self.sub_seq_size[1] > 0:
            crop_x = np.random.randint(0, self.reshape_size[1] - self.sub_seq_size[1]) if self.randomize else 0
        else:
            crop_x = 0
        
        crop_y_stop = crop_y + self.sub_seq_size[0]
        crop_x_stop = crop_x + self.sub_seq_size[1]
        flip = np.random.randint(0, 2, 2) if self.randomize else [0, 0]
        rotate = np.random.randint(0, 4) if self.randomize else 0
#            jump = np.random.randint(1,4) if self.randomize else 1
        jump = 1 
        
        if self.width_shift_range or self.height_shift_range:
            if self.width_shift_range:
                width_shift_range = random.uniform(-self.width_shift_range * self.reshape_size[1], self.width_shift_range * self.reshape_size[1])
            if self.height_shift_range:
                height_shift_range = random.uniform(-self.height_shift_range * self.reshape_size[1], self.height_shift_range * self.reshape_size[1])
        for j in range(0, self.unroll_len):
            img = cv2.imread(os.path.join(self.sequence_folder, 'train', self.filename[sample - (self.unroll_len - j - 1)*jump][0]), -1)
            if img is None:
                raise ValueError('Could not load image: {}'.format(os.path.join(self.sequence_folder, self.filename[sample - (self.unroll_len - j -1)*jump][0])))
            img = cv2.normalize(img.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX)
            if self.filename[sample- (self.unroll_len - j - 1)*jump][1] == 'None':
                seg = np.ones(img.shape[:2]).astype(np.float32) * (0)
            else:
                seg =cv2.imread(os.path.join(self.sequence_folder, 'labels', self.filename[sample - (self.unroll_len - j - 1)*jump][1]), -1)
                seg = cv2.normalize(seg.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX)
            img = cv2.resize(img, self.reshape_size, interpolation = cv2.INTER_AREA)
            seg = cv2.resize(seg, self.reshape_size, interpolation = cv2.INTER_AREA)
            img_crop, seg_crop = self.image_aug(img, seg, crop_y, crop_x, crop_y_stop, crop_x_stop, flip, rotate, width_shift_range, height_shift_range)
                
            sequence_image.append(img)
            sequence_seg.append(seg)
            
        
        return (sequence_image, sequence_seg)
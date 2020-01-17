import numpy as np
import random
import cv2
import os
import scipy

from tensorflow.python.keras.utils.data_utils import Sequence

#try:
#    # noinspection PyPackageRequirements
#    import tensorflow.python.keras as k
#except AttributeError:
#    # noinspection PyPackageRequirements,PyUnresolvedReferences
#    import tensorflow.keras as k

class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, filename, sequence_folder, shuffle = False, image_crop_size=(128, 128), image_reshape_size=(128,128), unroll_len=7,batch_size=4,
                 data_format='NCHW', randomize=True):
        'Initialization'
        self.list_IDs = list_IDs
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
        self.on_epoch_end()
        np.random.seed(1)

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        all_images, all_seg = self.__data_generation(list_IDs_temp)
        print('Image', all_images.shape)
        print('Seg', all_seg.shape)
        return all_images, all_seg

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    @staticmethod
    def _adjust_brightness_(image, delta):
        out_img = image + delta
        return out_img
    
    @staticmethod
    def _adjust_contrast_(image, factor):
        img_mean = image.mean()
        out_img = (image - img_mean) * factor + img_mean
        return out_img

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        image_batch = np.zeros((self.batch_size, self.unroll_len, *self.reshape_size, 1))
        seg_batch = np.zeros((self.batch_size, self.unroll_len, *self.reshape_size, 1))

        # Generate data        
        for i in range (0, self.batch_size):
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
    
            for j in range (0, self.unroll_len):
    
                img = cv2.imread(os.path.join(self.sequence_folder, 'train', self.filename[list_IDs_temp[i] - (self.unroll_len - j - 1)*jump][0]), -1)
                if img is None:
                    raise ValueError('Could not load image: {}'.format(os.path.join(self.sequence_folder, self.filename[list_IDs_temp[i] - (self.unroll_len - j -1)*jump][0])))
                img = cv2.normalize(img.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX)
                if self.filename[list_IDs_temp[i]- (self.unroll_len - j - 1)*jump][1] == 'None':
                    seg = np.ones(img.shape[:2]).astype(np.float32) * (0)
                else:
                    seg =cv2.imread(os.path.join(self.sequence_folder, 'labels', self.filename[list_IDs_temp[i] - (self.unroll_len - j - 1)*jump][1]), -1)
                    seg = cv2.normalize(seg.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX)
                    
                img = cv2.resize(img, self.reshape_size, interpolation = cv2.INTER_AREA)
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
                
                img_crop = np.expand_dims(img_crop, axis=-1)
                seg_crop = np.expand_dims(seg_crop, axis=-1)
                
                image_batch[i, j] = img_crop
                seg_batch[i, j] = seg_crop


        return image_batch, seg_batch

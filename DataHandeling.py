import random
import tensorflow as tf
import os
import cv2
import numpy as np
import pickle
import scipy
from sklearn.model_selection import train_test_split
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

class CTCRAMReaderSequence2D(object):
    def __init__(self, sequence_folder_list, image_crop_size=(128, 128), image_reshape_size=(128,128), unroll_len=7, batch_size=4,
                 data_format='NCHW', randomize=True, elastic_augmentation=False):
        if not isinstance(image_crop_size, tuple):
            image_crop_size = tuple(image_crop_size)
        self.unroll_len = unroll_len
        self.sequence_folder = sequence_folder_list
        self.elastic_augmentation = elastic_augmentation
        self.sub_seq_size = image_crop_size
        self.reshape_size = image_reshape_size
        self.batch_size = batch_size
        self.data_format = data_format
        self.randomize = randomize
        self.width_shift_range = 0.05
        self.height_shift_range = 0.05
        self.used_masks_train = []
        self.used_masks_val = []
        self.used_masks_val = []
        self.mode = 'random'
#        np.random.seed(1)
        #initialization of the different parts of dataset
        self.valid_list_train, self.valid_list_val, self.valid_list_test, self.metadata = self._dataset_split(self.mode)
        #define number of train and val steps per epoch
        self.num_steps_per_epoch = int(np.floor(len(self.valid_list_train)/self.batch_size))
        self.num_steps_per_val = int(np.floor(len(self.valid_list_val)/self.batch_size))
        #create a queue for the testings
        self.q, self.q_best = self.create_queue()


    @staticmethod
    def _adjust_brightness_(image, delta):
        out_img = image + delta
        return out_img
    
    @staticmethod
    def _adjust_contrast_(image, factor):
        img_mean = image.mean()
        out_img = (image - img_mean) * factor + img_mean
        return out_img
    
    @staticmethod
    #create indices for the grid elastic transformation
    def get_indices(shape, alpha, sigma, random_state=None):    
        random_state = np.random.RandomState(None)          
        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha    
        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
        indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))        
        return indices
    
    #create the training-test pairs for kfold validation
    def split_k_fold(self, kf):
        sequence_folder = self.sequence_folder
        with open(os.path.join(sequence_folder, 'dataset.pkl'), 'rb') as fobj:
            metadata = pickle.load(fobj)
        filename_list = metadata['filelist']
        valid_masks = [i for i, x in enumerate(filename_list) if x[1] != 'None']
        random.shuffle(valid_masks)
        valid_masks_array = np.array(valid_masks)
        valid_list_train = []
        valid_list_test = []
        for train_index, test_index in kf.split(valid_masks_array):
            X_train, X_test = valid_masks_array[train_index], valid_masks_array[test_index]
            valid_list_train.append(X_train)
            valid_list_test.append(X_test)
        return valid_list_train, valid_list_test        
    
    #splits the index of the labeled masks into different sets for training, validation and test
    def _dataset_split(self, mode):
        sequence_folder = self.sequence_folder
        with open(os.path.join(sequence_folder, 'dataset.pkl'), 'rb') as fobj:
            metadata = pickle.load(fobj)
        filename_list = metadata['filelist']
        valid_masks = [i for i, x in enumerate(filename_list) if x[1] != 'None']
        #random splits
        if mode == 'random':
            valid_list_training, valid_list_test = train_test_split(valid_masks, test_size= 0.1)
            valid_list_train, valid_list_val = train_test_split(valid_list_training, test_size= 0.2)
        #dataset temporal sequence is kept
        elif mode == 'ordered':
            valid_list_training = valid_masks[int(round(0.1*len(valid_masks))):]
            valid_list_test = valid_masks[:int(round(0.1*len(valid_masks)))]
            valid_list_train = valid_list_training[int(round(0.1*len(valid_list_training))):]
            valid_list_val = valid_list_training[:int(round(0.1*len(valid_list_training)))]
        #data are divided in batch and then assigned to train, val and test
        elif mode == 'by_batch':
            list_batch = []
            n = int(np.floor(len(valid_masks)*0.1))
            for i in range(0, len(valid_masks), n):
                list_batch.append(valid_masks[i:i + n])
            random.shuffle(list_batch) 
            valid_list_train = [item for sublist in list_batch[2:] for item in sublist]
            valid_list_val = list_batch[0]
            valid_list_test = list_batch[1]                   
        return valid_list_train, valid_list_val, valid_list_test, metadata

    #create queues for testing
    def create_queue(self):
        q = tf.queue.FIFOQueue(len(self.valid_list_test), dtypes = tf.float32, shapes = ())
        q_best = tf.queue.FIFOQueue(len(self.valid_list_test), dtypes = tf.float32, shapes = ())
        return q, q_best
    
    #create the batch of images and apply data augmentation
    def read_batch(self, flag, kfold, list_index):
        if len(self.metadata['shape'])== 3:
            img_size = self.reshape_size + (self.metadata['shape'][-1],)
            all_images = []
        else:
            img_size = self.reshape_size      
            all_images = []
        all_seg = []
        
        #choose random samples from dataset
        if flag == 'train':
            if kfold == True:              
                valid_masks = [i for i in list_index if i not in self.used_masks_train]
            else:                
                valid_masks = [i for i in self.valid_list_train if i not in self.used_masks_train]
            batch_index = random.sample(valid_masks, self.batch_size)
            self.used_masks_train.extend(batch_index)
            valid_future_masks = [i for i in valid_masks if i not in self.used_masks_train]
        else:
            if kfold == True:
                valid_masks = [i for i in list_index if i not in self.used_masks_val]
            else:
                valid_masks = [i for i in self.valid_list_val if i not in self.used_masks_val]
            batch_index = random.sample(valid_masks, self.batch_size)
            self.used_masks_val.extend(batch_index)
            valid_future_masks = [i for i in valid_masks if i not in self.used_masks_val]
            
        if len(valid_future_masks)<self.batch_size:
            if flag == 'train':
                self.used_masks_train = []
            else:
                self.used_masks_val = []
            is_last_batch = np.zeros(self.batch_size).astype(np.float32)
            is_last_batch = is_last_batch.tolist()
        else:
            is_last_batch = np.ones(self.batch_size).astype(np.float32)
            is_last_batch = is_last_batch.tolist()
        
        #iterates along the sequences
        for i in range (0, self.batch_size):
            image_seq = []
            seg_seq = []
            
            #define crop boundaries
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
            
            #define data augmentation values
            flip = np.random.randint(0, 2, 2) if self.randomize else [0, 0]
            rotate = np.random.randint(0, 4) if self.randomize else 0
            dir_seq = random.choice(['forward', 'backward']) if self.randomize else 1
            jump = np.random.randint(1,4) if self.randomize else 1
            indices = self.get_indices(self.reshape_size, self.reshape_size[0]*3, self.reshape_size[0]*0.15)  
            
            if self.width_shift_range or self.height_shift_range:
                if self.width_shift_range:
                    width_shift_range = random.uniform(-self.width_shift_range * img_size[1], self.width_shift_range * img_size[1])
                if self.height_shift_range:
                    height_shift_range = random.uniform(-self.height_shift_range * img_size[1], self.height_shift_range * img_size[1])
                    
            #iterate along the single sequence
            for j in range (0, self.unroll_len):
                #load image and label
                if dir_seq == 'backward':
                    #read images 
                    img = cv2.imread(os.path.join(self.sequence_folder, 'train', self.metadata['filelist'][batch_index[i] - (self.unroll_len - j - 1)*jump][0]), -1)
                    if img is None:
                        raise ValueError('Could not load image: {}'.format(os.path.join(self.sequence_folder, self.metadata['filelist'][batch_index[i] - (self.unroll_len - j -1)*jump][0])))
                    #if label does not exist, create a matrix of zeros
                    if self.metadata['filelist'][batch_index[i]- (self.unroll_len - j - 1)*jump][1] == 'None':
                        seg = np.ones(img.shape[:2]).astype(np.float32) * (0)
                    else:
                        #read labels
                        seg =cv2.imread(os.path.join(self.sequence_folder, 'labels', self.metadata['filelist'][batch_index[i] - (self.unroll_len - j - 1)*jump][1]), -1)
                        if seg is None:
                            print('Attention')
                        seg = cv2.resize(seg, self.reshape_size, interpolation = cv2.INTER_AREA)
                        seg = cv2.normalize(seg.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX)
                else: 
                    img = cv2.imread(os.path.join(self.sequence_folder, 'train', self.metadata['filelist'][batch_index[i] + (self.unroll_len - j - 1)*jump][0]), -1)
                    if img is None:
                        raise ValueError('Could not load image: {}'.format(os.path.join(self.sequence_folder, self.metadata['filelist'][batch_index[i] + (self.unroll_len - j -1)*jump][0])))
                    if self.metadata['filelist'][batch_index[i] + (self.unroll_len - j - 1)*jump][1] == 'None':
                        seg = np.ones(img.shape[:2]).astype(np.float32) * (0)
                    else:
                        seg =cv2.imread(os.path.join(self.sequence_folder, 'labels', self.metadata['filelist'][batch_index[i] + (self.unroll_len - j - 1)*jump][1]), -1)
                        if seg is None:
                            print('Attention')
                        seg = cv2.resize(seg, self.reshape_size, interpolation = cv2.INTER_AREA)
                        seg = cv2.normalize(seg.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX)
                img = cv2.resize(img, self.reshape_size, interpolation = cv2.INTER_AREA)
                img = cv2.normalize(img.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX)
                
                #apply data augmentation
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
                    
                    #flip on x
                    if flip[0]:
                        img_crop = cv2.flip(img_crop, 0)
                        seg_crop = cv2.flip(seg_crop, 0)
                    #flip on y
                    if flip[1]:
                        img_crop = cv2.flip(img_crop, 1)
                        seg_crop = cv2.flip(seg_crop, 1)
                    #90 degrees rotation
                    if rotate > 0:
                        img_crop = np.rot90(img_crop, rotate)
                        seg_crop = np.rot90(seg_crop, rotate)
                    
                    #elastic augmentation
                    img_crop = map_coordinates(img_crop, indices, order=1, mode='reflect').reshape(img_crop.shape)
                    seg_crop = map_coordinates(seg_crop, indices, order=1, mode='reflect').reshape(seg_crop.shape)   
                    #shift
                    img_crop = scipy.ndimage.shift(img_crop, [width_shift_range, height_shift_range])
                    seg_crop = scipy.ndimage.shift(seg_crop, [width_shift_range, height_shift_range])
                    
                #binarization of the label
                thresh, seg_crop = cv2.threshold(seg_crop,0.5,1,cv2.THRESH_BINARY)
                #append images in sequences
                image_seq.append(img_crop)
                seg_seq.append(seg_crop)
             #append sequences in the batch   
            all_images.append(image_seq)
            all_seg.append(seg_seq)
        
        #create tf stack        
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

    #calculate the number of tests iterations
    def num_test(self):
        return int(np.floor(len(self.valid_list_test) / self.batch_size))
    
    #enqueue the index of the test sets    
    def enqueue_index(self, type_test, test_index):
        if test_index is None:
            if type_test == 'best_test':
                for i in range(0, len(self.valid_list_test)):
                    self.q_best.enqueue(self.valid_list_test[i])
            else:
                for i in range(0, len(self.valid_list_test)):
                    self.q.enqueue(self.valid_list_test[i])
        else:
            self.q = tf.queue.FIFOQueue(len(test_index), dtypes = tf.float32, shapes = ())
            for i in range(0, len(test_index)):
                self.q.enqueue(test_index[i])
    
    #read the images in order to test     
    def read_new_image(self, type_test):
        if type_test == 'best_test':
            index = self.q_best.dequeue_many(self.batch_size)
        elif type_test == 'gif_test':
            index = self.valid_list_test[0: self.batch_size]
        else:
            index = self.q.dequeue_many(self.batch_size)

        image_batch = []
        seg_batch = []
        for j in range(0, self.batch_size):
            image_seq = []
            seg_seq = []
            for i in range(0, self.unroll_len):
                img = cv2.imread(os.path.join(self.sequence_folder, 'train', self.metadata['filelist'][index[j] - (self.unroll_len - i - 1)][0]), -1)
                img = cv2.resize(img, self.reshape_size, interpolation = cv2.INTER_AREA)
                if img is None:
                    raise ValueError('Could not load image: {}'.format(os.path.join(self.sequence_folder, self.metadata['filelist'][index[j] - (self.unroll_len - i - 1)][0])))
                img = cv2.normalize(img.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX)
                if self.metadata['filelist'][index[j] - (self.unroll_len - i - 1)][1] == 'None':
                    seg = np.ones(img.shape[:2]).astype(np.float32) * (0)
                else:
                    seg =cv2.imread(os.path.join(self.sequence_folder, 'labels', self.metadata['filelist'][index[j] - (self.unroll_len - i - 1)][1]), -1)
                    seg = cv2.resize(seg, self.reshape_size, interpolation = cv2.INTER_AREA)
                    seg = cv2.normalize(seg.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX)
                    thresh, seg = cv2.threshold(seg,0.5,1,cv2.THRESH_BINARY)
                image_seq.append(img)
                seg_seq.append(seg)
            image_batch.append(image_seq)
            seg_batch.append(seg_seq)
        
        image_batch = tf.expand_dims(image_batch, 4)
        seg_batch = tf.expand_dims(seg_batch, 4)

        return image_batch, seg_batch
        
    #read the images in order to do inference     
    def read_real_time(self):
        index = self.q.dequeue()
        image_seq = []
        seg_seq = []
        for i in range(0, self.unroll_len):
            img = cv2.imread(os.path.join(self.sequence_folder, 'train', self.metadata['filelist'][index - (self.unroll_len - i - 1)][0]), -1)
            if img is None:
                raise ValueError('Could not load image: {}'.format(os.path.join(self.sequence_folder, self.metadata['filelist'][index - (self.unroll_len - i - 1)][0])))
            img = cv2.resize(img, self.reshape_size, interpolation = cv2.INTER_AREA)
            img = cv2.normalize(img.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX)
            if self.metadata['filelist'][index - (self.unroll_len - i - 1)][1] == 'None':
                seg = np.ones(img.shape[:2]).astype(np.float32) * (0)
            else:
                seg =cv2.imread(os.path.join(self.sequence_folder, 'labels', self.metadata['filelist'][index - (self.unroll_len - i - 1)][1]), -1)
                seg = cv2.resize(seg, self.reshape_size, interpolation = cv2.INTER_AREA)
                seg = cv2.normalize(seg.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX)
                thresh, seg = cv2.threshold(seg,0.5,1,cv2.THRESH_BINARY)
            image_seq.append(img)
            seg_seq.append(seg)
        
        image_seq = tf.expand_dims(image_seq, 0)
        seg_seq = tf.expand_dims(seg_seq, 0)
        image_seq = tf.expand_dims(image_seq, 4)
        seg_seq = tf.expand_dims(seg_seq, 4)

        return image_seq, seg_seq
        
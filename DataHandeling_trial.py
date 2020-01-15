import random
import tensorflow as tf
import os
import glob
import cv2
import queue
import threading
import numpy as np
import pickle
import utils
import time
import scipy
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from PIL import Image

class CTCRAMReaderSequence2D(object):
    def __init__(self, sequence_folder_list, image_crop_size=(128, 128), image_reshape_size=(128,128), unroll_len=7, deal_with_end=0, batch_size=4,
                 queue_capacity=32, num_threads=3,
                 data_format='NCHW', randomize=True, elastic_augmentation=False):
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

        self.q_list, self.q_stat_list = self._create_queues()
        np.random.seed(1)

    def _read_sequence_to_ram_(self):
        sequence_folder = self.sequence_folder
        utils.log_print('Reading Sequence')
        with open(os.path.join(sequence_folder, 'full_csv_prova.pkl'), 'rb') as fobj:
            metadata = pickle.load(fobj)

        filename_list = metadata['filelist']
        valid_masks = [i for i, x in enumerate(filename_list) if x[1] != 'None']
        if len(metadata['shape'])== 3:
            img_size = self.reshape_size + (metadata['shape'][-1],)
            all_images = np.zeros((len(valid_masks)*self.unroll_len, img_size[0], img_size[1], img_size[2])).astype(np.float32)
        else:
            img_size = self.reshape_size      
            all_images = np.zeros((len(valid_masks)*self.unroll_len, img_size[0], img_size[1])).astype(np.float32)
        all_seg = np.zeros((len(valid_masks)*self.unroll_len, img_size[0], img_size[1])).astype(np.float32)
#        num_ones = 0
#        num_zeros = 0
        t=0
        file_list =[]
        for _, index in enumerate(valid_masks):
            for j in range(0, self.unroll_len):
                img = cv2.imread(os.path.join(sequence_folder, 'train', 
                                              filename_list[index - self.unroll_len + j +1][0]), -1)
                if img is None:
                    raise ValueError('Could not load image: {}'.format(os.path.join(sequence_folder, filename_list[index - self.unroll_len + j][0])))
#            img = img.astype(np.float32)
#            img = (img - img.mean()) / (img.std())
                img = cv2.normalize(img.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX)
                img = cv2.resize(img, self.reshape_size, interpolation = cv2.INTER_AREA)
                if filename_list[index - self.unroll_len + j +1][1] == 'None':
                    seg = np.ones(img.shape[:2]) * (0)
                else:
                    seg = cv2.imread(os.path.join(sequence_folder, 'labels', 
                                                  filename_list[index - self.unroll_len + j +1][1]), -1)
                    seg = cv2.normalize(seg.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX)
                seg = cv2.resize(seg, self.reshape_size, interpolation = cv2.INTER_AREA)
#            num_ones = num_ones + np.count_nonzero(seg)
#            num_zeros = num_zeros + (self.reshape_size[0]*self.reshape_size[1] - np.count_nonzero(seg))
                all_images[t] = img
                all_seg[t] = seg
                file_list.append((filename_list[index - self.unroll_len + j +1][0], filename_list[index - self.unroll_len + j+1][1]))
                t= t+1
#        print(num_ones)
#        print(num_zeros)
        self.sequence_data[sequence_folder] = {'images': np.array(all_images), 'segs': np.array(all_seg), 'metadata': metadata, 'file_list': file_list}

    def _read_sequence_data(self):
        sequence_folder = self.sequence_folder
        # if isinstance(sequence_folder, tuple):
        #     sequence_folder = sequence_folder[0]
        return self.sequence_data[sequence_folder], sequence_folder

    def _read_sequence_metadata(self):
        sequence_folder = random.choice(self.sequence_folder_list)
        if isinstance(sequence_folder, tuple):
            sequence_folder = sequence_folder[0]
        with open(os.path.join(sequence_folder, 'lstm_baseline_ds_bw.pkl'), 'rb') as fobj:
            metadata = pickle.load(fobj)
        return metadata

    @staticmethod
    def _get_elastic_affine_matrix_(shape_size, alpha_affine):
        """Elastic deformation of images as described in [Simard2003]_ (with modifications).
         .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
              Convolutional Neural Networks applied to Visual Document Analysis", in
              Proc. of the International Conference on Document Analysis and
              Recognition, 2003.
          Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
         """
        random_state = np.random.RandomState(None)

        # Random affine
        center_square = np.float32(shape_size) // 2
        square_size = min(shape_size) // 3
        pts1 = np.float32(
            [center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
             center_square - square_size])
        pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
        affine_matrix = cv2.getAffineTransform(pts1, pts2)

        return affine_matrix, random_state

    @staticmethod
    def _get_transformed_image_(image, affine_matrix, indices, seg=False):
        shape = image.shape
        trans_img = cv2.warpAffine(image, affine_matrix, shape[::-1], borderMode=cv2.BORDER_REFLECT_101)
        trans_coord = map_coordinates(trans_img, indices, order=1, mode='reflect').reshape(shape)
        return trans_coord

    @staticmethod
    def _get_indices4elastic_transform(shape, alpha, sigma, random_state):
        dxr = random_state.rand(*shape)
        dx = gaussian_filter((dxr * 2 - 1), sigma) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha

        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

        return indices
    
    def shift_img(self, image, segment, width_shift_range, height_shift_range, img_shape):
        """This fn will perform the horizontal or vertical shift"""
        if width_shift_range or height_shift_range:
            if width_shift_range:
                width_shift_range = random.uniform(-width_shift_range * img_shape[1], width_shift_range * img_shape[1])
            if height_shift_range:
                height_shift_range = random.uniform(-height_shift_range * img_shape[1], height_shift_range * img_shape[1])
            img = scipy.ndimage.shift(image, [width_shift_range, height_shift_range])
            seg = scipy.ndimage.shift(segment, [width_shift_range, height_shift_range])
            return img, seg

    @staticmethod
    def _adjust_brightness_(image, delta):
        out_img = image + delta
        return out_img

    @staticmethod
    def _adjust_contrast_(image, factor):
        img_mean = image.mean()
        out_img = (image - img_mean) * factor + img_mean
        return out_img

    def _load_and_enqueue(self, q, q_stat):
        try:
            while not self.coord.should_stop():
                seq_data, sequence_folder = self._read_sequence_data()
                if len(seq_data['metadata']['shape'])==3:
                    img_size = self.reshape_size + (seq_data['metadata']['shape'][-1],)
                else:
                    img_size = self.reshape_size
#                random_sub_sample = np.random.randint(1, 4) if self.randomize else 0
#                random_reverse = np.random.randint(0, 2) if self.randomize else 0
                if img_size[0] - self.sub_seq_size[0] > 0:
                    crop_y = np.random.randint(0, img_size[0] - self.sub_seq_size[0]) if self.randomize else 0
                else:
                    crop_y = 0
                if img_size[1] - self.sub_seq_size[1] > 0:
                    crop_x = np.random.randint(0, img_size[1] - self.sub_seq_size[1]) if self.randomize else 0
                else:
                    crop_x = 0

                flip = np.random.randint(0, 2, 2) if self.randomize else [0, 0]
                rotate = np.random.randint(0, 4) if self.randomize else 0
                if self.elastic_augmentation:
                    affine_matrix, random_state = self._get_elastic_affine_matrix_(self.sub_seq_size,
                                                                                   self.sub_seq_size[1] * 0.08)
                    indices = self._get_indices4elastic_transform(self.sub_seq_size, self.sub_seq_size[1] * 2,
                                                                  self.sub_seq_size[1] * 0.15,
                                                                  random_state)
                else:
                    affine_matrix = indices = None

                filename_idx = list(range(len(seq_data['file_list'])))
#
#                if random_reverse:
#                    filename_idx.reverse()
#                if random_sub_sample:
#                    filename_idx = filename_idx[::random_sub_sample]
#                remainder = seq_len % unroll_len
#
#                if remainder:
#                    if self.deal_with_end == 0:
#                        filename_idx = filename_idx[:-remainder]
#                    elif self.deal_with_end == 1:
#                        filename_idx += filename_idx[-2:-unroll_len + remainder - 2:-1]
#                    elif self.deal_with_end == 2:
#                        filename_idx += filename_idx[-1:] * (unroll_len - remainder)
                crop_y_stop = crop_y + self.sub_seq_size[0]
                crop_x_stop = crop_x + self.sub_seq_size[1]
                img_crops = seq_data['images'][:, crop_y:crop_y_stop, crop_x:crop_x_stop]
                img_max = seq_data['images'].max()

                seg_crops = seq_data['segs'][:, crop_y:crop_y_stop, crop_x:crop_x_stop]
                processed_image = []
                processed_seg = []
                is_last = []
                all_fnames = []
                for t, file_idx in enumerate(filename_idx):
                    filename = seq_data['file_list'][file_idx][0]
                    img_crop = img_crops[file_idx].copy()
                    seg_crop = seg_crops[file_idx].copy()
                    if self.randomize:
                        # contrast factor between [0.5, 1.5]
                        random_constrast_factor = np.random.rand() + 0.5
                        # random brightness delta plus/minus 10% of maximum value
                        random_brightness_delta = (np.random.rand() - 0.5) * 0.2 * img_max
                        img_crop = self._adjust_contrast_(img_crop, random_constrast_factor)
                        img_crop = self._adjust_brightness_(img_crop, random_brightness_delta)
#                        img_crop = cv2.normalize(img_crop, None, 0.0, 1.0, cv2.NORM_MINMAX)

                    if self.elastic_augmentation:
                        for i in range(img_crop.shape[2]):
                            trans_img = self._get_transformed_image_(img_crop[:,:, i], affine_matrix, indices)
                            img_crop[:, :, i] = trans_img
                        if np.any(np.isnan(img_crop)):
                            raise ValueError('NaN in image {} from sequence: {}'.format(file_idx, sequence_folder))
                        if np.any(np.isinf(img_crop)):
                            raise ValueError('Inf in image {} from sequence: {}'.format(file_idx, sequence_folder))
                        if not np.equal(seg_crop, -1).all():
#                            seg_not_valid = np.equal(seg_crop, -1)
#                            labeled_gt = seg_crop
#                            labeled_gt[:, 0] = 0
#                            labeled_gt[:, -1] = 0
#                            labeled_gt[-1, :] = 0
#                            labeled_gt[0, :] = 0imard20
                            seg_crop = self._get_transformed_image_(seg_crop.astype(np.float32), affine_matrix,
                                                                     indices, seg=True)
#                            trans_not_valid = self._get_transformed_image_(seg_not_valid.astype(np.float32),
#                                                                           affine_matrix,
#                                                                           indices, seg=True)
#                            trans_seg_fix = self._fix_transformed_segmentation(trans_seg)
#                            trans_not_valid = np.logical_or(np.greater(trans_not_valid, 0.5), np.equal(trans_seg, -1))
#                            seg_crop = trans_seg_fix
#                            seg_crop[trans_not_valid] = 0
                            if np.any(np.isnan(seg_crop)):
                                raise ValueError('NaN in Seg {} from sequence: {}'.format(file_idx, sequence_folder))
                            if np.any(np.isinf(seg_crop)):
                                raise ValueError('Inf in Seg {} from sequence: {}'.format(file_idx, sequence_folder))
                    if flip[0]:
                        img_crop = cv2.flip(img_crop, 0)
                        seg_crop = cv2.flip(seg_crop, 0)
                    if flip[1]:
                        img_crop = cv2.flip(img_crop, 1)
                        seg_crop = cv2.flip(seg_crop, 1)
                    if rotate > 0:
                        img_crop = np.rot90(img_crop, rotate)
                        seg_crop = np.rot90(seg_crop, rotate)
                    
                    img_crop, seg_crop = self.shift_img(img_crop, seg_crop, 0.1, 0.1, img_crop.shape)
                    is_last_frame = 1. if (t + 1) < len(filename_idx) else 0.
                    if self.num_threads == 1:
                        try:

                            while q_stat().numpy() > 0.9:
                                if self.coord.should_stop():
                                    return
                                time.sleep(1)
                            q.enqueue([img_crop, seg_crop, is_last_frame, filename])

                        except tf.errors.CancelledError:
                            pass
                    else:
                        is_last.append(is_last_frame)
                        processed_image.append(img_crop)
                        processed_seg.append(seg_crop)
                        all_fnames.append(filename)
                        if self.coord.should_stop():
                            return
                if self.num_threads == 1:
                    continue

                try:

                    while q_stat().numpy() > 0.9:
                        if self.coord.should_stop():
                            return
                        time.sleep(1)
                    q.enqueue_many([processed_image, processed_seg, is_last, all_fnames])

                except tf.errors.CancelledError:
                    pass

        except tf.errors.CancelledError:
            pass

        except Exception as err:
            print('ERROR FROM DATA PROCESS')
            self.coord.request_stop()
            raise err

    def _create_queues(self):
        def normed_size(_q):
            @tf.function
            def q_stat():
                return tf.cast(_q.size(), tf.float32) / self.queue_capacity

            return q_stat

        with tf.name_scope('DataHandler'):
            dtypes = [tf.float32, tf.float32,  tf.float32, tf.string]
            shapes = [self.sub_seq_size, self.sub_seq_size, (), ()] ####
            q_list = []

            q_stat_list = []
            for b in range(self.batch_size):
                q = tf.queue.FIFOQueue(self.queue_capacity, dtypes=dtypes, shapes=shapes, name='data_q_{}'.format(b))
                q_list.append(q)
                q_stat_list.append(normed_size(q))

        return q_list, q_stat_list

    def _batch_queues_(self):
        with tf.name_scope('DataHandler'):
            img_list = []
            seg_list = []
            is_last_list = []
            # fname_list = []
            for q in self.q_list:
                img, seg, is_last, fnames = q.dequeue_many(self.unroll_len)
                img_list.append(img)
                seg_list.append(seg)
                is_last_list.append(is_last[-1])


            image_batch = tf.stack(img_list, axis=0)
            seg_batch = tf.stack(seg_list, axis=0)
            is_last_batch = tf.stack(is_last_list, axis=0)


            if self.data_format == 'NHWC':
                image_batch = tf.expand_dims(image_batch, 4)
                seg_batch = tf.expand_dims(seg_batch, 4)
            elif self.data_format == 'NCHW':
                image_batch = tf.expand_dims(image_batch, 2)
                seg_batch = tf.expand_dims(seg_batch, 2)
            else:
                raise ValueError()

        return image_batch, seg_batch, is_last_batch

    def _create_sequence_queue(self):
        sequence_queue = queue.Queue(maxsize=len(self.sequence_folder_list))
        for sequence in self.sequence_folder_list:
            sequence_queue.put(sequence)
        return sequence_queue

    def start_queues(self, coord=tf.train.Coordinator(), debug=False):
        self._read_sequence_to_ram_()
        threads = []
        self.coord = coord
        for q, q_stat in zip(self.q_list, self.q_stat_list):
            if debug:
                self._load_and_enqueue(q, q_stat)
            for _ in range(self.num_threads):
                t = threading.Thread(target=self._load_and_enqueue, args=(q, q_stat))
                t.daemon = True
                t.start()
                threads.append(t)
                self.coord.register_thread(t)

        t = threading.Thread(target=self._monitor_queues_)
        t.daemon = True
        t.start()
        self.coord.register_thread(t)
        threads.append(t)
        return threads

    def _monitor_queues_(self):
        while not self.coord.should_stop():
            time.sleep(1)
        for q in self.q_list:
            q.close(cancel_pending_enqueues=True)

    def get_batch(self):
        return self._batch_queues_()


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
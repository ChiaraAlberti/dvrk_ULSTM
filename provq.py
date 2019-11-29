import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('/home/stormlab/seg/lstm_dataset/train/disp_lobe1_01013.jpeg', -1).astype(np.float32)
plt.imshow(img/255)
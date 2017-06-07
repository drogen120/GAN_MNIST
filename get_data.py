import tensorflow as tf
import os
import numpy as np
from PIL import Image
import cv2
from tensorflow.examples.tutorials.mnist import input_data
from utils import *

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# print(mnist.train.images.shape )
# print(mnist.train.labels.shape)
# print(mnist.test.images.shape )
# print(mnist.test.labels.shape )
# print(mnist.test.labels[0])
# print(mnist.test.images[0].shape)

# print (np.where(mnist.test.labels[0]>0)[0][0])

for i in range(10):
    if not os.path.exists('./train_data'):
        os.mkdir('./train_data')
    if not os.path.exists('./test_data'):
        os.mkdir('./test_data')
    if not os.path.exists('./train_data/%d'%i):
        os.mkdir('./train_data/%d'%i)
    if not os.path.exists('./test_data/%d'%i):
        os.mkdir('./test_data/%d'%i)

for i in range(55000):
    tmp = np.reshape(mnist.train.images[i],[28,28])

    tmp[np.where(tmp >0)] = tmp[np.where(tmp>0)] * 255.0

    tmp = np.expand_dims(tmp,-1)

    # img_train = Image.fromarray(tmp.astype(np.uint8))
    # img_train.save("./train_data/%d/%05d.jpg"%(np.where(mnist.train.labels[i]>0)[0][0],i))
    cv2.imwrite("./train_data/%d/%05d.jpg"%(np.where(mnist.train.labels[i]>0)[0][0],i),tmp)
    if i <10000:
        tmp = np.reshape(mnist.test.images[i],[28,28])

        tmp[np.where(tmp >0)] = tmp[np.where(tmp>0)] * 255.0

        tmp = np.expand_dims(tmp,-1)

        # img_test = Image.fromarray(tmp.astype(np.uint8))
        # img_test.save("./test_data/%d/%05d.jpg"%(np.where(mnist.test.labels[i]>0)[0][0],i))
        cv2.imwrite("./test_data/%d/%05d.jpg"%(np.where(mnist.test.labels[i]>0)[0][0],i),tmp)

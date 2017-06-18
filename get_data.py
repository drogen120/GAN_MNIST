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
# print(mnist.train.images[0])
# print (mnist.train.labels[0])
# raise
# print(mnist.test.images.shape )
# print(mnist.test.labels.shape )
# print(mnist.test.labels[0])
# print(mnist.test.images[0].shape)

# print (np.where(mnist.test.labels[0]>0)[0][0])

# test for tfrecord_writer
# for j in range(2):
#     record_filename = "%s/gen_%d.tfrecord"%('./',j)
#     with tf.python_io.TFRecordWriter(record_filename) as tfrecord_writer:
#         for k in range(100):
#             samples = np.reshape(mnist.train.images[k],[28,28])*255.0
#             img_raw = samples.astype(np.uint8).tostring()
#             label_raw = int(j)
#             example = to_tfexample_raw(img_raw,label_raw)
#             tfrecord_writer.write(example.SerializeToString())
#         tfrecord_writer.close()
# raise

# write traininng data to tfrecord
# record_filename = './train.tfrecord'
# with tf.python_io.TFRecordWriter(record_filename) as tfrecord_writer:
#     for i in range(55000):
#         samples = np.reshape(mnist.train.images[i],[28,28])*255.0
#         img_raw = samples.astype(np.uint8).tostring()
#         label_raw = mnist.train.labels[i].astype(np.uint8).tostring()
#         example = to_tfexample_raw(img_raw,label_raw)
#         tfrecord_writer.write(example.SerializeToString())
#         if i <5000:
#             samples = np.reshape(mnist.validation.images[i],[28,28])*255.0
#             img_raw = samples.astype(np.uint8).tostring()
#             label_raw = mnist.validation.labels[i].astype(np.uint8).tostring()
#             example = to_tfexample_raw(img_raw,label_raw)
#             tfrecord_writer.write(example.SerializeToString())
#
#     tfrecord_writer.close()
# raise


#test for read tf_record
# record_iterator = tf.python_io.tf_record_iterator(path = './data_tf_test/gen.tfrecord')
# count = 0
# for string_record in record_iterator:
#
#     example = tf.train.Example()
#     example.ParseFromString(string_record)
#
#     label = (example.features.feature['label'].bytes_list.value[0])
#     label_1d = np.fromstring(label, dtype = np.uint8)
#     label_f = label_1d.reshape((10,))
#     print (label_f)
#
#     img = (example.features.feature['image'].bytes_list.value[0])
#     img_1d = np.fromstring(img,dtype = np.uint8)
#     #
#     img_f = img_1d.reshape((28,28,-1))
#     #
#     #
#     # print (img_f)
#     # raise
#     cv2.imshow('img-f',img_f)
#     cv2.waitKey(0)
# raise



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
    if i <5000:
        tmp = np.reshape(mnist.validation.images[i],[28,28])

        tmp[np.where(tmp >0)] = tmp[np.where(tmp>0)] * 255.0

        tmp = np.expand_dims(tmp,-1)

        # img_test = Image.fromarray(tmp.astype(np.uint8))
        # img_test.save("./test_data/%d/%05d.jpg"%(np.where(mnist.test.labels[i]>0)[0][0],i))
        cv2.imwrite("./train_data/%d/val_%05d.jpg"%(np.where(mnist.validation.labels[i]>0)[0][0],i),tmp)

    if i <10000:
        tmp = np.reshape(mnist.test.images[i],[28,28])

        tmp[np.where(tmp >0)] = tmp[np.where(tmp>0)] * 255.0

        tmp = np.expand_dims(tmp,-1)

        # img_test = Image.fromarray(tmp.astype(np.uint8))
        # img_test.save("./test_data/%d/%05d.jpg"%(np.where(mnist.test.labels[i]>0)[0][0],i))
        cv2.imwrite("./test_data/%d/%05d.jpg"%(np.where(mnist.test.labels[i]>0)[0][0],i),tmp)

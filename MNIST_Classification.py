import tensorflow as tf
import numpy as np
slim = tf.contrib.slim
from glob import glob
import os


class Classification_Model(object):

    def __init__(self):

        self.img_size = 28
        self.feature_size = 100
        self.end_points = {}
        self.input_fname_pattern = '*.jpg'
        self.batch_size = 128

    def transform(self, img):
        return img/127.5 - 1

    def model_arg_scope(self, weight_decay=0.0005, is_training = True):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn = tf.nn.relu,
                            normalizer_fn = slim.batch_norm,
                            normalizer_params={'is_training': is_training, 'decay': 0.9},
                            weights_regularizer=slim.l2_regularizer(weight_decay),
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.01)
                            # biases_initializer=tf.zeros_initializer()
                            ):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d], padding = 'SAME') as sc:
                return sc

    def net(self, inputs, is_training=True, dropout_keep_prob=0.5, reuse=False, scope='AlexNet'):
        with tf.variable_scope(scope, 'AlexNet', [inputs], reuse=reuse) as net_scope:
            with slim.arg_scope(self.model_arg_scope(is_training = is_training)):
                if reuse:
                    net_scope.reuse_variables()
                net = slim.conv2d(inputs, 16, [3, 3], scope='conv1')
                net = slim.max_pool2d(net, [2, 2], scope='pool1')
                net = slim.conv2d(net, 32, [3, 3], scope='conv2')
                net = slim.conv2d(net, 64, [3, 3], scope='conv3')
                net = slim.max_pool2d(net, [2, 2], scope='pool3')
                net = slim.conv2d(net, 128, [3, 3], scope='conv4')
                net = slim.conv2d(net, 256, [3, 3], scope='conv5')
                net = slim.conv2d(net, 512, [3, 3], scope='conv6')
                net = slim.avg_pool2d(net, [7, 7], stride = 1, scope='average_pool')
                net = slim.flatten(net)
                features = slim.fully_connected(net, self.feature_size, scope='features')
                self.end_points['Features'] = features
                logits = slim.fully_connected(features, 10, activation_fn = None, scope='logits')
                self.end_points['Logits'] = logits
                predictions = tf.nn.softmax(logits, name='Predictions')
                self.end_points['Predictions'] = predictions
        return logits, self.end_points
    #
    # def net_test(self, inputs, is_training=False, dropout_keep_prob=0.5, reuse=True, scope='AlexNet'):
    #     with tf.variable_scope(scope, 'AlexNet', [inputs], reuse=reuse) as net_scope:
    #         with slim.arg_scope(self.model_arg_scope(is_training = False )):
    #             if reuse:
    #                 net_scope.reuse_variables()
    #             net = slim.conv2d(inputs, 16, [3, 3], scope='conv1')
    #             net = slim.max_pool2d(net, [2, 2], scope='pool1')
    #             net = slim.conv2d(net, 32, [3, 3], scope='conv2')
    #             net = slim.conv2d(net, 64, [3, 3], scope='conv3')
    #             net = slim.max_pool2d(net, [2, 2], scope='pool3')
    #             net = slim.conv2d(net, 128, [3, 3], scope='conv4')
    #             net = slim.conv2d(net, 256, [3, 3], scope='conv5')
    #             net = slim.conv2d(net, 512, [3, 3], scope='conv6')
    #             net = slim.avg_pool2d(net, [7, 7], stride = 1, scope='average_pool')
    #             net = slim.flatten(net)
    #             features = slim.fully_connected(net, self.feature_size, scope='features')
    #             self.end_points['Features'] = features
    #             logits = slim.fully_connected(features, 10, activation_fn = None, scope='logits')
    #             self.end_points['Logits'] = logits
    #             predictions = tf.nn.softmax(logits, name='Predictions')
    #             self.end_points['Predictions'] = predictions
    #     return logits, self.end_points


    def losses(self, logits, labels, scope='model_losses'):

        with tf.name_scope(scope, 'model_losses'):
            losses = tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=logits)
            losses = tf.reduce_mean(losses)

        return losses

    def optimizer(self, learning_rate, scope = 'model_optimizer'):

        with tf.name_scope(scope, 'model_optimizer'):
            optimizer = tf.train.AdadeltaOptimizer(learning_rate)

        return optimizer

    def change_dataset(self, dataset_name = "0"):
        self.dataset_name = dataset_name
        self.file_list = glob(os.path.join("./samples", self.dataset_name, self.input_fname_pattern))
        self.filename_queue = tf.train.string_input_producer(self.file_list[:])
        self.image_reader = tf.WholeFileReader()
        _, self.image_file = self.image_reader.read(self.filename_queue)

        self.image = tf.image.decode_png(self.image_file)

        self.image = tf.cast(tf.reshape(self.image, shape=[28, 28, 1]), dtype=tf.float32)

    def get_batch(self):

        num_preprocess_threads = 1
        min_queue_examples = 256

        batch_images = tf.train.batch([self.image],
                                       batch_size=self.batch_size,
                                       num_threads=num_preprocess_threads,
                                       capacity=min_queue_examples + 3 * self.batch_size)
        batch_images = self.transform(batch_images)
        batch_labels = tf.one_hot(tf.ones([self.batch_size], dtype=tf.int32) * int(self.dataset_name), depth=10)
        return batch_images, batch_labels

    def read_tf(self,filename_queue):

        reader = tf.TFRecordReader()
        _,serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(
            serialized_example,
            features ={
                'image': tf.FixedLenFeature([],tf.string),
                'label': tf.FixedLenFeature([],tf.string),
            }
        )
        label = tf.decode_raw(features['label'],tf.uint8)
        image = tf.decode_raw(features['image'],tf.uint8)

        image = tf.reshape(image,(28,28,1))
        label = tf.reshape(label,(10,))

        num_preprocess_threads = 10
        min_queue_examples = 256

        image,label = tf.train.shuffle_batch([image,label],batch_size = self.batch_size,
            num_threads = num_preprocess_threads,capacity = min_queue_examples + 3*self.batch_size,
                min_after_dequeue = min_queue_examples)

        return image,label

    def get_batch_tf(self,tfrecords_path):
        tfrecords_filename = glob(tfrecords_path + '*.tfrecord')
        # print (tfrecords_filename)
        # print ('**************88')
        # raise
        filename_queue = tf.train.string_input_producer(tfrecords_filename[:])
        image,label = self.read_tf(filename_queue)
        image = tf.cast(image,tf.float32)
        image = self.transform(image)
        label = tf.cast(label,tf.float32)
        return image,label

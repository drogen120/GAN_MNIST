import tensorflow as tf
from collections import namedtuple
from math import sqrt
import numpy as np
slim = tf.contrib.slim
from glob import glob
import os
from tensorflow.examples.tutorials.mnist import input_data
import time

class Resnet_model(object):

    def __init__(self):
        self.LayerBlock = namedtuple('LayerBlock',['num_repeats','num_filters','bottleneck_size'])
        self.blocks = [self.LayerBlock(3, 128, 32),
              self.LayerBlock(3, 256, 64),
              self.LayerBlock(3, 512, 128),
              self.LayerBlock(3, 1024, 256)]
        self.batch_size = 64
        self.img_size = 28
        self.feature_size = 100
        self.end_points = {}
        self.input_fname_pattern = '*.jpg'

    def transform(self, img):
        return img/127.5 - 1

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

        num_preprocess_threads = 1
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

    def net_infer(self,x,is_training=True,reuse = False ):
        input_shape = x.get_shape().as_list()
        if len(input_shape) == 2:
            ndim = int(sqrt(input_shape[1]))
            if ndim * ndim != input_shape[1]:
                raise ValueError('input_shape should be square')
            x = tf.reshape(x, [-1, ndim, ndim, 1])

        with tf.variable_scope("resnet",reuse = reuse ) as scope:
            with slim.arg_scope(self.model_arg_scope(is_training = is_training)):
                if reuse:
                    scope.reuse_variables()

                # net = conv2d(x, 64, k_h=7, k_w=7,
                #      name='conv1',
                #      activation=tf.nn.relu)

                net = slim.conv2d(x, 64, [3, 3], scope='conv1')

                # Max pool and downsampling
                # net = tf.nn.max_pool(
                #     net, [1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

                net = slim.max_pool2d(net, [3, 3],padding = 'SAME')
                # %%
                # Setup first chain of resnets
                # net = conv2d(net, self.blocks[0].num_filters, k_h=1, k_w=1,
                #              stride_h=1, stride_w=1, padding='VALID', name='conv2')

                net = slim.conv2d(net,self.blocks[0].num_filters,[1,1],padding = 'VALID',scope = 'conv2')
                # %%
                # Loop through all res blocks
                for block_i, block in enumerate(self.blocks):
                    for repeat_i in range(block.num_repeats):

                        name = 'block_%d/repeat_%d' % (block_i, repeat_i)

                        # conv = conv2d(net, block.bottleneck_size, k_h=1, k_w=1,
                        #               padding='VALID', stride_h=1, stride_w=1,
                        #               activation=tf.nn.relu,
                        #               name=name + '/conv_in')
                        conv = slim.conv2d(net,block.bottleneck_size,[1,1],padding='VALID',scope = name +'/conv_in')

                        # conv = conv2d(conv, block.bottleneck_size, k_h=3, k_w=3,
                        #               activation=tf.nn.relu,
                        #               padding='SAME', stride_h=1, stride_w=1,
                        #               name=name + '/conv_bottleneck')
                        conv = slim.conv2d(conv,block.bottleneck_size,[3,3],scope= name+'/conv_bottleneck')

                        # conv = conv2d(conv, block.num_filters, k_h=1, k_w=1,
                        #               padding='VALID', stride_h=1, stride_w=1,
                        #               activation=tf.nn.relu,
                        #               name=name + '/conv_out')
                        conv = slim.conv2d(conv,block.num_filters,[1,1],padding='VALID',scope=name + '/conv_out')

                        net = conv + net
                    try:
                        # upscale to the next block size
                        next_block = self.blocks[block_i + 1]

                        # net = conv2d(net, next_block.num_filters, k_h=1, k_w=1,
                        #              padding='SAME', stride_h=1, stride_w=1, bias=False,
                        #              name='block_%d/conv_upscale' % block_i)
                        net = slim.conv2d(net,next_block.num_filters,[1,1],
                                    normalizer_fn = None,scope = 'block_%d/conv_upscale' % block_i)
                    except IndexError:
                        pass

                # %%
                # net = tf.nn.avg_pool(net,
                #                      ksize=[1, net.get_shape().as_list()[1],
                #                             net.get_shape().as_list()[2], 1],
                #                      strides=[1, 1, 1, 1], padding='VALID')
                net = slim.avg_pool2d(net, [net.get_shape().as_list()[1],net.get_shape().as_list()[2]],
                                        padding = 'VALID',stride = 1, scope='average_pool')

                # net = tf.reshape(
                #     net,
                #     [-1, net.get_shape().as_list()[1] *
                #      net.get_shape().as_list()[2] *
                #      net.get_shape().as_list()[3]])

                # logits = linear(net, 10)
                # add linear to feature vector size
                net = slim.flatten(net)
                features = features = slim.fully_connected(net, self.feature_size, scope='features')
                self.end_points['Features'] = features
                logits = slim.fully_connected(features, 10, activation_fn = None, scope='logits')
                self.end_points['Logits'] = logits
                predictions = tf.nn.softmax(logits, name='Predictions')
                self.end_points['Predictions'] = predictions

        return logits, self.end_points

    def losses(self, logits, labels, scope='model_losses'):

        with tf.name_scope(scope, 'model_losses'):
            losses = tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=logits)
            losses = tf.reduce_mean(losses)

        return losses

    def optimizer(self, learning_rate, scope = 'model_optimizer'):

        with tf.name_scope(scope, 'model_optimizer'):
            # optimizer = tf.train.AdadeltaOptimizer(learning_rate)
            # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            optimizer = tf.train.AdamOptimizer(learning_rate)

        return optimizer

def start_R(iteration,start = True):

    tf.logging.set_verbosity(tf.logging.DEBUG)
    tfrecords_path = './data_tf/'

    with tf.Graph().as_default():

        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
        if not os.path.exists('./checkpoint_pretrain_res/'):
            os.mkdir('./checkpoint_pretrain_res/')
        ckpt = tf.train.get_checkpoint_state(os.path.dirname('./checkpoint_pretrain_res/checkpoint'))

        sess = tf.InteractiveSession()

        global_step = slim.create_global_step()
        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

        mnist_net = Resnet_model()
        # mnist_net.change_dataset("5")
        # x, y_ = mnist_net.get_batch()
        x,y_ = mnist_net.get_batch_tf(tfrecords_path)

        end_points = {}

        logits, end_points = mnist_net.net_infer(x)
        # losses = mnist_net.losses(logits, y_)
        # train_step = mnist_net.optimizer(0.001).minimize(losses, global_step=global_step)

        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(extra_update_ops):
            losses = mnist_net.losses(logits, y_)
            train_step = mnist_net.optimizer(0.001).minimize(losses, global_step=global_step)

        #total_loss = tf.losses.get_total_loss()
        summaries.add(tf.summary.image("img", tf.cast(x, tf.float32)))
        summaries.add(tf.summary.scalar('loss', losses))

        for variable in tf.trainable_variables():
            summaries.add(tf.summary.histogram(variable.op.name, variable))
        train_writer = tf.summary.FileWriter('./train_res',
                                             sess.graph)

        correct_prediction = tf.equal(tf.argmax(end_points['Predictions'], 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.float32)) / mnist_net.batch_size
        summaries.add(tf.summary.scalar('accuracy', accuracy))

        summary_op = tf.summary.merge(list(summaries), name='summary_op')
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        saver = tf.train.Saver(max_to_keep=5,
                               keep_checkpoint_every_n_hours=1.0,
                               write_version=2,
                               pad_step_number=False)

        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        save_test = []
        for i in range(iteration):

            if i %100 == 0 :
                x_test = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
                y_test = tf.placeholder(tf.float32, shape=[None, 10])

                logits_test, end_points_test = mnist_net.net_infer(x_test,reuse = True)

                correct_prediction_test = tf.equal(tf.argmax(end_points_test['Predictions'], 1), tf.argmax(y_test, 1))
                # accuracy_test = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                num_correct = tf.reduce_sum(tf.cast(correct_prediction_test,tf.float32))
                sum_accuracy_test = 0
                batch_size = 100
                for count in range(100):
                    test_image = np.reshape(mnist.test.images[count*batch_size:(count+1)*batch_size],(batch_size,28,28,1)) *2.0 -1
                    test_label = np.reshape(mnist.test.labels[count*batch_size:(count+1)*batch_size],(batch_size,10))
                    # test_batch = mnist.test.next_batch(100)
                    # accuracy_test_str = sess.run(accuracy_test,
                    #     feed_dict={x_test: np.reshape(test_batch[0], (-1, 28, 28, 1)), y_test: test_batch[1]})
                    num_c = sess.run(num_correct,
                        feed_dict = {x_test:test_image, y_test:test_label})
                    sum_accuracy_test += num_c
                    # sum_accuracy_test += accuracy_test_str
                # print ("test accuracy is: %f" % (sum_accuracy_test /100.0 ))
                print('****************************')
                print ("test accuracy is: %f" % (sum_accuracy_test /10000.0 ))
                print('****************************')
                if start:
                    if not save_test:
                        save_test.append(sum_accuracy_test)
                    else :
                        save_test.append(sum_accuracy_test)
                        if sum_accuracy_test >= save_test[0] + 2 and sum_accuracy_test < 10000:
                            print ('u are getting better!!!!')
                            break
                        else:
                            print('ops, not this time ~!')
                else:
                    if sum_accuracy_test/10000.0 >= 0.9950:
                        break
            _,summary_str = sess.run([train_step,summary_op])
            if i %10 == 0:
                train_writer.add_summary(summary_str,i)
                print('%diteration'%i,sess.run(accuracy))
        coord.request_stop()
        coord.join(threads)
        print ('saving model')
        saver.save(sess, "./checkpoint_pretrain_res/",global_step= global_step.eval())
        time.sleep(3)

def main(_):
    start_R(1000000,start= False)

if __name__ == '__main__':
    tf.app.run()

import GAN_10
import tensorflow as tf
import numpy as np
import os
import cv2
from ops import *
from utils import *
import time
from tensorflow.examples.tutorials.mnist import input_data
from MNIST_Classification import Classification_Model
from glob import glob
slim = tf.contrib.slim


flags = tf.app.flags
flags.DEFINE_integer("iter", 9002, "iter to train ")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("sample_num", 64, "The size of sample images ")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("input_height", 28, "The size of image to use (will be center cropped). [108]")
flags.DEFINE_integer("input_width", 28, "The size of image to use (will be center cropped). If None, same value as input_height [None]")
flags.DEFINE_integer("output_height", 28, "The size of the output images to produce [64]")
flags.DEFINE_integer("output_width", 28, "The size of the output images to produce. If None, same value as output_height [None]")
flags.DEFINE_string("dataset", "5", "The name of dataset [...]")
flags.DEFINE_string("input_fname_pattern", "*.jpg", "Glob pattern of filename of input images [*]")
flags.DEFINE_string("checkpoint_dir", "./checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "./samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("train", True, "True for training, False for testing [False]")
flags.DEFINE_boolean("crop", True, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
flags.DEFINE_integer("C_iter", 100, "The iteration of training C")
flags.DEFINE_integer("C_batch_size", 64, "The batch_size of extracting feature vector of C")
FLAGS = flags.FLAGS


def start_GAN():
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

    with tf.Graph().as_default():
        run_config = tf.ConfigProto()
        run_config.gpu_options.allow_growth=True

        with tf.Session(config = run_config) as sess:
            dcgan = GAN_10.DCGAN(
                    sess,
                    input_width=FLAGS.input_width,
                    input_height=FLAGS.input_height,
                    output_width=FLAGS.output_width,
                    output_height=FLAGS.output_height,
                    batch_size=FLAGS.batch_size,
                    sample_num=FLAGS.sample_num,
                    # dataset_name=FLAGS.dataset,
                    input_fname_pattern=FLAGS.input_fname_pattern,
                    crop=FLAGS.crop,
                    checkpoint_dir=FLAGS.checkpoint_dir,
                    sample_dir=FLAGS.sample_dir)
            dcgan.train(FLAGS)


def start_C(iteration):

    tf.logging.set_verbosity(tf.logging.DEBUG)
    tfrecords_path = './data_tf/'

    with tf.Graph().as_default():

        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

        ckpt = tf.train.get_checkpoint_state(os.path.dirname('./checkpoint_pretrain/checkpoint'))

        sess = tf.InteractiveSession()

        global_step = slim.create_global_step()
        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

        mnist_net = Classification_Model()
        # mnist_net.change_dataset("5")
        # x, y_ = mnist_net.get_batch()
        x,y_ = mnist_net.get_batch_tf(tfrecords_path)

        # arg_scope = mnist_net.model_arg_scope()
        end_points = {}

        # with slim.arg_scope(arg_scope):
        logits, end_points = mnist_net.net(x)
        losses = mnist_net.losses(logits, y_)
        train_step = mnist_net.optimizer(0.001).minimize(losses, global_step=global_step)

        #total_loss = tf.losses.get_total_loss()
        summaries.add(tf.summary.image("img", tf.cast(x, tf.float32)))
        summaries.add(tf.summary.scalar('loss', losses))

        for variable in tf.trainable_variables():
            summaries.add(tf.summary.histogram(variable.op.name, variable))
        train_writer = tf.summary.FileWriter('./train',
                                             sess.graph)

        correct_prediction = tf.equal(tf.argmax(end_points['Predictions'], 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
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

        for i in range(iteration):
            print('%diteration'%i,sess.run(accuracy))
            _,summary_str = sess.run([train_step,summary_op])
            if i %10 == 0:
                train_writer.add_summary(summary_str,i)
            if i %100 == 0:
                x_test = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
                y_test = tf.placeholder(tf.float32, shape=[None, 10])

                logits_test, end_points_test = mnist_net.net(x_test,is_training = False,reuse = True)

                correct_prediction_test = tf.equal(tf.argmax(end_points_test['Predictions'], 1), tf.argmax(y_test, 1))
                accuracy_test = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

                sum_accuracy_test = 0.0
                for count in range(100):
                    test_batch = mnist.test.next_batch(100)
                    accuracy_test_str = sess.run([accuracy_test],
                        feed_dict={x_test: np.reshape(test_batch[0], (-1, 28, 28, 1)), y_test: test_batch[1]})
                    sum_accuracy_test += accuracy_test_str[0]
                print ("test accuracy is: %f" % (sum_accuracy_test /100.0 ))

        coord.request_stop()
        coord.join(threads)

        time.sleep(3)


def get_feature(batch_size ,id ):

    with tf.Graph().as_default():

        ckpt = tf.train.get_checkpoint_state(os.path.dirname('./checkpoint_pretrain/checkpoint'))
        sess = tf.InteractiveSession()

        num_preprocess_threads = 1
        min_queue_examples = 256
        image_reader = tf.WholeFileReader()

        file_list = glob(os.path.join("./train_data",str(id),"*.jpg"))
        filename_queue = tf.train.string_input_producer(file_list[:])
        _,image_file = image_reader.read(filename_queue)
        image = tf.image.decode_jpeg(image_file)
        image = tf.cast(tf.reshape(image,shape = [28,28,1]), dtype = tf.float32)

        batch_images = tf.train.batch([image],batch_size = batch_size,
                                            num_threads = num_preprocess_threads,
                                            capacity = min_queue_examples + 3*batch_size)
        batch_images = batch_images/ 127.5 -1

        # summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
        # summaries.add(tf.summary.image("batch_img", tf.cast(batch_images, tf.float32)))
        #
        # train_writer = tf.summary.FileWriter('./get_feature/%d'%id, sess.graph)

        mnist_net = Classification_Model()

        logits, end_points = mnist_net.net(batch_images,is_training =False)

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        # summary_op = tf.summary.merge(list(summaries), name='summary_op')

        saver = tf.train.Saver(max_to_keep=5,
                               keep_checkpoint_every_n_hours=1.0,
                               write_version=2,
                               pad_step_number=False)

        if ckpt and ckpt.model_checkpoint_path:

            saver.restore(sess, ckpt.model_checkpoint_path)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord = coord )

        all_features = np.zeros((batch_size*100,100))

        for count in range(100):
            # summary_str = sess.run(summary_op)
            # train_writer.add_summary(summary_str,count)
            featurens_str = sess.run([end_points["Features"]])
            all_features[count*batch_size:(count+1)*batch_size,:] = featurens_str[0]

        np.save("./outputs/features_%d"%id,all_features)
        print ('******************************')
        print('succed save npz once with %d'%id)
        print ('******************************')
        coord.request_stop()
        coord.join(threads)

def main(_):

    while True:
        start_C(FLAGS.C_iter)
        for i in range(10):
            get_feature(FLAGS.C_batch_size,i)
        start_GAN()

if __name__ == '__main__':
    tf.app.run()

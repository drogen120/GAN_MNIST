import tensorflow as tf
import numpy as np
import os
import time
from tensorflow.contrib.tensorboard.plugins import projector

slim = tf.contrib.slim

from MNIST_Classification_with_embedding import Classification_Model

from tensorflow.examples.tutorials.mnist import input_data


def main(_):

    tf.logging.set_verbosity(tf.logging.DEBUG)
    tfrecords_path = './data_tf/'
    iteration = 50000

    with tf.Graph().as_default():

        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
        is_training = tf.placeholder(tf.bool)

        ckpt = tf.train.get_checkpoint_state(os.path.dirname('./checkpoint_pretrain/checkpoint'))

        sess = tf.InteractiveSession()

        global_step = slim.create_global_step()
        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

        mnist_net = Classification_Model()
        x,y_ = mnist_net.get_batch_tf(tfrecords_path)
        # x = tf.placeholder(tf.float32, [None, 28, 28, 1], name='x-input')
        # y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')

        # arg_scope = mnist_net.model_arg_scope()
        end_points = {}

        # with slim.arg_scope(arg_scope):
        logits, end_points = mnist_net.net(x, is_training = is_training, reuse = None)
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            losses = mnist_net.losses(logits, y_)
            train_step = mnist_net.optimizer(0.001).minimize(losses, global_step=global_step)

        embedding, config = mnist_net.get_embedding('./checkpoint_pretrain/')

        #total_loss = tf.losses.get_total_loss()
        summaries.add(tf.summary.image("img", tf.cast(x, tf.float32)))
        summaries.add(tf.summary.scalar('loss', losses))

        for variable in tf.trainable_variables():
            summaries.add(tf.summary.histogram(variable.op.name, variable))
        train_writer = tf.summary.FileWriter('./checkpoint_pretrain/',
                                             sess.graph)

        projector.visualize_embeddings(train_writer, config)

        correct_prediction = tf.equal(tf.argmax(end_points['Predictions'], 1), tf.argmax(y_, 1))
        train_num_correct = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
        accuracy = train_num_correct / mnist_net.batch_size
        #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
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

        x_test = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
        y_test = tf.placeholder(tf.float32, shape=[None, 10])
        test_is_training = tf.placeholder(tf.bool)
        logits_test, end_points_test = mnist_net.net(x_test, is_training = test_is_training, reuse = True)
        correct_prediction_test = tf.equal(tf.argmax(end_points_test['Predictions'], 1), tf.argmax(y_test, 1))
        #accuracy_test = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        num_correct = tf.reduce_sum(tf.cast(correct_prediction_test, tf.float32))

        assignment = embedding.assign(end_points_test['Features'])

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for i in range(iteration):

            batch = mnist.train.next_batch(mnist_net.batch_size)
            _,summary_str = sess.run([train_step,summary_op], feed_dict={is_training : True})
            #_,summary_str = sess.run([train_step,summary_op], feed_dict={x: np.reshape(batch[0], (-1, 28, 28, 1)), y_:batch[1], is_training : True})
            if i %100 == 0:
                global_step_str = global_step.eval()
                train_writer.add_summary(summary_str, global_step_str)
                #print('%diteration'%global_step_str,sess.run(accuracy, feed_dict={is_training : True}))
                print('####################################')
                variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

            if i % 1000 == 0:
                sum_accuracy_test = 0.0
                test_batch_x = mnist.test.images[:10000] * 2.0 - 1
                test_batch_y = mnist.test.labels[:10000]
                accuracy_test_str, _ = sess.run([num_correct, assignment],
                    feed_dict={x_test: np.reshape(test_batch_x, (-1, 28, 28, 1)), y_test: test_batch_y, test_is_training: False})
                sum_accuracy_test += accuracy_test_str
                print ("test accuracy is: %f" % (sum_accuracy_test /10000.0 ))
                saver.save(sess, "./checkpoint_pretrain/",global_step=global_step_str)

        coord.request_stop()
        coord.join(threads)

        time.sleep(3)

if __name__ == '__main__':
    tf.app.run()

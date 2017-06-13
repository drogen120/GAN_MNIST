import tensorflow as tf
import numpy as np
import os

slim = tf.contrib.slim

from MNIST_Classification import Classification_Model

from tensorflow.examples.tutorials.mnist import input_data


def main(_):

    tf.logging.set_verbosity(tf.logging.DEBUG)
    tfrecords_path = './data_tf/'

    with tf.Graph().as_default():

        #mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

        ckpt = tf.train.get_checkpoint_state(os.path.dirname('./checkpoint_pretrain/checkpoint'))

        sess = tf.InteractiveSession()

        global_step = slim.create_global_step()
        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

        mnist_net = Classification_Model()
        mnist_net.change_dataset("5")
        x, y_ = mnist_net.get_batch()
        # x,y_ = mnist_net.get_batch_tf(tfrecords_path)

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
        saver = tf.train.Saver(max_to_keep=5,
                               keep_checkpoint_every_n_hours=1.0,
                               write_version=2,
                               pad_step_number=False)

        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        train_num = 0
        all_features = np.zeros((5000, 100))

        for i in range(100):

            # _, summary_str = sess.run([train_step, summary_op])
            featurens_str = sess.run([mnist_net.end_points["Features"]])
            all_features[i*50:i*50+50,:] = featurens_str[0]

            global_step_str = global_step.eval()

            # if global_step_str % 10000 == 0:
            #     train_num = (train_num + 1) % 10
            #     mnist_net.change_dataset(str(train_num))
                #t_x, t_y = mnist_net.get_batch()
                #tf.assign(x, t_x)
                #tf.assign(y_, t_y)

            # if global_step_str % 100 == 0:
            #     train_writer.add_summary(summary_str, global_step_str)
            #     loss_str = sess.run([losses])
            #     print ("step %d, loss = %f" % (global_step_str, loss_str[0]))
            #
            # if global_step_str % 1000 == 0:
            #     sum_accuracy = 0.0
            #     for i in range(10):
            #         accuracy_str = sess.run([accuracy])
            #         sum_accuracy += accuracy_str[0]
            #
            #     print ("accuracy: %f" % (sum_accuracy / 10.0))

        coord.request_stop()
        coord.join(threads)

        np.save("./outputs/features_5", all_features)

            # if global_step_str % 1000 == 0:
            #     sum_accuracy = 0.0
            #     for i in range(600):
            #         test_batch = mnist.train.next_batch(100)
            #         accuracy_str = sess.run([accuracy], feed_dict={x: np.reshape(test_batch[0], (-1, 28, 28, 1)), y_: test_batch[1]})
            #         sum_accuracy += accuracy_str[0]
            #
            #     print ("accuracy: %f" % (sum_accuracy / 600.0))
            #     saver.save(sess, "./checkpoint_pretrain/",global_step=global_step_str)


        # sum_accuracy = 0.0
        # for i in range(100):
        #     test_batch = mnist.test.next_batch(100)
        #     accuracy_str = sess.run([accuracy], feed_dict={x: np.reshape(test_batch[0], (-1, 28, 28, 1)), y_: test_batch[1]})
        #     sum_accuracy += accuracy_str[0]

        #print ("accuracy: %f" % (sum_accuracy / 100.0))

if __name__ == '__main__':
    tf.app.run()

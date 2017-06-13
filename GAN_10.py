from __future__ import division
import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange
import cv2
from ops import *
from utils import *
import thread
from tensorflow.examples.tutorials.mnist import input_data

def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))
def transform(img):
    return img/127.5 - 1

class DCGAN(object):
    def __init__(self, sess, input_height=28, input_width=28, crop=True,
         batch_size=64, sample_num = 64, output_height=28, output_width=28,
         z_dim=100, gf_dim=64, df_dim=64,
         gfc_dim=1024, dfc_dim=1024, c_dim=1, dataset_name='default',
         input_fname_pattern='*.jpg', checkpoint_dir=None, sample_dir=None):
        """

        Args:
        sess: TensorFlow session
        batch_size: The size of batch. Should be specified before training.
        z_dim: (optional) Dimension of dim for Z. [100]
        gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
        df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
        gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
        dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
        c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
        """
        # self.dataset = dataset_name
        self.sess = sess
        self.crop = crop

        self.batch_size = batch_size
        self.sample_num = sample_num

        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width

        self.c_dim = c_dim

        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')

        self.d_bn3 = batch_norm(name = 'd_bn3')
        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')


        self.g_bn3 = batch_norm(name = 'g_bn')
        self.dataset_name = dataset_name
        self.input_fname_pattern = input_fname_pattern
        self.checkpoint_dir = checkpoint_dir
        self.sample_dir = sample_dir

        # self.data = glob(os.path.join("./data",self.dataset_name,self.input_fname_pattern))
        # self.c_dim = c_dim
        # self.grayscale = (self.c_dim == 1)

        # make the queue into a list
        self.file_list = []
        filename_queue = []
        self.batch_images = []
        num_preprocess_threads = 1
        min_queue_examples = 256
        image_reader = tf.WholeFileReader()
        for i in range(10):
            self.file_list.append(glob(os.path.join("./data",str(i),self.input_fname_pattern)))
            filename_queue.append(tf.train.string_input_producer(self.file_list[i][:]))
            _,image_file = image_reader.read(filename_queue[i])
            image = tf.image.decode_jpeg(image_file)
            image = tf.cast(tf.reshape(image,shape = [28,28,1]), dtype = tf.float32)

        # self.file_list = glob(os.path.join("./data",self.dataset_name,self.input_fname_pattern))
        # filename_queue = tf.train.string_input_producer(self.file_list[:])
        # image_reader = tf.WholeFileReader()
        #
        # _,image_file = image_reader.read(filename_queue)
        #
        # image = tf.image.decode_jpeg(image_file)
        #
        # image = tf.cast(tf.reshape(image,shape = [28,28,1]), dtype = tf.float32)
            self.batch_images.append(tf.train.shuffle_batch([image],
                                                batch_size = self.batch_size,
                                                num_threads = num_preprocess_threads,
                                                capacity = min_queue_examples + 3*self.batch_size,
                                                min_after_dequeue = min_queue_examples)
                                                    )
            self.batch_images[i] = transform(self.batch_images[i])
        # self.batch_images = tf.train.shuffle_batch([image],
        #                                     batch_size = self.batch_size,
        #                                     num_threads = num_preprocess_threads,
        #                                     capacity = min_queue_examples + 3*self.batch_size,
        #                                     min_after_dequeue = min_queue_examples)
        # self.batch_images = transform(self.batch_images)
        self.build_model()

    def build_model(self):

        self.G = []
        self.D = []
        self.D_logits = []
        self.D_ = []
        self.D_logits_ = []
        self.sampler_output = []
        self.d_loss_real = []
        self.d_loss_fake = []
        self.g_loss = []
        self.d_loss = []
        self.z = []
        for i in range(10):
            self.z.append(tf.placeholder(
              tf.float32, [None, self.z_dim], name='z_%s'%str(i))
              )
            self.G.append(self.generator(self.z[i],name = str(i)))
            tmp_D,tmp_D_logits = self.discriminator(self.batch_images[i],name = str(i))
            self.D.append(tmp_D)
            self.D_logits.append(tmp_D_logits)
            self.sampler_output.append(self.sampler(self.z[i],name = str(i)))
            tmp_D_ , tmp_D_logits_ = self.discriminator(self.G[i],reuse = True, name = str(i))
            self.D_.append(tmp_D_)
            self.D_logits_.append(tmp_D_logits_)
            self.d_loss_real.append(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                                    labels = tf.ones_like(self.D[i]), logits = self.D_logits[i] )))
            self.d_loss_fake.append(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                                    labels = tf.zeros_like(self.D_[i]), logits = self.D_logits_[i] )))
            self.g_loss.append(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                                    labels = tf.ones_like(self.D_[i]), logits = self.D_logits_[i] )))
            self.d_loss.append(self.d_loss_real[i] + self.d_loss_fake[i])
        # self.z = tf.placeholder(
        #   tf.float32, [None, self.z_dim], name='z')

        # self.z_sum = histogram_summary("z", self.z)


        # self.G = self.generator(self.z)
        # self.D, self.D_logits = self.discriminator(self.batch_images)

        # self.sampler = self.sampler(self.z)
        # self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)

        # self.d_sum = histogram_summary("d", self.D)
        # self.d__sum = histogram_summary("d_", self.D_)
        # self.G_sum = image_summary("G", self.G)

        # def sigmoid_cross_entropy_with_logits(x, y):
        #     try:
        #         return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
        #     except:
        #         return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

        # self.d_loss_real = tf.reduce_mean(
        #   sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
        # self.d_loss_fake = tf.reduce_mean(
        #   sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
        # self.g_loss = tf.reduce_mean(
        #   sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))

        # self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
        # self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)
        # self.d_loss = self.d_loss_real + self.d_loss_fake

        # self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
        # self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = []
        self.g_vars = []
        self.saver = tf.train.Saver()
        for i in range(10):
            self.d_vars.append([var for var in t_vars if 'd_%s'%str(i) in var.name])
            self.g_vars.append([var for var in t_vars if 'g_%s'%str(i) in var.name])

        # self.d_vars = [var for var in t_vars if 'd_' in var.name]
        # self.g_vars = [var for var in t_vars if 'g_' in var.name]
        #
        # self.saver = tf.train.Saver()

    def train(self, config):
        d_optim = []
        g_optim = []
        z_pool = []
        current_num = []
        for i in range(10):
            d_optim.append(tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                      .minimize(self.d_loss[i], var_list=self.d_vars[i]) )
            g_optim.append(tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                      .minimize(self.g_loss[i], var_list=self.g_vars[i]) )

            z_pool.append(np.load('./outputs/features_%s.npy'%str(i)) )
            current_num.append( z_pool[i].shape[0] -1 )

        # d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
        #           .minimize(self.d_loss, var_list=self.d_vars)
        # g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
        #           .minimize(self.g_loss, var_list=self.g_vars)
        # try:
        #     tf.global_variables_initializer().run()
        # except:
        #     tf.initialize_all_variables().run()

        tf.global_variables_initializer().run()

        # self.g_sum = merge_summary([self.z_sum, self.d__sum,
        #   self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
        # self.d_sum = merge_summary(
        #     [self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        # self.writer = SummaryWriter("./logs/{}/".format(self.dataset), self.sess.graph)

        # sample_z = np.random.uniform(-1, 1, size=(self.sample_num , self.z_dim))

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord = coord )

        # start_time = time.time()

        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            counter = 0
            print(" [!] Load failed...")

        # z_pool = np.load('./outputs/features_5.npy')
        # current_num = z_pool.shape[0] -1

        for i in xrange(counter,config.iter):
            # batch_z = []
            # for j in range(10):
            #     random_index = np.round(np.random.uniform(0,1,[self.batch_size]) * current_num[j]).astype(np.int32)
            #     tmp = np.zeros([self.batch_size,self.z_dim],dtype = np.float32)
            #     tmp = z_pool[j][random_index[:]]
            #     batch_z.append(tmp)

            for j in range(10):
            # random_index = np.round(np.random.uniform(0,1,[self.batch_size]) * current_num).astype(np.int32)
            # batch_z = np.zeros([self.batch_size,self.z_dim],dtype = np.float32)
            # batch_z = z_pool[random_index[:]]

            # batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]) \
            #       .astype(np.float32)
                start_time = time.time()
                random_index = np.round(np.random.uniform(0,1,[self.batch_size]) * current_num[j]).astype(np.int32)
                batch_z = z_pool[j][random_index[:]]

                _ = self.sess.run(d_optim[j],feed_dict = {self.z[j]: batch_z })
                # the same , we update G    twice
                _ = self.sess.run(g_optim[j],feed_dict = {self.z[j]:batch_z })
                _ = self.sess.run(g_optim[j],feed_dict = {self.z[j]:batch_z })
                errD_fake = self.d_loss_fake[j].eval({self.z[j] : batch_z})
                errD_real = self.d_loss_real[j].eval({self.z[j] : batch_z})
                errG = self.g_loss[j].eval({self.z[j] : batch_z})

                print("for_%s_iteration: [%2d]  time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                  % (str(j),i,time.time() - start_time, errD_fake+errD_real, errG))
                print('**************************')

            if np.mod(i ,1000) ==0:
                for j in range(10):
                    if not os.path.exists('./{}/{}'.format(config.sample_dir,str(j))):
                        os.makedirs('./{}/{}'.format(config.sample_dir,str(j)))

                    random_index = np.round(np.random.uniform(0,1,[self.batch_size]) * current_num[j]).astype(np.int32)
                    batch_z_sample = z_pool[j][random_index[:]]

                    samples = self.sess.run(self.sampler_output[j],{self.z[j] : batch_z_sample} )
                    manifold_h = int(np.ceil(np.sqrt(samples.shape[0])))
                    manifold_w = int(np.floor(np.sqrt(samples.shape[0])))
                    save_images(samples, [manifold_h, manifold_w],
                    './{}/{}/train_{:02d}.png'.format(config.sample_dir,str(j),i))
                    print ('succed save once ')
            if np.mod(i, 1000) == 0:
                self.save(self.checkpoint_dir, i)

        coord.request_stop()
        coord.join(threads)

            # _, summary_str = self.sess.run([d_optim, self.d_sum],
            #     feed_dict={ self.z: batch_z })
            # self.writer.add_summary(summary_str, i)
            #
            #   # Update G network
            # _, summary_str = self.sess.run([g_optim, self.g_sum],
            #     feed_dict={ self.z: batch_z })
            # self.writer.add_summary(summary_str, i)
            #
            #   # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
            # _, summary_str = self.sess.run([g_optim, self.g_sum],
            #     feed_dict={ self.z: batch_z })
            # self.writer.add_summary(summary_str, i)

            # errD_fake = self.d_loss_fake.eval({ self.z: batch_z })
            # errD_real = self.d_loss_real.eval()
            # errG = self.g_loss.eval({self.z: batch_z})
            #
            # print("iteration: [%2d]  time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
            #   % (i,time.time() - start_time, errD_fake+errD_real, errG))

            # if np.mod(i, 10) == 1:
            #     try:
            #         # samples, d_loss, g_loss = self.sess.run(
            #         # [self.sampler, self.d_loss, self.g_loss],
            #         # feed_dict={
            #         # self.z: sample_z,
            #         # }
            #         # )
            #         for k in range(2):
            #             sample_z = np.random.uniform(-1, 1, size=(self.sample_num , self.z_dim))
            #             samples = self.sess.run(
            #             self.sampler,
            #             feed_dict={
            #             self.z: sample_z,
            #             }
            #             )
            #             manifold_h = int(np.ceil(np.sqrt(samples.shape[0])))
            #             manifold_w = int(np.floor(np.sqrt(samples.shape[0])))
            #             save_images(samples, [manifold_h, manifold_w],
            #                 './{}/train_{:02d}_{:d}.png'.format(config.sample_dir, i,k))
            #             print ('succed save once ')
            #         # print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))
            #     except:
            #         print("one pic error!...")
            # if (i > 3000):
            #     sample_index = np.round(np.random.uniform(0,1-1e-20,[self.batch_size]) * current_num).astype(np.int32)
            #     # sample_z = np.random.uniform(-1, 1, size=(self.sample_num , self.z_dim))
            #     sample_z = np.zeros([self.batch_size,self.z_dim],dtype = np.float32)
            #     sample_z = z_pool[random_index[:]]
            #     samples = 255 * inverse_transform(self.sess.run(self.sampler,feed_dict = {self.z: sample_z}))
            #     for num_images in range(self.sample_num):
            #         cv2.imwrite('./{}/{}/{:6d}_{:2d}.png'.format(self.sample_dir,self.dataset,i,num_images),samples[num_images,:,:,:])
            # if np.mod(i, 500) == 2:
            #     self.save(self.checkpoint_dir, i)

        # coord.request_stop()
        # coord.join(threads)
        if not os.path.exists('./data_tf'):
            os.mkdir('./data_tf')
        record_filename = "./data_tf/gen.tfrecord"
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        with tf.python_io.TFRecordWriter(record_filename) as tfrecord_writer:
            for j in range(10):
                # record_filename = "%s/gen_%d.tfrecord"%('./data_tf',j)
                # with tf.python_io.TFRecordWriter(record_filename) as tfrecord_writer:
                    for k in range(100):
                        random_index = np.round(np.random.uniform(0,1,[self.batch_size]) * current_num[j]).astype(np.int32)
                        batch_z_sample = z_pool[j][random_index[:]]
                        samples = self.sess.run(self.sampler_output[j],{self.z[j] : batch_z_sample} )*255.0
                        for count in range(self.batch_size):
                            img_raw = np.reshape(samples[count,:,:,:],[28,28,1])
                            img_raw = img_raw.astype(np.uint8).tostring()
                            label_raw = np.zeros([10,])
                            label_raw[int(j)] = 1
                            label_raw = label_raw.astype(np.uint8).tostring()
                            example = to_tfexample_raw(img_raw,label_raw)
                            tfrecord_writer.write(example.SerializeToString())
                # tfrecord_writer.close()
            time.sleep(5)
            for i in range(55000):
                samples = np.reshape(mnist.train.images[i],[28,28,1])*255.0
                img_raw = samples.astype(np.uint8).tostring()

                # label_raw =  np.where(mnist.train.labels[i]>0)[0][0]
                label_raw = mnist.train.labels[i].astype(np.uint8).tostring()
                # print (label_raw)
                # print (label_raw.shape)
                # raise
                example = to_tfexample_raw(img_raw,label_raw)
                tfrecord_writer.write(example.SerializeToString())

            tfrecord_writer.close()


    def discriminator(self, image, y=None, reuse=False,name = None ):
        with tf.variable_scope("discriminator_%s"%name) as scope:
            if reuse:
                scope.reuse_variables()

            h0 = lrelu(conv2d(image, self.df_dim, name='d_%s_h0_conv'%name))
            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_%s_h1_conv'%name)))
            h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_%s_h2_conv'%name)))
            h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_%s_h3_conv'%name)))
            h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_%s_h3_lin'%name)

            return tf.nn.sigmoid(h4), h4


    def generator(self, z, y=None,name = None):
        with tf.variable_scope("generator_%s"%name) as scope:
            s_h, s_w = self.output_height, self.output_width
            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
            s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

            # project `z` and reshape
            self.z_, self.h0_w, self.h0_b = linear(
                z, self.gf_dim*8*s_h16*s_w16, 'g_%s_h0_lin'%name, with_w=True)

            self.h0 = tf.reshape(
                self.z_, [-1, s_h16, s_w16, self.gf_dim * 8])
            h0 = tf.nn.relu(self.g_bn0(self.h0))

            self.h1, self.h1_w, self.h1_b = deconv2d(
                h0, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_%s_h1'%name, with_w=True)
            h1 = tf.nn.relu(self.g_bn1(self.h1))

            h2, self.h2_w, self.h2_b = deconv2d(
                h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_%s_h2'%name, with_w=True)
            h2 = tf.nn.relu(self.g_bn2(h2))

            h3, self.h3_w, self.h3_b = deconv2d(
                h2, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_%s_h3'%name, with_w=True)
            h3 = tf.nn.relu(self.g_bn3(h3))

            h4, self.h4_w, self.h4_b = deconv2d(
                h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_%s_h4'%name, with_w=True)

            return tf.nn.tanh(h4)


    def sampler(self, z, y=None,name = None):
        with tf.variable_scope("generator_%s"%name) as scope:
            scope.reuse_variables()

            s_h, s_w = self.output_height, self.output_width
            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
            s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

            # project `z` and reshape
            h0 = tf.reshape(
                linear(z, self.gf_dim*8*s_h16*s_w16, 'g_%s_h0_lin'%name),
                [-1, s_h16, s_w16, self.gf_dim * 8])
            h0 = tf.nn.relu(self.g_bn0(h0, train=False))

            h1 = deconv2d(h0, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_%s_h1'%name)
            h1 = tf.nn.relu(self.g_bn1(h1, train=False))

            h2 = deconv2d(h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_%s_h2'%name)
            h2 = tf.nn.relu(self.g_bn2(h2, train=False))

            h3 = deconv2d(h2, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_%s_h3'%name)
            h3 = tf.nn.relu(self.g_bn3(h3, train=False))

            h4 = deconv2d(h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_%s_h4'%name)

            return tf.nn.tanh(h4)

    @property
    def model_dir(self):
        return "{}_{}_{}".format(
             self.batch_size,
            self.output_height, self.output_width)

    def save(self, checkpoint_dir, step ,name = None):
        model_name = "DCGAN.model"
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                os.path.join(checkpoint_dir, model_name),
                global_step=step)

    def load(self, checkpoint_dir, name = None):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

#
# flags = tf.app.flags
# flags.DEFINE_integer("iter", 9002, "iter to train ")
# flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
# flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
# flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
# flags.DEFINE_integer("sample_num", 64, "The size of sample images ")
# flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
# flags.DEFINE_integer("input_height", 28, "The size of image to use (will be center cropped). [108]")
# flags.DEFINE_integer("input_width", 28, "The size of image to use (will be center cropped). If None, same value as input_height [None]")
# flags.DEFINE_integer("output_height", 28, "The size of the output images to produce [64]")
# flags.DEFINE_integer("output_width", 28, "The size of the output images to produce. If None, same value as output_height [None]")
# flags.DEFINE_string("dataset", "5", "The name of dataset [...]")
# flags.DEFINE_string("input_fname_pattern", "*.jpg", "Glob pattern of filename of input images [*]")
# flags.DEFINE_string("checkpoint_dir", "./checkpoint", "Directory name to save the checkpoints [checkpoint]")
# flags.DEFINE_string("sample_dir", "./samples", "Directory name to save the image samples [samples]")
# flags.DEFINE_boolean("train", True, "True for training, False for testing [False]")
# flags.DEFINE_boolean("crop", True, "True for training, False for testing [False]")
# flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
# FLAGS = flags.FLAGS
#
# def main(_):
#
#     if not os.path.exists(FLAGS.checkpoint_dir):
#         os.makedirs(FLAGS.checkpoint_dir)
#     if not os.path.exists(FLAGS.sample_dir):
#         os.makedirs(FLAGS.sample_dir)
#
#     run_config = tf.ConfigProto()
#     run_config.gpu_options.allow_growth=True
#
#     with tf.Session(config = run_config) as sess:
#         dcgan = DCGAN(
#                 sess,
#                 input_width=FLAGS.input_width,
#                 input_height=FLAGS.input_height,
#                 output_width=FLAGS.output_width,
#                 output_height=FLAGS.output_height,
#                 batch_size=FLAGS.batch_size,
#                 sample_num=FLAGS.sample_num,
#                 # dataset_name=FLAGS.dataset,
#                 input_fname_pattern=FLAGS.input_fname_pattern,
#                 crop=FLAGS.crop,
#                 checkpoint_dir=FLAGS.checkpoint_dir,
#                 sample_dir=FLAGS.sample_dir)
#         dcgan.train(FLAGS)
#
# if __name__ == '__main__':
#     tf.app.run()

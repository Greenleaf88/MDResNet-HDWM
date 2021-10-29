# coding=UTF-8
import os
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import cv2
import math
import skimage
import matplotlib.pyplot as plt
import tensorflow as tf
#from tensorflow.python.data.ops.iterator_ops import Iterator
#from tensorflow.data import Iterator
Iterator = tf.data.Iterator
from Dataset import WatermarkDataLoader

from utils import Chi_Square_Loss_V2, Mean_Chi_Square_Loss_V2, attack_py_func, normalized_correlation

from ResNet import network
from HDimageProcess import getEmbeddingRegion, getRevealingRegion, normlizeWatermark, replaceStegoRegion, rotateImg, rotateWm, drawKeypoint, drawRegion
from utils import compute_nc, compute_psnr, compute_ssim, compute_ber

import random


class Model():

    def __init__(self):
        self.run_time = time.strftime("%m%d-%H%M")
        # self.learning_rate = 0.0003
        self.starter_learning_rate = 0.001

        self.epoches = 30
        self.thres = 0.9
        self.log_path = 'logs/' + self.run_time + '/'
        # self.alpha = 0.0005
        self.batch_size = 4
        # self.batch_size = 2
        self.multi_scales = [128, 256, 512]

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        self.sess = tf.InteractiveSession(config=config)
        self.current_scale = tf.placeholder(
            dtype=tf.int32, name="current_scale")
        self.secret_scale = self.current_scale
        self.secret_tensor = tf.placeholder(
            shape=[None, None, None, 1], dtype=tf.float32, name="secret_tensor")
        self.cover_tensor = tf.placeholder(
            shape=[None, None, None, 3], dtype=tf.float32, name="cover_tensor")
        self.stego_tensor = tf.placeholder(
            shape=[None, None, None, 1], dtype=tf.float32, name="stego_tensor")
        self.global_step_tensor = tf.Variable(
            0, trainable=False, name='global_step')

    def get_y_channel(self, tensor):
        Y = 0 + 0.299 * tensor[:, :, :, 0] + 0.587 * \
            tensor[:, :, :, 1] + 0.114 * tensor[:, :, :, 2]
        Y = tf.expand_dims(Y, -1)
        return Y

    def get_hiding_network_op(self, cover_tensor, secret_tensor, is_training):

        Y = 0 + 0.299 * cover_tensor[:, :, :, 0] + 0.587 * \
            cover_tensor[:, :, :, 1] + 0.114 * cover_tensor[:, :, :, 2]
        CB = 128.0 / 255 - 0.168736 * \
            cover_tensor[:, :, :, 0] - 0.331264 * \
            cover_tensor[:, :, :, 1] + 0.5 * cover_tensor[:, :, :, 2]
        CR = 128.0 / 255 + 0.5 * cover_tensor[:, :, :, 0] - 0.418688 * cover_tensor[
            :, :, :, 1] - 0.081312 * cover_tensor[:, :, :, 2]

        Y = tf.expand_dims(Y, -1)
        CB = tf.expand_dims(CB, -1)
        CR = tf.expand_dims(CR, -1)

        concat_input = tf.concat(
            [Y, secret_tensor], axis=-1, name='images_features_concat')
        y_output = network(
            concat_input, n_class=1, is_training=is_training, name='encode')

        output_r = y_output + 1.402 * CR - 1.402 * 128.0 / 255
        output_g = y_output - 0.344136 * CB + 0.344136 * \
            128.0 / 255 - 0.714136 * CR + 0.714136 * 128.0 / 255
        output_b = y_output + 1.772 * CB - 1.772 * 128.0 / 255
        output = tf.concat(
            [output_r, output_g, output_b], axis=-1, name='rgb_concat')

        return y_output, output

    def get_reveal_network_op(self, container_tensor, is_training, transform=False):
        if transform:
            container_tensor = 0 + 0.299 * \
                container_tensor[
                    :, :, :, 0] + 0.587 * container_tensor[:, :, :, 1] + 0.114 * container_tensor[:, :, :, 2]
            container_tensor = tf.expand_dims(container_tensor, -1)
        output = network(
            container_tensor, n_class=1, is_training=is_training, name='decode')
        # output = tf.nn.sigmoid(output)
        return output

    def get_noise_layer_op_with_py_func(self, tensor, secret_tensor, mode='train'):
        stego_noise, secret_noise = tf.py_func(
            attack_py_func,
            [tensor, secret_tensor],
            [tf.float32, tf.float32])

        stego_noise.set_shape([None, None, None, 3])
        secret_noise.set_shape([None, None, None, 1])

        return stego_noise, secret_noise

    def get_noise_layer_op(self, tensor, secret_tensor, current_scale=512, mode='train'):

        print 'add noise using mode ', mode

        def scale_up(tensor, secret_tensor, current_scale):

            # tf.random_uniform([1], 1, 1.1, dtype=tf.float32)[0]
            up_scale = 1.2
            input_size = tf.cast(
                (up_scale * tf.cast(current_scale, tf.float32)), tf.int32)
            up_size = tf.cast(
                (up_scale * tf.cast(self.secret_scale, tf.float32)), tf.int32)
            tensor = tf.image.resize_images(tensor, [input_size, input_size])

            tensor = tf.image.resize_image_with_crop_or_pad(
                tensor, current_scale, current_scale)
            secret_tensor = tf.image.resize_images(
                secret_tensor, [up_size, up_size])

            secret_tensor = tf.image.resize_image_with_crop_or_pad(
                secret_tensor, self.secret_scale, self.secret_scale)

            return tensor, secret_tensor

        def scale_down(tensor, secret_tensor, current_scale):

            # tf.random_uniform([1], 0.7, 1, dtype=tf.float32)[0]
            down_scale = 0.8
            input_size = tf.cast(
                (down_scale * tf.cast(current_scale, tf.float32)), tf.int32)
            down_size = tf.cast(
                (down_scale * tf.cast(self.secret_scale, tf.float32)), tf.int32)
            tensor = tf.image.resize_images(tensor, [input_size, input_size])
            tensor = tf.image.resize_image_with_crop_or_pad(
                tensor, current_scale, current_scale)

            secret_tensor = tf.image.resize_images(
                secret_tensor, [down_size, down_size])
            secret_tensor = tf.image.resize_image_with_crop_or_pad(
                secret_tensor, self.secret_scale, self.secret_scale)

            return tensor, secret_tensor

        def rotate(tensor, secret_tensor, angle=0.):
            if angle == 0.:
                angle = tf.random_uniform(
                    [1], math.radians(1), math.radians(10), dtype=tf.float32)
            tensor = tf.contrib.image.rotate(tensor, angle)
            secret_tensor = tf.contrib.image.rotate(secret_tensor, angle)
            return tensor, secret_tensor

        def translate(tensor, secret_tensor):
            size = 20  # tf.random_uniform([1], 0, 10, dtype=tf.float32)[0]
            size1 = 0
            tensor = tf.contrib.image.translate(tensor, [size1, size])
            secret_tensor = tf.contrib.image.translate(
                secret_tensor, [size1, size])
            return tensor, secret_tensor

        def jpeg_encoding(tensor, secret_tensor):
            # quality = tf.random_uniform([1], 75, 95, dtype=tf.int32)[0]
            tensor = tf.cast(tf.clip_by_value(tensor, 0, 1) * 255, tf.uint8)
            tensor = tf.stack([tf.image.decode_jpeg(tf.image.encode_jpeg(
                tensor[i], quality=80), channels=3) for i in range(self.batch_size)], 0)
            tensor = tf.cast(tensor, tf.float32) / 255.
            return tensor, secret_tensor

        def gaussian_kernel(size, mean, std):
            """Makes 2D gaussian Kernel for convolution."""

            d = tf.distributions.Normal(mean, std)

            vals = d.prob(
                tf.range(start=-size, limit=size + 1, dtype=tf.float32))

            gauss_kernel = tf.einsum('i,j->ij', vals, vals)

            return gauss_kernel / tf.reduce_sum(gauss_kernel)

        def gaussian_blur(tensor, secret_tensor):

            # Make Gaussian Kernel with desired specs.
            # gauss_kernel = gaussian_kernel(2, 0.0, 0.7)
            gauss_kernel = gaussian_kernel(1, 0.0, 1.0)

            # Expand dimensions of `gauss_kernel` for `tf.nn.conv2d` signature.
            gauss_kernel = gauss_kernel[:, :, tf.newaxis, tf.newaxis]
            gauss_kernel = tf.concat(
                [gauss_kernel, gauss_kernel, gauss_kernel], axis=2)
            # Convolve.
            tensor = tf.nn.depthwise_conv2d(
                tensor, gauss_kernel, strides=[1, 1, 1, 1], padding="SAME")
            return tensor, secret_tensor

        def add_gaussain_noise(tensor, secret_tensor):
            random = tf.random_normal(
                shape=tf.shape(tensor), mean=0.0, stddev=0.005, dtype=tf.float32)
            # random =tf.random_normal(shape=tf.shape(tensor), mean=0.0, stddev=0.005, dtype=tf.float32)
            tensor = tensor + random
            # secret_tensor = secret_tensor + random
            return tensor, secret_tensor

        def add_sp_noise(tensor, secret_tensor):
            mask = tf.random_uniform(
                shape=tf.shape(tensor),  minval=0.0, maxval=1.0, dtype=tf.float32)
            tensor = tensor + tf.cast(mask > 0.995, tf.float32)
            tensor = tensor + tf.cast(mask < 0.005, tf.float32) * (-1)
            tensor = tf.clip_by_value(tensor, 0, 1)
            return tensor, secret_tensor

        def add_gamma_correction(tensor, secret_tensor, gamma):
            tensor = tf.pow(tf.clip_by_value(tensor, 0, 1), gamma)
            #tensor = tf.clip_by_value(tensor, 0, 1)

            return tensor, secret_tensor

        def add_noise(tensor, secret_tensor):
            with tf.variable_scope("transform_layer"):
                stego_noise = tensor
                secret_noise = tf.cast(secret_tensor, tf.int32)
                stego_noise, secret_noise = tf.cond(tf.random_uniform([1])[0] > self.thres,
                                                    lambda: add_gaussain_noise(
                                                        stego_noise, secret_noise),
                                                    lambda: (stego_noise, secret_noise))
                stego_noise, secret_noise = tf.cond(tf.random_uniform([1])[0] > self.thres,
                                                    lambda: add_sp_noise(
                                                        stego_noise, secret_noise),
                                                    lambda: (stego_noise, secret_noise))
                stego_noise, secret_noise = tf.cond(tf.random_uniform([1])[0] > self.thres,
                                                    lambda: jpeg_encoding(
                                                        stego_noise, secret_noise),
                                                    lambda: (stego_noise, secret_noise))
                stego_noise, secret_noise = tf.cond(tf.random_uniform([1])[0] > self.thres,
                                                    lambda: gaussian_blur(
                                                        stego_noise, secret_noise),
                                                    lambda: (stego_noise, secret_noise))
                stego_noise, secret_noise = tf.cond(tf.random_uniform([1])[0] > self.thres,
                                                    lambda: rotate(
                                                        stego_noise, secret_noise),
                                                    lambda: (stego_noise, secret_noise))
                stego_noise, secret_noise = tf.cond(tf.random_uniform([1])[0] > self.thres,
                                                    lambda: translate(
                                                        stego_noise, secret_noise),
                                                    lambda: (stego_noise, secret_noise))
                stego_noise, secret_noise = tf.cond(tf.random_uniform([1])[0] > self.thres,
                                                    lambda: scale_up(
                                                        stego_noise, secret_noise, current_scale),
                                                    lambda: (stego_noise, tf.cast(secret_noise, tf.float32)))
                stego_noise, secret_noise = tf.cond(tf.random_uniform([1])[0] > self.thres,
                                                    lambda: scale_down(
                                                        stego_noise, secret_noise, current_scale),
                                                    lambda: (stego_noise, tf.cast(secret_noise, tf.float32)))
                secret_noise = tf.cast(secret_noise, tf.float32)
                return stego_noise, secret_noise

        def add_la(tensor, secret_tensor):
            secret_tensor = tf.cast(secret_tensor, tf.float32)
            tensor = tf.add(tensor, -0.2)

            tensor = tensor / 0.6

            tensor = tf.clip_by_value(tensor, 0, 1)

            return tensor, secret_tensor

        def add_la2(tensor, secret_tensor):
            secret_tensor = tf.cast(secret_tensor, tf.float32)

            tensor = tensor * 0.6

            tensor = tf.add(tensor, 0.2)

            tensor = tf.clip_by_value(tensor, 0, 1)

            return tensor, secret_tensor

        def add_nothing(tensor, secret_tensor):
            return tensor, secret_tensor

        if mode == 'test':
            #             secret_noise = tf.cast(secret_tensor, tf.int32)
            #             stego_noise, secret_tensor = rotate(
            #                 tensor, secret_tensor, angle=math.radians(6.4))
            #             stego_noise, secret_noise = rotate(
            #                 tensor, secret_noise, angle=math.radians(10))
            #             stego_noise, secret_tensor = jpeg_encoding(tensor, secret_tensor)

            #             stego_noise, secret_tensor = add_sp_noise(
            #                 stego_noise, secret_tensor)
            #             secret_noise = tf.cast(secret_tensor, tf.int32)
            stego_noise, secret_tensor = scale_down(tensor, secret_tensor, 512)
            stego_noise, secret_tensor = gaussian_blur(
                stego_noise, secret_tensor)
            #             stego_noise, secret_tensor = add_gaussain_noise(
            #                 tensor, secret_tensor)

            #             stego_noise, secret_tensor = add_la2(
            #                 tensor, secret_tensor)

            #                         secret_noise = tf.cast(secret_noise, tf.float32)
#             stego_noise, secret_tensor = self.get_noise_layer_op_with_py_func(
#                 tensor, secret_tensor, mode='test')
#             stego_noise, secret_tensor = add_gamma_correction(
#                 stego_noise, secret_tensor, 2)
#             stego_noise, secret_tensor = translate(tensor, secret_tensor)

            return stego_noise, secret_tensor

        if mode == 'train':
            stego_noise, secret_noise = tf.cond(tf.random_uniform([1])[0] > self.thres,
                                                lambda: add_noise(
                                                    tensor, secret_tensor),
                                                lambda: add_nothing(tensor, secret_tensor))
            return stego_noise, secret_noise

        if mode == 'None':
            return tensor, secret_tensor

    def get_loss_op(self, secret_true, secret_pred, cover_true, cover_pred, test=False):

        with tf.variable_scope("huber_losses"):
            if test:

                cover_pred = tf.clip_by_value(cover_pred, 0, 1)

            secret_mse = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=secret_true, logits=secret_pred))

            cover_mse = tf.losses.mean_squared_error(cover_true, cover_pred)

        with tf.variable_scope("ssim_losses"):
            if test:
                secret_pred = tf.cast(secret_pred > 0.5, tf.float32)
                secret_ssim = normalized_correlation(secret_pred, secret_true)
                cover_ssim = tf.reduce_mean(
                    tf.image.ssim(cover_true, cover_pred, max_val=1.0))
            else:
                secret_ssim = tf.zeros(shape=[1])[0]

                cover_ssim = 1 - \
                    tf.reduce_mean(
                        tf.image.ssim(cover_true, cover_pred, max_val=1.0))

        G_final_loss = cover_mse + secret_mse + cover_ssim

        return G_final_loss, secret_mse, cover_mse, secret_ssim, cover_ssim

    def get_tensor_to_img_op(self, tensor):
        with tf.variable_scope("", reuse=True):

            return tf.clip_by_value(tensor, 0, 1)

    def prepare_training_graph(self, secret_tensor, cover_tensor, global_step_tensor):
        cover_tensor = tf.image.resize_images(
            cover_tensor, [self.current_scale, self.current_scale])
        secret_tensor = tf.image.resize_images(
            secret_tensor, [self.secret_scale, self.secret_scale])

        y_output, hidden = self.get_hiding_network_op(
            cover_tensor=cover_tensor, secret_tensor=secret_tensor, is_training=True)
        hidden_transformed, secret_tensor_transformed = self.get_noise_layer_op(
            hidden, secret_tensor, self.current_scale, mode='train')
        y_output_transformed = self.get_y_channel(hidden_transformed)
        reveal_output_op = self.get_reveal_network_op(
            y_output_transformed, is_training=True)

        G_final_loss, secret_mse, cover_mse, secret_ssim, cover_ssim = self.get_loss_op(
            secret_tensor_transformed, reveal_output_op, cover_tensor, hidden)

        global_variables = tf.global_variables()
        gan_varlist = [
            i for i in global_variables if i.name.startswith('Discriminator')]

        en_de_code_varlist = [
            i for i in global_variables if i not in gan_varlist]

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            G_minimize_op = tf.train.AdamOptimizer(self.learning_rate).minimize(
                G_final_loss, var_list=en_de_code_varlist, global_step=global_step_tensor)

        tf.summary.scalar('G_loss', G_final_loss, family='train')
        tf.summary.scalar('secret_mse', secret_mse, family='train')
        tf.summary.scalar('cover_mse', cover_mse, family='train')
        tf.summary.scalar('learning_rate', self.learning_rate, family='train')

        # tf.summary.scalar('secret_ssim', secret_ssim)
        tf.summary.scalar('cover_ssim', cover_ssim, family='train')

        tf.summary.image('secret', self.get_tensor_to_img_op(
            secret_tensor), max_outputs=1, family='train')
        tf.summary.image('secret_attacked', self.get_tensor_to_img_op(
            secret_tensor_transformed), max_outputs=1, family='train')
        tf.summary.image('cover', self.get_tensor_to_img_op(
            cover_tensor), max_outputs=1, family='train')
        tf.summary.image(
            'stego', self.get_tensor_to_img_op(hidden), max_outputs=1, family='train')
        tf.summary.image('stego_attacked', self.get_tensor_to_img_op(
            hidden_transformed), max_outputs=1, family='train')
        tf.summary.image('secret_revealed', self.get_tensor_to_img_op(
            tf.nn.sigmoid(reveal_output_op)), max_outputs=1, family='train')

        merged_summary_op = tf.summary.merge_all()

        return G_minimize_op, G_final_loss, merged_summary_op, secret_mse, cover_mse, secret_ssim, cover_ssim

    def prepare_test_graph(self, secret_tensor, cover_tensor):
        secret_tensor = tf.image.resize_images(
            secret_tensor, [self.current_scale, self.current_scale])

        y_output, hidden = self.get_hiding_network_op(
            cover_tensor=cover_tensor, secret_tensor=secret_tensor, is_training=False)

        hidden_transformed, secret_transformed = self.get_noise_layer_op(
            hidden, secret_tensor, self.current_scale, mode='test')
        y_output_transformed = self.get_y_channel(hidden_transformed)
        reveal_output_op = self.get_reveal_network_op(
            y_output_transformed, is_training=False)
        reveal_output_image = tf.nn.sigmoid(reveal_output_op)

        G_final_loss, secret_mse, cover_mse, secret_ssim, cover_ssim = self.get_loss_op(
            secret_transformed, reveal_output_op, cover_tensor, hidden, test=True)

        return hidden, reveal_output_image, hidden_transformed, secret_transformed, G_final_loss, secret_mse, cover_mse, secret_ssim, cover_ssim

    def save_chkp(self, path):
        global_step = self.sess.run(self.global_step_tensor)
        self.saver.save(self.sess, path, global_step)

    def load_chkp(self, path):
        self.saver.restore(self.sess, path)
        print("LOADED")

    def train(self):

        with tf.device('/gpu:0'):

            segdl = WatermarkDataLoader('/home/gl/Documents/ILSVRC2012_img_val/'.replace('/home/gl', '/mnt/2T/moliq'),
                                        '/home/gl/Documents/binaryimages/'.replace(
                                            '/home/gl', '/mnt/2T/moliq'), self.batch_size,
                                        (512, 512), (512, 512),
                                        'dataset/watermark_train.txt', split='train')

            segdl_val = WatermarkDataLoader('/home/gl/Documents/ILSVRC2012_img_val/'.replace('/home/gl', '/mnt/2T/moliq'),
                                            '/home/gl/Documents/binaryimages/'.replace(
                                                '/home/gl', '/mnt/2T/moliq'), self.batch_size,
                                            (512, 512), (512, 512),
                                            'dataset/watermark_valid.txt', split='val')

        steps_per_epoch = segdl.data_len / segdl.batch_size
        steps_per_epoch_val = segdl_val.data_len / segdl_val.batch_size

        self.learning_rate = tf.train.exponential_decay(
            self.starter_learning_rate, self.global_step_tensor, steps_per_epoch * 8, 0.1, staircase=True)

        self.train_op_G, G_final_loss, self.summary_op, self.secret_mse, self.cover_mse, self.secret_ssim, self.cover_ssim = \
            self.prepare_training_graph(
                self.secret_tensor, self.cover_tensor, self.global_step_tensor)

        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        self.writer = tf.summary.FileWriter(self.log_path, self.sess.graph)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=50)

        loader = tf.train.latest_checkpoint('logs/0710-2105')
        global_variables = tf.global_variables()
        exclude_vars1 = ['beta1_power:0', 'beta2_power:0',
                         'global_step:0', 'beta1_power_1:0', 'beta2_power_1:0']
        encode_decode_varlist = [
            i for i in global_variables if not i.name in exclude_vars1]
        custom_saver = tf.train.Saver(var_list=encode_decode_varlist)
        custom_saver.restore(self.sess, loader)
        print('load model %s' % loader)

        for epoch in range(1, 1 + self.epoches):
            with tf.device('/cpu:0'):

                segdl = WatermarkDataLoader('/home/gl/Documents/ILSVRC2012_img_val/'.replace('/home/gl', '/mnt/2T/moliq'),
                                            '/home/gl/Documents/binaryimages/'.replace(
                                                '/home/gl', '/mnt/2T/moliq'), self.batch_size,
                                            (512, 512), (512, 512),
                                            'dataset/watermark_train.txt', split='train')

                segdl_val = WatermarkDataLoader('/home/gl/Documents/ILSVRC2012_img_val/'.replace('/home/gl', '/mnt/2T/moliq'),
                                                '/home/gl/Documents/binaryimages/'.replace(
                                                    '/home/gl', '/mnt/2T/moliq'), self.batch_size,
                                                (512, 512), (512, 512),
                                                'dataset/watermark_valid.txt', split='val')

                iterator = Iterator.from_structure(
                    segdl.data_tr.output_types, segdl.data_tr.output_shapes)
                iterator_val = Iterator.from_structure(
                    segdl_val.data_tr.output_types, segdl_val.data_tr.output_shapes)
                next_batch = iterator.get_next()
                next_batch_val = iterator_val.get_next()
                training_init_op = iterator.make_initializer(segdl.data_tr)
                training_init_op_val = iterator_val.make_initializer(
                    segdl_val.data_tr)

            print('epoch %d' % epoch)
            self.sess.run(training_init_op)
            for i in range(steps_per_epoch):
                try:
                    scale = self.multi_scales[i % 3]
                    cover_tensor, secret_tensor = self.sess.run(next_batch)

                except:
                    continue
                _, G_loss, secret_mse, cover_mse, secret_ssim, cover_ssim, summary, global_step = \
                    self.sess.run([self.train_op_G, G_final_loss, self.secret_mse, self.cover_mse, self.secret_ssim, self.cover_ssim, self.summary_op, self.global_step_tensor],
                                  feed_dict={self.secret_tensor: secret_tensor, self.cover_tensor: cover_tensor, self.current_scale: scale})

                if i % 10 == 0:
                    self.writer.add_summary(summary, global_step)

                if i % 100 == 0:
                    print('Epoch [{}/{}]  Step [{}/{}] G_final_Loss {:.4f}  encoder_ssim {:.4f} encoder_mse {:.4f}'
                          '  decoder_ssim {:.4f}  decoder_mse {:.4f} '.format(
                              epoch, self.epoches, i, steps_per_epoch, G_loss,
                              cover_ssim, cover_mse, secret_ssim, secret_mse))

            # run validation
            self.sess.run(training_init_op_val)

            secret_ssim_this_epoch = []
            secret_mse_this_epoch = []
            cover_ssim_this_epoch = []
            cover_mse_this_epoch = []

            for i in range(steps_per_epoch_val):
                cover_tensor_val, secret_tensor_val = self.sess.run(
                    next_batch_val)
                secret_mse, cover_mse, secret_ssim, cover_ssim = \
                    self.sess.run([self.secret_mse, self.cover_mse, self.secret_ssim, self.cover_ssim],
                                  feed_dict={self.secret_tensor: secret_tensor_val,
                                             self.cover_tensor: cover_tensor_val, self.current_scale: 512})

                secret_ssim_this_epoch.append(secret_ssim)
                secret_mse_this_epoch.append(secret_mse)
                cover_ssim_this_epoch.append(cover_ssim)
                cover_mse_this_epoch.append(cover_mse)
            mean_secret_ssim_this_epoch = sum(
                secret_ssim_this_epoch) / len(secret_ssim_this_epoch)
            mean_secret_mse_this_epoch = sum(
                secret_mse_this_epoch) / len(secret_mse_this_epoch)
            mean_cover_ssim_this_epoch = sum(
                cover_ssim_this_epoch) / len(cover_ssim_this_epoch)
            mean_cover_mse_this_epoch = sum(
                cover_mse_this_epoch) / len(cover_mse_this_epoch)
            print('VALIDATION Epoch {} global step {} valid_encoder_ssim {:.4f} valid_decoder_ssim {:.4f}, valid_encoder_mse {:.4f} valid_decoder_mse {:.4f}'.format(
                epoch, global_step, mean_cover_ssim_this_epoch, mean_secret_ssim_this_epoch, mean_cover_mse_this_epoch, mean_secret_mse_this_epoch))

            self.save_chkp(self.log_path)


#    parameters :
#   imgPath: the path of high definition image
#   key: user defined key, should be 5~6 digits
#   num: the number of embedding regions, 4 or 5   is optimal
def test_embedding(imgPath, key, num):
    log_path = 'logs/0710-2105'
    train_model = Model()
    input_placeholder = tf.placeholder(
        shape=[None, None, None, 3], dtype=tf.float32)
    reveal_output_op = train_model.get_reveal_network_op(
        input_placeholder, is_training=False, transform=True)
    # # reveal_output_op = self.get_reveal_network_op(self.stego_tensor, is_training=False, transform=True)
    # reveal_output_op = self.get_reveal_network_op(self.stego_tensor, is_training=False)
    reveal_output_op = tf.nn.sigmoid(reveal_output_op)

    y_output, hidden = train_model.get_hiding_network_op(
        cover_tensor=train_model.cover_tensor, secret_tensor=train_model.secret_tensor, is_training=False)

    if os.path.exists(log_path + '.meta'):
        loader = log_path
    else:
        loader = tf.train.latest_checkpoint(log_path)
#     global_variables = tf.global_variables()
#     encode_vars = [
#         i for i in global_variables if i.name.startswith('encode')]
    train_model.sess.run(tf.global_variables_initializer())
    train_model.saver = tf.train.Saver()
    train_model.saver.restore(train_model.sess, loader)
    print('load model %s' % loader)
    img = cv2.imread(imgPath)
    if img is None or img.size <= 0:
        os._exit(0)
    regions = getEmbeddingRegion(img, key, num)
    for i in range(0, num):
        if regions[i] is not None:
            edge = regions[i].xmax - regions[i].xmin
            embeddingRegion = img[
                regions[i].ymin:regions[i].ymax, regions[i].xmin:regions[i].xmax]

            dir = imgPath[0:imgPath.rindex('/')]
            secret_origin = normlizeWatermark(
                'data/secret.jpg', edge)
            extract_image = embeddingRegion / 255.0
            secret_image = (
                np.expand_dims(secret_origin, 0) > 128).astype(np.uint8)
            secret_image = np.expand_dims(secret_image, -1)
            # stego = train_model.get_stego(
            #    'logs/0621-0153', np.expand_dims(extract_image, 0).astype(np.float32), secret_image.astype(np.float32))
            stego = train_model.sess.run(
                hidden,
                feed_dict={train_model.secret_tensor: secret_image.astype(np.float32),
                           train_model.cover_tensor: np.expand_dims(extract_image, 0).astype(np.float32)})

            stego = (np.clip(stego, 0, 1) * 255).astype(np.uint8)

#             cv2.imwrite('data/regionstego.jpg',
#                         stego[0])
            img = replaceStegoRegion(img, stego[0], regions[i])
    cv2.imwrite('data/stego.jpg',
                img)
    img = cv2.imread('data/stego.jpg')
    nc = 0.0
    key2 = 123457
    regions = getEmbeddingRegion(img, key2, num)

    i = 0
    while nc < 0.6 and i < num:
        if regions[i] is not None:
            edge = regions[i].xmax - regions[i].xmin
            embeddingRegion = img[
                regions[i].ymin:regions[i].ymax, regions[i].xmin:regions[i].xmax]
            cv2.imwrite('data/redetectedstego.jpg',
                        embeddingRegion)
            stego = np.expand_dims(embeddingRegion, 0)
            secret_reveal = train_model.sess.run(
                reveal_output_op, feed_dict={input_placeholder: stego / 255.})
            secret_image = (
                np.clip(secret_reveal, 0, 1) * 255).astype(np.uint8)

            #secret_image = train_model.get_secret('logs/0621-0153', stego / 255.)
            edgeinner = int(edge * 0.6)
            startidx = int(round((edge - edgeinner) / 2))

            wm = secret_image[0]
            secret_origin = cv2.imread(
                'data/secret.jpg', cv2.IMREAD_GRAYSCALE)
            secret_origin = cv2.resize(secret_origin, (edgeinner, edgeinner))

            res = cv2.matchTemplate(
                wm, secret_origin, cv2.TM_CCOEFF_NORMED)
            nc = np.max(res)
            loc = np.where(res >= nc)
            loc = zip(*loc[::-1])[0]
#
            cv2.imwrite('data/revealedsecret.jpg',
                        (secret_image[0]).astype(np.uint8))
            ber = compute_ber(secret_origin, wm[loc[1]:loc[1] + edgeinner,
                                                loc[0]:loc[0] + edgeinner])

            print 'nc=', nc, 'ber=', ber
        i = i + 1


def test_embedding_images(image_dir, key=123456, rnum=5):
    log_path = 'logs/0710-2105'
    train_model = Model()
    input_placeholder = tf.placeholder(
        shape=[None, None, None, 3], dtype=tf.float32)
    reveal_output_op = train_model.get_reveal_network_op(
        input_placeholder, is_training=False, transform=True)
    # # reveal_output_op = self.get_reveal_network_op(self.stego_tensor, is_training=False, transform=True)
    # reveal_output_op = self.get_reveal_network_op(self.stego_tensor, is_training=False)
    reveal_output_op = tf.nn.sigmoid(reveal_output_op)

    y_output, hidden = train_model.get_hiding_network_op(
        cover_tensor=train_model.cover_tensor, secret_tensor=train_model.secret_tensor, is_training=False)

    if os.path.exists(log_path + '.meta'):
        loader = log_path
    else:
        loader = tf.train.latest_checkpoint(log_path)
#     global_variables = tf.global_variables()
#     encode_vars = [
#         i for i in global_variables if i.name.startswith('encode')]
    train_model.sess.run(tf.global_variables_initializer())
    train_model.saver = tf.train.Saver()
    train_model.saver.restore(train_model.sess, loader)
    print('load model %s' % loader)
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

    dir = 'results_encrypt'
    for folder in ['regionstego', 'stego', 'redetectedstego', 'revealedsecret']:
        if not os.path.exists(os.path.join(dir, folder)):
            os.makedirs(os.path.join(dir, folder))

    image_names = os.listdir(image_dir)
    image_names.sort()
    total_nc = 0.0
    total_ber = 0.0
    sum_psnr_region = 0.0
    sum_psnr_whole = 0.0
    sum_ssim_region = 0.0
    sum_ssim_whole = 0.0
    num = 0
    embedding_time = 0.0
    revealing_time = 0.0
    for i, img_name in enumerate(image_names):
        #         if i >= 10:
        #             break
        print img_name
        img = cv2.imread(
            os.path.join(image_dir, img_name))

        if img is None or img.size <= 0:
            os._exit(0)
        cover_img = img.copy()
        start_ = time.time()

        regions = getEmbeddingRegion(img, key, rnum)
        edge = regions[0].xmax - regions[0].xmin
        secret_origin = normlizeWatermark(
            'data/secret.jpg', edge)
        secret_image = (
            np.expand_dims(secret_origin, 0) > 128).astype(np.uint8)
        secret_image = np.expand_dims(secret_image, -1)
        for j in range(0, rnum):
            if regions[j] is not None:

                embeddingRegion = img[
                    regions[j].ymin:regions[j].ymax, regions[j].xmin:regions[j].xmax]

                extract_image = embeddingRegion / 255.0

                stego = train_model.sess.run(
                    hidden,
                    feed_dict={train_model.secret_tensor: secret_image.astype(np.float32),
                               train_model.cover_tensor: np.expand_dims(extract_image, 0).astype(np.float32)})

                stego = (np.clip(stego, 0, 1) * 255).astype(np.uint8)

                cv2.imwrite(dir + '/regionstego/%s' % (img_name),
                            stego[0])
                psnr_region = compute_psnr(embeddingRegion, stego[0])
                ssim_region = compute_ssim(embeddingRegion, stego[0])
                sum_psnr_region = sum_psnr_region + psnr_region
                sum_ssim_region = sum_ssim_region + ssim_region
                num = num + 1
                img = replaceStegoRegion(img, stego[0], regions[j])
        end_ = time.time()
        embedding_time = embedding_time + (end_ - start_)

        cv2.imwrite(dir + '/stego/%s' % (img_name),
                    img)
        img = cv2.imread(dir + '/stego/%s' % (img_name))
        psnr_whole = compute_psnr(img, cover_img)
        ssim_whole = compute_ssim(img, cover_img)
        sum_psnr_whole = sum_psnr_whole + psnr_whole
        sum_ssim_whole = sum_ssim_whole + ssim_whole
        start_ = time.time()
        regions = getRevealingRegion(img, key, rnum)
        j = 0
        nc = 0.0
        max_nc = 0.0
        min_ber = 1.0
        edge = regions[0].xmax - regions[0].xmin
        edgeinner = int(edge * 0.6)
        startidx = int(round((edge - edgeinner) / 2))
        secret_origin = cv2.imread(
            'data/secret.jpg', cv2.IMREAD_GRAYSCALE)
        secret_origin = cv2.resize(
            secret_origin, (edgeinner, edgeinner))
        while max_nc < 0.6 and j < rnum:
            if regions[j] is not None:

                embeddingRegion = img[
                    regions[j].ymin:regions[j].ymax, regions[j].xmin:regions[j].xmax]
                cv2.imwrite(dir + '/redetectedstego/%s' % (img_name),
                            embeddingRegion)
                stego = np.expand_dims(embeddingRegion, 0)

                secret_reveal = train_model.sess.run(
                    reveal_output_op, feed_dict={input_placeholder: stego / 255.})
                secret_image = (
                    np.clip(secret_reveal, 0, 1) * 255).astype(np.uint8)

                wm = secret_image[0]

                res = cv2.matchTemplate(
                    wm, secret_origin, cv2.TM_CCOEFF_NORMED)
                nc = np.max(res)
                loc = np.where(res >= nc)
                loc = zip(*loc[::-1])[0]

#                 nc = compute_nc(secret_origin, wm[startidx:startidx + edgeinner,
# startidx:startidx + edgeinner], edgeinner)
                ber = compute_ber(secret_origin, wm[loc[1]:loc[1] + edgeinner,
                                                    loc[0]:loc[0] + edgeinner])
                if max_nc < nc:
                    max_nc = nc
                    min_ber = ber
                    cv2.imwrite(dir + '/revealedsecret/%s' % (img_name),
                                (secret_image[0]).astype(np.uint8))
                    print 'nc=', nc, 'ber=', ber
            j = j + 1
        end_ = time.time()
        revealing_time = revealing_time + (end_ - start_)
        total_nc = total_nc + max_nc
        total_ber = total_ber + min_ber

    avg_em_time = embedding_time / (i + 1)
    avg_revealing_time = revealing_time / (i + 1)
    print embedding_time, revealing_time, (i + 1), avg_em_time, avg_revealing_time

    avg_nc = total_nc / (i + 1)
    avg_ber = total_ber / (i + 1)
    avg_psnr_region = sum_psnr_region / num
    avg_psnr_whole = sum_psnr_whole / (i + 1)
    avg_ssim_region = sum_ssim_region / num
    avg_ssim_whole = sum_ssim_whole / (i + 1)
    fo = open("results_encrytlqzhu.txt", "w")
    fo.write("psnr_region=%s,psnr_whole=%s,ssim_region=%s,ssim_whole=%s,nc=%s,ber=%s,em_time=%s,re_time=%s" %
             (avg_psnr_region, avg_psnr_whole, avg_ssim_region, avg_ssim_whole, avg_nc, avg_ber, avg_em_time, avg_revealing_time))

# 关闭文件
    fo.close()


if __name__ == '__main__':

    test_embedding(
        'data/DSC_0206.JPG', 123456, 4)

    # train_model.train()
#     train_model = Model()

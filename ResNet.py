import time
import tensorflow as tf
import tensorflow.contrib as tf_contrib
import numpy as np


# Xavier : tf_contrib.layers.xavier_initializer()
# He : tf_contrib.layers.variance_scaling_initializer()
# Normal : tf.random_normal_initializer(mean=0.0, stddev=0.02)
# l2_decay : tf_contrib.layers.l2_regularizer(0.0001)

weight_init = tf_contrib.layers.variance_scaling_initializer()
weight_regularizer = tf_contrib.layers.l2_regularizer(0.0001)


##########################################################################
# Layer
##########################################################################

def SRM_conv2d(x, channels=30, kernel=5, stride=1, padding='SAME', dilation=1, use_bias=False, scope='conv_SRM', reuse=True):
    SRM_npy = np.load('SRM_Kernels.npy').astype(dtype=np.float32)
    SRM_npy = np.transpose(SRM_npy, [2, 3, 1, 0])
    # SRM_npy = np.concatenate([SRM_npy, SRM_npy, SRM_npy], axis=2)
    # with tf.name_scope('convSRM') as scope:
    with tf.variable_scope('convSRM', reuse=reuse) as scope:
        kernel = tf.get_variable(
            name='weights', initializer=SRM_npy, trainable=False)
        conv = tf.nn.conv2d(x, kernel, strides=[1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable(shape=[30], dtype=tf.float32, initializer=tf.zeros_initializer(),
                                 trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
    return out


def conv(x, channels, kernel=3, stride=2, padding='SAME', dilation=1, use_bias=True, scope='conv_0'):
    with tf.variable_scope(scope):
        x = tf.layers.conv2d(inputs=x, filters=channels,
                             kernel_size=kernel, dilation_rate=dilation, kernel_initializer=weight_init,
                             kernel_regularizer=weight_regularizer,
                             strides=stride, use_bias=use_bias, padding=padding)
        # x = tf.layers.separable_conv2d(inputs=x, filters=channels,
        #                                kernel_size=kernel, dilation_rate=dilation, depthwise_initializer=weight_init, pointwise_initializer=weight_init,
        #                                depthwise_regularizer=weight_regularizer, pointwise_regularizer=weight_regularizer,
        # strides=stride, use_bias=use_bias, padding=padding)

    return x


def deconv_layer(x, channels, is_training, kernel=3, stride=2, padding='SAME', use_bias=True, scope='conv_0'):
    with tf.variable_scope(scope):
        x = tf.layers.conv2d_transpose(inputs=x, filters=channels,
                                       kernel_size=kernel, kernel_initializer=weight_init,
                                       kernel_regularizer=weight_regularizer,
                                       strides=stride, use_bias=use_bias, padding=padding)
        x = batch_norm(x, is_training=is_training)
        x = relu(x)
    return x


def fully_conneted(x, units, use_bias=True, scope='fully_0'):
    with tf.variable_scope(scope):
        x = tf.layers.dense(x, units=units, kernel_initializer=weight_init,
                            kernel_regularizer=weight_regularizer, use_bias=use_bias)
        return x


def ResUnit(x_init, channels, is_training=True, dilate=1, use_bias=True, downsample=False, upsample=False, scope='resblock'):
    with tf.variable_scope(scope):

        x = conv(x_init, channels, kernel=1, stride=1,
                 use_bias=use_bias, scope='conv_0')
        x = relu(x)

        x = conv(x, channels, kernel=3, stride=1,
                 use_bias=use_bias, scope='conv_1')
        x = relu(x)

        if downsample == True:
            x = conv(
                x, channels, kernel=1, stride=2, use_bias=use_bias, scope='conv_2')
            x1 = conv(x_init, channels, kernel=1, stride=2,
                      use_bias=use_bias, scope='conv_3')
        elif upsample == True:
            x = deconv_layer(x, channels, is_training, kernel=1, stride=2,
                             padding='SAME', use_bias=use_bias, scope='deconv_1')
            x1 = deconv_layer(x_init, channels, is_training, kernel=1,
                              stride=2, padding='SAME', use_bias=use_bias, scope='deconv_2')
        else:
            x = conv(
                x, channels, kernel=1, stride=1, use_bias=use_bias, scope='conv_2')
            x1 = conv(x_init, channels, kernel=1, stride=1,
                      use_bias=use_bias, scope='conv_3')

        x_concat = tf.concat([x, x1], axis=-1)
        x_out = relu(x_concat)

        return x_out


def resblock(x_init, channels, is_training=True, dilate=1, use_bias=True, downsample=False, scope='resblock'):
    with tf.variable_scope(scope):

        x = batch_norm(x_init, is_training, scope='batch_norm_0')
        x = relu(x)

        if downsample:
            x = conv(
                x, channels, kernel=3, stride=2, use_bias=use_bias, scope='conv_0')
            x_init = conv(
                x_init, channels, kernel=1, stride=2, use_bias=use_bias, scope='conv_init')

        else:
            x = conv(x, channels, kernel=3, dilation=dilate,
                     stride=1, use_bias=use_bias, scope='conv_0')

        x = batch_norm(x, is_training, scope='batch_norm_1')
        x = relu(x)
        x = conv(x, channels, kernel=3, stride=1, dilation=dilate,
                 use_bias=use_bias, scope='conv_1')

        return x + x_init


def bottle_resblock(x_init, channels, is_training=True, dilate=1, use_bias=True, downsample=False, scope='bottle_resblock'):
    with tf.variable_scope(scope):
        x = batch_norm(x_init, is_training, scope='batch_norm_1x1_front')
        shortcut = relu(x)

        x = conv(shortcut, channels, dilation=dilate,  kernel=1,
                 stride=1, use_bias=use_bias, scope='conv_1x1_front')
        x = batch_norm(x, is_training, scope='batch_norm_3x3')
        x = relu(x)

        if downsample:
            x = conv(x, channels, dilation=dilate, kernel=3,
                     stride=2, use_bias=use_bias, scope='conv_0')
            shortcut = conv(shortcut, channels * 4, dilation=dilate,
                            kernel=1, stride=2, use_bias=use_bias, scope='conv_init')

        else:
            x = conv(x, channels, dilation=dilate, kernel=3,
                     stride=1, use_bias=use_bias, scope='conv_0')
            shortcut = conv(shortcut, channels * 4, dilation=dilate,
                            kernel=1, stride=1, use_bias=use_bias, scope='conv_init')

        x = batch_norm(x, is_training, scope='batch_norm_1x1_back')
        x = relu(x)
        x = conv(x, channels * 4, dilation=dilate, kernel=1,
                 stride=1, use_bias=use_bias, scope='conv_1x1_back')

        return x + shortcut


def get_residual_layer(res_n):
    x = []

    if res_n == 18:
        x = [2, 2, 2, 2]

    if res_n == 34:
        x = [3, 4, 6, 3]

    if res_n == 50:
        x = [3, 4, 6, 3]

    if res_n == 101:
        x = [3, 4, 23, 3]

    if res_n == 152:
        x = [3, 8, 36, 3]

    return x


##########################################################################
# Sampling
##########################################################################

def flatten(x):
    return tf.layers.flatten(x)


def global_avg_pooling(x):
    gap = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
    return gap


def avg_pooling(x, pool_size=2, stride=2):
    return tf.layers.average_pooling2d(x, pool_size=pool_size, strides=stride, padding='SAME')

##########################################################################
# Activation function
##########################################################################


def relu(x):
    return tf.nn.relu(x)


##########################################################################
# Normalization function
##########################################################################

def batch_norm(x, is_training=True, scope='batch_norm'):
    return tf_contrib.layers.batch_norm(x,
                                        decay=0.9, epsilon=1e-05,
                                        center=True, scale=True, updates_collections=None,
                                        is_training=is_training, scope=scope)

##########################################################################
# Loss function
##########################################################################


def classification_loss(logit, label):
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(labels=label, logits=logit))
    prediction = tf.equal(tf.argmax(logit, -1), tf.argmax(label, -1))
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

    return loss, accuracy


def resnet_nopooling(x, n_class, name, res_n=50, is_training=True, reuse=False, dilate=[1, 1, 1, 1]):
    # with tf.variable_scope("res_network", reuse=reuse):
    with tf.variable_scope(name, reuse=reuse):

        if res_n < 50:
            residual_block = resblock
        else:
            residual_block = bottle_resblock

        residual_list = get_residual_layer(res_n)

        ch = 8  # paper is 64
        x = conv(x, channels=ch, kernel=3, stride=1, scope='conv')

        dilation_rate = dilate[0]
        for i in range(residual_list[0]):
            x = residual_block(x, channels=ch, is_training=is_training,
                               dilate=dilation_rate, downsample=False, scope='resblock0_' + str(i))

        #######################################################################

        x = residual_block(x, channels=ch * 2, is_training=is_training,
                           dilate=dilation_rate, downsample=True, scope='resblock1_0')

        dilation_rate = dilate[1]
        for i in range(1, residual_list[1]):
            x = residual_block(x, channels=ch * 2, dilate=dilation_rate, is_training=is_training, downsample=False,
                               scope='resblock1_' + str(i))

        #######################################################################

        x = residual_block(x, channels=ch * 4, dilate=dilation_rate,
                           is_training=is_training, downsample=True, scope='resblock2_0')

        dilation_rate = dilate[2]
        for i in range(1, residual_list[2]):
            x = residual_block(x, channels=ch * 4, dilate=dilation_rate, is_training=is_training, downsample=False,
                               scope='resblock2_' + str(i))

        #######################################################################

        x = residual_block(x, channels=ch * 2, dilate=dilation_rate,
                           is_training=is_training, downsample=False, scope='resblock_3_0')

        dilation_rate = dilate[3]
        for i in range(1, residual_list[3]):
            x = residual_block(x, channels=ch * 2, dilate=dilation_rate, is_training=is_training, downsample=False,
                               scope='resblock_3_' + str(i))

        #######################################################################

        x = conv(x, channels=n_class, kernel=3, stride=1, scope='conv_last')

        # x = batch_norm(x, is_training, scope='batch_norm')
        # x = relu(x)

        # x = global_avg_pooling(x)
        # x = fully_conneted(x, units=self.label_dim, scope='logit')

        return x


def network2(x, z, n_class, name, is_training=True, reuse=False, block_type='resblock'):

    with tf.variable_scope(name, reuse=reuse):

        if block_type == 'resblock':
            residual_block = resblock
        else:
            residual_block = bottle_resblock

        ch = 32

        x = conv(x, channels=ch, kernel=5, stride=1, scope='conv0')

        x = residual_block(x, channels=ch, is_training=is_training,
                           dilate=1, downsample=False, scope='resblock1_0')
        x = residual_block(x, channels=ch, is_training=is_training,
                           dilate=1, downsample=False, scope='resblock1_1')
        #######################################################################
        x = residual_block(x, channels=ch, is_training=is_training,
                           dilate=1, downsample=True, scope='resblock1_2')
#         ch = 8
#         z = conv(z, channels=ch, kernel=5, stride=1, scope='conv0')
#
#         z = residual_block(z, channels=ch, is_training=is_training,
#                            dilate=1, downsample=False, scope='resblock1_0')
#         z = residual_block(z, channels=ch, is_training=is_training,
#                            dilate=1, downsample=False, scope='resblock1_1')
#         #######################################################################
#         z = residual_block(z, channels=ch, is_training=is_training,
#                            dilate=1, downsample=True, scope='resblock1_2')

        x = residual_block(x, channels=ch, is_training=is_training,
                           dilate=1, downsample=False, scope='resblock2_0')
        x = residual_block(x, channels=ch, is_training=is_training,
                           dilate=1, downsample=False, scope='resblock2_1')
        pred_up1 = deconv_layer(
            x, channels=ch, is_training=is_training, stride=2, scope='deconv_0')
        #######################################################################
        x = residual_block(x, channels=ch, is_training=is_training,
                           dilate=1, downsample=True, scope='resblock2_2')
        # high level embedding######################################

        #############################################################
        x = residual_block(x, channels=ch, is_training=is_training,
                           dilate=1, downsample=False, scope='resblock3_0')
        x = residual_block(x, channels=ch, is_training=is_training,
                           dilate=2, downsample=False, scope='resblock3_1')
        x = residual_block(x, channels=ch, is_training=is_training,
                           dilate=3, downsample=False, scope='resblock3_2')
        pred_up2 = deconv_layer(
            x, channels=ch, is_training=is_training, stride=4, scope='deconv_1')

        x = residual_block(x, channels=ch, is_training=is_training,
                           dilate=3, downsample=False, scope='resblock4_0')
        x = residual_block(x, channels=ch, is_training=is_training,
                           dilate=2, downsample=False, scope='resblock4_1')
        x = residual_block(x, channels=ch, is_training=is_training,
                           dilate=1, downsample=False, scope='resblock4_2')

        pred_up3 = deconv_layer(
            x, channels=ch, is_training=is_training, stride=4, scope='deconv_2')

        pred_concat = tf.concat(
            [pred_up1, pred_up2, pred_up3], axis=-1, name='pred_concatnation')
        final_pred = conv(
            pred_concat, channels=n_class, kernel=3, stride=1, scope='conv_last')

        return final_pred


def network(x, n_class, name, is_training=True, reuse=False, block_type='resblock'):

    with tf.variable_scope(name, reuse=reuse):

        if block_type == 'resblock':
            residual_block = resblock
        else:
            residual_block = bottle_resblock

        ch = 32

        x = conv(x, channels=ch, kernel=5, stride=1, scope='conv0')

        x = residual_block(x, channels=ch, is_training=is_training,
                           dilate=1, downsample=False, scope='resblock1_0')
        x = residual_block(x, channels=ch, is_training=is_training,
                           dilate=1, downsample=False, scope='resblock1_1')
        #######################################################################
        x = residual_block(x, channels=ch, is_training=is_training,
                           dilate=1, downsample=True, scope='resblock1_2')

        x = residual_block(x, channels=ch, is_training=is_training,
                           dilate=1, downsample=False, scope='resblock2_0')
        x = residual_block(x, channels=ch, is_training=is_training,
                           dilate=1, downsample=False, scope='resblock2_1')
        pred_up1 = deconv_layer(
            x, channels=ch, is_training=is_training, stride=2, scope='deconv_0')
        #######################################################################
        x = residual_block(x, channels=ch, is_training=is_training,
                           dilate=1, downsample=True, scope='resblock2_2')

        x = residual_block(x, channels=ch, is_training=is_training,
                           dilate=1, downsample=False, scope='resblock3_0')
        x = residual_block(x, channels=ch, is_training=is_training,
                           dilate=2, downsample=False, scope='resblock3_1')
        x = residual_block(x, channels=ch, is_training=is_training,
                           dilate=3, downsample=False, scope='resblock3_2')
        pred_up2 = deconv_layer(
            x, channels=ch, is_training=is_training, stride=4, scope='deconv_1')

        x = residual_block(x, channels=ch, is_training=is_training,
                           dilate=3, downsample=False, scope='resblock4_0')
        x = residual_block(x, channels=ch, is_training=is_training,
                           dilate=2, downsample=False, scope='resblock4_1')
        x = residual_block(x, channels=ch, is_training=is_training,
                           dilate=1, downsample=False, scope='resblock4_2')

        pred_up3 = deconv_layer(
            x, channels=ch, is_training=is_training, stride=4, scope='deconv_2')

        pred_concat = tf.concat(
            [pred_up1, pred_up2, pred_up3], axis=-1, name='pred_concatnation')
        final_pred = conv(
            pred_concat, channels=n_class, kernel=3, stride=1, scope='conv_last')

        return final_pred


def WMencoder(x, y, n_class, name, is_training=True, reuse=False, block_type='resblock'):

    with tf.variable_scope(name, reuse=reuse):

        if block_type == 'resblock':
            residual_block = ResUnit
        else:
            residual_block = bottle_resblock

        cover = x
        ch = 8
        x = residual_block(x, channels=ch, is_training=is_training,
                           dilate=1, downsample=True, upsample=False, scope='resblock_1')
        x = residual_block(x, channels=16, is_training=is_training,
                           dilate=1, downsample=True, upsample=False, scope='resblock_2')
        x = residual_block(x, channels=32, is_training=is_training,
                           dilate=1, downsample=True, upsample=False, scope='resblock_3')

        x = tf.concat([x, y], axis=-1)

        x = residual_block(x, channels=16, is_training=is_training,
                           dilate=1, downsample=False, upsample=True, scope='resblock_4')
        x = residual_block(x, channels=8, is_training=is_training,
                           dilate=1, downsample=False, upsample=True, scope='resblock_5')
        x = residual_block(x, channels=4, is_training=is_training,
                           dilate=1, downsample=False, upsample=True, scope='resblock_6')

        x = conv(x, channels=n_class, kernel=3, stride=1, scope='conv_last')

        # out1 = cover + 0.01 * 0.8 * x
        # out = 0.8 * (out1 - cover) + cover

        out1 = cover + 0.01 * 0.90 * x
        out = 0.9 * (out1 - cover) + cover

        return out


def WMdecoder(x, n_class, name, is_training=True, reuse=False, block_type='resblock'):

    with tf.variable_scope(name, reuse=reuse):

        if block_type == 'resblock':
            residual_block = ResUnit
        else:
            residual_block = bottle_resblock

        ch = 8
        x = residual_block(x, channels=ch, is_training=is_training,
                           dilate=1, downsample=True, upsample=False, scope='resblock_1')
        x = residual_block(x, channels=16, is_training=is_training,
                           dilate=1, downsample=True, upsample=False, scope='resblock_2')
        x = residual_block(x, channels=32, is_training=is_training,
                           dilate=1, downsample=True, upsample=False, scope='resblock_3')

        x = conv(x, channels=n_class, kernel=3, stride=1, scope='conv_last2')

        out = x
        return out


class ResNet(object):

    def __init__(self, sess, args):
        self.model_name = 'ResNet'
        self.sess = sess
        self.dataset_name = args.dataset

        if self.dataset_name == 'cifar10':
            self.train_x, self.train_y, self.test_x, self.test_y = load_cifar10()
            self.img_size = 32
            self.c_dim = 3
            self.label_dim = 10

        if self.dataset_name == 'cifar100':
            self.train_x, self.train_y, self.test_x, self.test_y = load_cifar100()
            self.img_size = 32
            self.c_dim = 3
            self.label_dim = 100

        if self.dataset_name == 'mnist':
            self.train_x, self.train_y, self.test_x, self.test_y = load_mnist()
            self.img_size = 28
            self.c_dim = 1
            self.label_dim = 10

        if self.dataset_name == 'fashion-mnist':
            self.train_x, self.train_y, self.test_x, self.test_y = load_fashion()
            self.img_size = 28
            self.c_dim = 1
            self.label_dim = 10

        if self.dataset_name == 'tiny':
            self.train_x, self.train_y, self.test_x, self.test_y = load_tiny()
            self.img_size = 64
            self.c_dim = 3
            self.label_dim = 200

        self.checkpoint_dir = args.checkpoint_dir
        self.log_dir = args.log_dir

        self.res_n = args.res_n

        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.iteration = len(self.train_x) // self.batch_size

        self.init_lr = args.lr

    ##########################################################################
    # Generator
    ##########################################################################

    def network(self, x, is_training=True, reuse=False):
        with tf.variable_scope("network", reuse=reuse):

            if self.res_n < 50:
                residual_block = resblock
            else:
                residual_block = bottle_resblock

            residual_list = get_residual_layer(self.res_n)

            ch = 32  # paper is 64
            x = conv(x, channels=ch, kernel=3, stride=1, scope='conv')

            for i in range(residual_list[0]):
                x = residual_block(
                    x, channels=ch, is_training=is_training, downsample=False, scope='resblock0_' + str(i))

            ###################################################################

            x = residual_block(
                x, channels=ch * 2, is_training=is_training, downsample=True, scope='resblock1_0')

            for i in range(1, residual_list[1]):
                x = residual_block(
                    x, channels=ch * 2, is_training=is_training, downsample=False, scope='resblock1_' + str(i))

            ###################################################################

            x = residual_block(
                x, channels=ch * 4, is_training=is_training, downsample=True, scope='resblock2_0')

            for i in range(1, residual_list[2]):
                x = residual_block(
                    x, channels=ch * 4, is_training=is_training, downsample=False, scope='resblock2_' + str(i))

            ###################################################################

            x = residual_block(
                x, channels=ch * 8, is_training=is_training, downsample=True, scope='resblock_3_0')

            for i in range(1, residual_list[3]):
                x = residual_block(
                    x, channels=ch * 8, is_training=is_training, downsample=False, scope='resblock_3_' + str(i))

            ###################################################################

            x = batch_norm(x, is_training, scope='batch_norm')
            x = relu(x)

            x = global_avg_pooling(x)
            x = fully_conneted(x, units=self.label_dim, scope='logit')

            return x

    ##########################################################################
    # Model
    ##########################################################################

    def build_model(self):
        """ Graph Input """
        self.train_inptus = tf.placeholder(tf.float32, [
                                           self.batch_size, self.img_size, self.img_size, self.c_dim], name='train_inputs')
        self.train_labels = tf.placeholder(
            tf.float32, [self.batch_size, self.label_dim], name='train_labels')

        self.test_inptus = tf.placeholder(tf.float32, [len(
            self.test_x), self.img_size, self.img_size, self.c_dim], name='test_inputs')
        self.test_labels = tf.placeholder(
            tf.float32, [len(self.test_y), self.label_dim], name='test_labels')

        self.lr = tf.placeholder(tf.float32, name='learning_rate')

        """ Model """
        self.train_logits = self.network(self.train_inptus)
        self.test_logits = self.network(
            self.test_inptus, is_training=False, reuse=True)

        self.train_loss, self.train_accuracy = classification_loss(
            logit=self.train_logits, label=self.train_labels)
        self.test_loss, self.test_accuracy = classification_loss(
            logit=self.test_logits, label=self.test_labels)

        """ Training """
        self.optim = tf.train.MomentumOptimizer(
            self.lr, momentum=0.9).minimize(self.train_loss)

        """" Summary """
        self.summary_train_loss = tf.summary.scalar(
            "train_loss", self.train_loss)
        self.summary_train_accuracy = tf.summary.scalar(
            "train_accuracy", self.train_accuracy)

        self.summary_test_loss = tf.summary.scalar("test_loss", self.test_loss)
        self.summary_test_accuracy = tf.summary.scalar(
            "test_accuracy", self.test_accuracy)

        self.train_summary = tf.summary.merge(
            [self.summary_train_loss, self.summary_train_accuracy])
        self.test_summary = tf.summary.merge(
            [self.summary_test_loss, self.summary_test_accuracy])

    ##########################################################################
    # Train
    ##########################################################################

    def train(self):
        # initialize all variables
        tf.global_variables_initializer().run()

        # saver to save model
        self.saver = tf.train.Saver()

        # summary writer
        self.writer = tf.summary.FileWriter(
            self.log_dir + '/' + self.model_dir, self.sess.graph)

        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            epoch_lr = self.init_lr
            start_epoch = (int)(checkpoint_counter / self.iteration)
            start_batch_id = checkpoint_counter - start_epoch * self.iteration
            counter = checkpoint_counter

            if start_epoch >= int(self.epoch * 0.75):
                epoch_lr = epoch_lr * 0.01
            elif start_epoch >= int(self.epoch * 0.5) and start_epoch < int(self.epoch * 0.75):
                epoch_lr = epoch_lr * 0.1
            print(" [*] Load SUCCESS")
        else:
            epoch_lr = self.init_lr
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

        # loop for epoch
        start_time = time.time()
        for epoch in range(start_epoch, self.epoch):
            if epoch == int(self.epoch * 0.5) or epoch == int(self.epoch * 0.75):
                epoch_lr = epoch_lr * 0.1

            # get batch data
            for idx in range(start_batch_id, self.iteration):
                batch_x = self.train_x[
                    idx * self.batch_size:(idx + 1) * self.batch_size]
                batch_y = self.train_y[
                    idx * self.batch_size:(idx + 1) * self.batch_size]

                batch_x = data_augmentation(
                    batch_x, self.img_size, self.dataset_name)

                train_feed_dict = {
                    self.train_inptus: batch_x,
                    self.train_labels: batch_y,
                    self.lr: epoch_lr
                }

                test_feed_dict = {
                    self.test_inptus: self.test_x,
                    self.test_labels: self.test_y
                }

                # update network
                _, summary_str, train_loss, train_accuracy = self.sess.run(
                    [self.optim, self.train_summary, self.train_loss, self.train_accuracy], feed_dict=train_feed_dict)
                self.writer.add_summary(summary_str, counter)

                # test
                summary_str, test_loss, test_accuracy = self.sess.run(
                    [self.test_summary, self.test_loss, self.test_accuracy], feed_dict=test_feed_dict)
                self.writer.add_summary(summary_str, counter)

                # display training status
                counter += 1
                print("Epoch: [%2d] [%5d/%5d] time: %4.4f, train_accuracy: %.2f, test_accuracy: %.2f, learning_rate : %.4f"
                      % (epoch, idx, self.iteration, time.time() - start_time, train_accuracy, test_accuracy, epoch_lr))

            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading
            # pre-trained model
            start_batch_id = 0

            # save model
            self.save(self.checkpoint_dir, counter)

        # save model for final step
        self.save(self.checkpoint_dir, counter)

    @property
    def model_dir(self):
        return "{}{}_{}_{}_{}".format(self.model_name, self.res_n, self.dataset_name, self.batch_size, self.init_lr)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(
            checkpoint_dir, self.model_name + '.model'), global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(
                self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(ckpt_name.split('-')[-1])
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def test(self):
        tf.global_variables_initializer().run()

        self.saver = tf.train.Saver()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)

        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        test_feed_dict = {
            self.test_inptus: self.test_x,
            self.test_labels: self.test_y
        }

        test_accuracy = self.sess.run(
            self.test_accuracy, feed_dict=test_feed_dict)
        print("test_accuracy: {}".format(test_accuracy))


def xunet(x, n_class, name, is_training=True, reuse=False):

    with tf.variable_scope(name, reuse=reuse):
        # x = x*255.
        # x = tf.image.rgb_to_grayscale(x, name='cvt_color')
        # x = SRM_conv2d(x, reuse=reuse)
        x = conv(x, channels=8, kernel=5, stride=1, scope='conv1')
        x = tf.abs(x)
        x = batch_norm(x, is_training=is_training, scope='bn1')
        x = relu(x)
        x = avg_pooling(x, pool_size=5)

        x = conv(x, channels=16, kernel=5, stride=1, scope='conv2')
        x = batch_norm(x, is_training=is_training, scope='bn2')
        x = tf.nn.tanh(x)
        x = avg_pooling(x, pool_size=5)

        x = conv(x, channels=32, kernel=1, stride=1, scope='conv3')
        x = batch_norm(x, is_training=is_training, scope='bn3')
        x = tf.nn.tanh(x)
        x = avg_pooling(x, pool_size=5)

        x = conv(x, channels=64, kernel=1, stride=1, scope='conv4')
        x = batch_norm(x, is_training=is_training, scope='bn4')
        x = tf.nn.tanh(x)
        x = avg_pooling(x, pool_size=5)

        x = conv(x, channels=128, kernel=1, stride=1, scope='conv5')
        x = batch_norm(x, is_training=is_training, scope='bn5')
        x = tf.nn.tanh(x)
        x = avg_pooling(x, pool_size=32, stride=32)

        x = flatten(x)
        x.set_shape([None, 128])
        x = fully_conneted(x, 128, use_bias=True, scope='fc1')
        x = relu(x)
        logits = fully_conneted(x, 1, use_bias=False, scope='fc2')
        pred = tf.nn.sigmoid(logits)
    return logits, pred

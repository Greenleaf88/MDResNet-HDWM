import glob
import os
import numpy as np
import tensorflow as tf
import cv2
import math
import skimage
import random

from skimage import transform

import matplotlib.pyplot as plt
# from sklearn.metrics import mean_squared_error




def bin_count():
    a = tf.ones(shape=[256, 256, 3], dtype=tf.int32)
    bins = tf.bincount(a, maxlength=5)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        result = sess.run(bins)
        print(result)


def Chi_Square_Loss(pred_image, true_image):
    pred_image = tf.cast(pred_image * 255, tf.int32)
    true_image = tf.cast(true_image * 255, tf.int32)
    pred_bins = tf.bincount(
        pred_image, minlength=0, maxlength=255, dtype=tf.float32)
    true_bins = tf.bincount(
        true_image, minlength=0, maxlength=255, dtype=tf.float32)
    loss = tf.square(pred_bins - true_bins) / (pred_bins + true_bins)
    loss = tf.reduce_mean(loss) / 255.
    return loss


def Chi_Square_Loss_V2(pred_image, true_image, epsilon=0.01):
    pred_bins = tf.histogram_fixed_width(
        tf.reshape(pred_image, [-1]), value_range=[0.0, 1.0], nbins=256, dtype=tf.int32)
    true_bins = tf.histogram_fixed_width(
        tf.reshape(true_image, [-1]), value_range=[0.0, 1.0], nbins=256, dtype=tf.int32)
    loss = tf.cast(tf.square(pred_bins - true_bins), tf.float32) / \
        (tf.cast((pred_bins + true_bins), tf.float32) +
         tf.constant(epsilon, tf.float32))
    # loss = tf.reduce_mean(loss)/255.
    loss = tf.reduce_mean(loss)
    # return pred_bins, true_bins, loss
    return loss


def Mean_Chi_Square_Loss_V2(pred_image, true_image, epsilon=0.01):
    pred_bins = tf.map_fn(
        lambda x: tf.histogram_fixed_width(
            tf.reshape(x, [-1]), value_range=[0.0, 1.0], nbins=256, dtype=tf.int32),
        pred_image, dtype=tf.int32)
    true_bins = tf.map_fn(
        lambda x: tf.histogram_fixed_width(
            tf.reshape(x, [-1]), value_range=[0.0, 1.0], nbins=256, dtype=tf.int32),
        true_image, dtype=tf.int32)
    loss = tf.cast(tf.square(pred_bins - true_bins), tf.float32) / \
        (tf.cast((pred_bins + true_bins), tf.float32) +
         tf.constant(epsilon, tf.float32))
    loss = tf.reduce_mean(loss)
    # loss = tf.reduce_mean(loss, axis=1)
    # loss = tf.reduce_mean(loss, axis=0)
    # return pred_bins, true_bins, loss
    return loss

# def Mean_Chi_Square_Loss_V2(pred_image, true_image, batch_size):
#     # losses = []
#
#     losses = tf.map_fn(lambda x: Chi_Square_Loss_V2(x[0], x[1]), (pred_image, true_image))
#     # loss = Chi_Square_Loss_V2(pred_image[i], true_image[i])
#     # losses.append(loss)
#     total_loss = tf.reduce_mean(losses)
#     return total_loss


def Chi_Square_Loss_test():
    cover = np.load('logs/0 0.799 0.013 0.002 0.136 0.051_cover.npy')
    secret = np.load('logs/0 0.799 0.013 0.002 0.136 0.051_secret.npy')
    secret_reveal = np.load(
        'logs/0 0.799 0.013 0.002 0.136 0.051_secret_reveal.npy')
    stego = np.load('logs/0 0.799 0.013 0.002 0.136 0.051_stego.npy')
    # a = to_image(secret[0])
    # b = to_image(secret_reveal[0])
    secret_true = tf.placeholder(shape=[256, 256, 3], dtype=tf.float32)
    secret_pred = tf.placeholder(shape=[256, 256, 3], dtype=tf.float32)
    pred_bins_tensor, true_bins_tensor, loss = Chi_Square_Loss_V2(
        secret_pred, secret_true)
    # loss = Chi_Square_Loss_V2(secret_pred, secret_true)
    with tf.Session() as sess:
        for i in range(cover.shape[0]):
            print(i)
            sess.run(tf.global_variables_initializer())
            pred_bins, true_bins, result = sess.run([pred_bins_tensor, true_bins_tensor, loss], feed_dict={
                                                    secret_true: secret[0], secret_pred: secret_reveal[0]})
            # result = sess.run(loss, feed_dict={secret_true:secret[i], secret_pred: secret_reveal[i]})
            print(result)
            # result = sess.run(loss, feed_dict={secret_true:cover[i], secret_pred: stego[i]})
            pred_bins, true_bins, result = sess.run([pred_bins_tensor, true_bins_tensor, loss], feed_dict={
                                                    secret_true: cover[i], secret_pred: stego[i]})
            print(result)


def Mean_Chi_Square_Loss_test():
    cover = np.load('logs/0 0.799 0.013 0.002 0.136 0.051_cover.npy')
    secret = np.load('logs/0 0.799 0.013 0.002 0.136 0.051_secret.npy')
    secret_reveal = np.load(
        'logs/0 0.799 0.013 0.002 0.136 0.051_secret_reveal.npy')
    stego = np.load('logs/0 0.799 0.013 0.002 0.136 0.051_stego.npy')
    # a = to_image(secret[0])
    # b = to_image(secret_reveal[0])
    secret_true = tf.placeholder(shape=[None, 256, 256, 3], dtype=tf.float32)
    secret_pred = tf.placeholder(shape=[None, 256, 256, 3], dtype=tf.float32)
    loss = Mean_Chi_Square_Loss_V2(secret_pred, secret_true)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        result = sess.run(
            [loss], feed_dict={secret_true: secret, secret_pred: secret_reveal})
        print(result)
        result = sess.run(
            [loss], feed_dict={secret_true: cover, secret_pred: stego})
        print(result)
        # pred_bins, true_bins, result = sess.run([pred_bins_tensor, true_bins_tensor, loss], feed_dict={secret_true:cover[i], secret_pred: stego[i]})
        # print(result)


def Mean_Chi_Square_Loss_test_():
    cover = np.load('logs/0 0.799 0.013 0.002 0.136 0.051_cover.npy')
    secret = np.load('logs/0 0.799 0.013 0.002 0.136 0.051_secret.npy')
    secret_reveal = np.load(
        'logs/0 0.799 0.013 0.002 0.136 0.051_secret_reveal.npy')
    stego = np.load('logs/0 0.799 0.013 0.002 0.136 0.051_stego.npy')
    # a = to_image(secret[0])
    # b = to_image(secret_reveal[0])
    secret_true = tf.placeholder(shape=[None, 256, 256, 3], dtype=tf.float32)
    secret_pred = tf.placeholder(shape=[None, 256, 256, 3], dtype=tf.float32)
    # pred_bins_tensor, true_bins_tensor, loss = Chi_Square_Loss_V2(secret_pred, secret_true)
    loss = Chi_Square_Loss_V2(secret_pred, secret_true)
    with tf.Session() as sess:
        result = sess.run(
            loss, feed_dict={secret_true: secret, secret_pred: secret_reveal})
        print(result)


def luv_loss(pred_image, true_image):
    delta_r_square = tf.reduce_mean(
        tf.squared_difference(pred_image[..., 0], true_image[..., 0]))
    delta_g_square = tf.reduce_mean(
        tf.squared_difference(pred_image[..., 1], true_image[..., 1]))
    delta_b_square = tf.reduce_mean(
        tf.squared_difference(pred_image[..., 2], true_image[..., 2]))
    mean_r = tf.reduce_mean(pred_image[..., 0] / 2 + true_image[..., 0] / 2)
    final_loss = tf.sqrt((2 + mean_r) * delta_r_square +
                         4 * delta_g_square + (3 - mean_r) * delta_b_square)
    return final_loss


def rgb2yuv_tf(image):
    Y = 0.299 * image[:, :, :, 0] + 0.587 * \
        image[:, :, :, 1] + 0.114 * image[:, :, :, 2]
    U = - 0.14713 * image[:, :, :, 0] - 0.28886 * \
        image[:, :, :, 1] + 0.436 * image[:, :, :, 2]
    V = 0.615 * image[:, :, :, 0] - 0.51449 * \
        image[:, :, :, 1] - 0.10001 * image[:, :, :, 2]
    return tf.stack([Y, U, V], axis=-1)


def yuv2rgb_tf(image):
    R = image[:, :, :, 0] + 1.13983 * image[:, :, :, 2]
    G = image[:, :, :, 0] - 0.39465 * \
        image[:, :, :, 1] - 0.58060 * image[:, :, :, 2]
    B = image[:, :, :, 0] + 2.03211 * image[:, :, :, 1]
    return tf.stack([R, G, B], axis=-1)


def rgb2yuv_np(image):
    Y = 0.299 * image[:, :, :, 0] + 0.587 * \
        image[:, :, :, 1] + 0.114 * image[:, :, :, 2]
    U = - 0.14713 * image[:, :, :, 0] - 0.28886 * \
        image[:, :, :, 1] + 0.436 * image[:, :, :, 2]
    V = 0.615 * image[:, :, :, 0] - 0.51449 * \
        image[:, :, :, 1] - 0.10001 * image[:, :, :, 2]
    return np.stack([Y, U, V], axis=-1)


def yuv2rgb_np(image):
    R = image[:, :, :, 0] + 1.13983 * image[:, :, :, 2]
    G = image[:, :, :, 0] - 0.39465 * \
        image[:, :, :, 1] - 0.58060 * image[:, :, :, 2]
    B = image[:, :, :, 0] + 2.03211 * image[:, :, :, 1]
    return np.stack([R, G, B], axis=-1)


def read_pgm_file(item):
    # print(item)
    # print(item.dtype)
    data = cv2.imread(item.decode("utf-8"))
    return data.astype(np.float32)


def mean_square_error(a, b):
    diff = a - b
    diff = diff * diff
    # diff = np.sqrt(diff)
    diff = np.mean(diff)
    return diff


def to_image(image):
    image = (((image - image.min()) * 255) /
             (image.max() - image.min())).astype(np.uint8)
    return image


def rgb2ycbcr(image):
    Y = 0 + 0.299 * image[:, :, 0] + 0.587 * \
        image[:, :, 1] + 0.114 * image[:, :, 2]
    CB = 128.0 / 255 - 0.168736 * \
        image[:, :, 0] - 0.331264 * image[:, :, 1] + 0.5 * image[:, :, 2]
    CR = 128.0 / 255 + 0.5 * \
        image[:, :, 0] - 0.418688 * image[:, :, 1] - 0.081312 * image[:, :, 2]
    return np.stack([Y, CB, CR], axis=-1)


def rgb2yuv(image):
    Y = 0.299 * image[:, :, 0] + 0.587 * \
        image[:, :, 1] + 0.114 * image[:, :, 2]
    U = - 0.14713 * image[:, :, 0] - 0.28886 * \
        image[:, :, 1] + 0.436 * image[:, :, 2]
    V = 0.615 * image[:, :, 0] - 0.51449 * \
        image[:, :, 1] - 1.0001 * image[:, :, 2]
    return np.stack([Y, U, V], axis=-1)





def yuv_vs_rgb(a, b):
    rgb = mean_square_error(a, b)
    a_hsv = rgb2yuv(a)
    b_hsv = rgb2yuv(b)
    hsv = mean_square_error(a_hsv, b_hsv)
    return rgb, hsv


def ColorDistance(rgb1, rgb2):
    '''d = {} distance between two colors(3)'''
    rm = 0.5 * (rgb1[0] + rgb2[0])
    d = sum((2 + rm, 4, 3 - rm) * (rgb1 - rgb2)**2)**0.5
    return d





def compute_psnr(cover, stego):
    mse = mean_square_error(cover, stego)
    return 10 * np.log10(255.0 * 255.0 / mse)


def normalized_correlation(w1, w2):
    numerator = tf.reduce_sum(w1 * w2, axis=(1, 2, 3))
    denominator = tf.sqrt(tf.reduce_sum(tf.square(w1), axis=(
        1, 2, 3))) * tf.sqrt(tf.reduce_sum(tf.square(w2), axis=(1, 2, 3)))
    return tf.reduce_mean(numerator / denominator)


def compute_nc(secret, revealed, edgeinner):

    secret_image = (np.expand_dims(secret, 0) > 128).astype(np.uint8)
    secret_image = np.expand_dims(secret_image, -1)
    secret_image = np.concatenate([secret_image, secret_image], axis=0)

    secret_image2 = (np.expand_dims(revealed, 0) > 128).astype(np.uint8)
    if len(secret_image2.shape) < 4:
        secret_image2 = np.expand_dims(secret_image2, -1)
    secret_image2 = np.concatenate([secret_image2, secret_image2], axis=0)

    secret_tensor = tf.placeholder(
        shape=[None, edgeinner, edgeinner, 1], dtype=tf.int32, name="secret_tensor")
    cover_tensor = tf.placeholder(
        shape=[None, edgeinner, edgeinner, 1], dtype=tf.int32, name="cover_tensor")
    nc = normalized_correlation(
        tf.cast(secret_tensor, tf.float32), tf.cast(cover_tensor, tf.float32))

    with tf.Session() as sess:
        nc_val = sess.run(
            [nc], feed_dict={cover_tensor: secret_image, secret_tensor: secret_image2})
        return nc_val[0]


def compute_ssim(a, b):

    h = a.shape[0]
    w = a.shape[1]
    a = np.expand_dims(a, 0) / 255.0
    b = np.expand_dims(b, 0) / 255.0

    secret_true = tf.placeholder(shape=[1, h, w, 3], dtype=tf.float32)
    secret_pred = tf.placeholder(shape=[1, h, w, 3], dtype=tf.float32)

    ssim0 = tf.reduce_mean(
        tf.image.ssim(secret_true, secret_pred, max_val=1.0))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        s0 = sess.run(ssim0, feed_dict={secret_true: a, secret_pred: b})

        return s0


def compute_ber(a, b):
    if len(a.shape) > 3:
        n = a.shape[0] * a.shape[1] * a.shape[2]
    else:
        n = a.shape[0] * a.shape[1]
    a = (a > 128).astype(np.uint8)
    if len(a.shape) < 3:
        a = np.expand_dims(a, -1)
    if len(b.shape) < 3:
        b = np.expand_dims(b, -1)
    b = (b > 128).astype(np.uint8)

    diff = (a != b).astype(np.uint8)

    # diff = np.sqrt(diff)
    diff = np.sum(diff)
    ber = diff * 1.0 / n
    return ber




def attack_py_func(tensor, secret_tensor):

    mode = 'test'
    # print 'add noise using mode ', mode
    bs, cols, rows, _ = tensor.shape

#     tensor = (np.clip(tensor, 0, 1) * 255).astype(np.uint8)
#     secret_tensor = (secret_tensor[:, :, :, 0] * 255).astype(np.uint8)

    def scale(tensor, secret_tensor):
        scales = [0.8, 0.9, 1.0, 1.1, 1.2]
        index = np.random.randint(0, 5)
        matrix = cv2.getRotationMatrix2D(
            (cols / 2, rows / 2), angle=0, scale=scales[index])
        tensor = cv2.warpAffine(tensor, matrix, (cols, rows))
        secret_tensor = cv2.warpAffine(secret_tensor, matrix, (cols, rows))

        return tensor, secret_tensor

    def rotate(tensor, secret_tensor, angle=0.):
        if angle == 0.:
            angle = np.random.randint(0, 30)
        matrix = cv2.getRotationMatrix2D(
            (cols / 2, rows / 2), angle=angle, scale=1)
        tensor = cv2.warpAffine(tensor, matrix, (cols, rows))
        secret_tensor = cv2.warpAffine(secret_tensor, matrix, (cols, rows))

        return tensor, secret_tensor

    def translate(tensor, secret_tensor):
        offset = np.random.randint(0, 15)
        tform = transform.AffineTransform(
            scale=(1, 1), rotation=0, translation=(offset, 0))
        tensor = cv2.warpPerspective(tensor, tform.params, (cols, rows))
        secret_tensor = cv2.warpPerspective(
            secret_tensor, tform.params, (cols, rows))
        return tensor, secret_tensor

    def crop(tensor, secret_tensor, factor):
        edge1 = int(cols * math.sqrt(factor))
        edge2 = int(rows * math.sqrt(factor))
        seed1 = random.randint(0, cols - edge1)
        seed2 = random.randint(0, rows - edge2)
        tensor[:, seed1:seed1 + edge1, seed2:seed2 + edge2, :] = 0
        return tensor, secret_tensor

    def gaussian_blur(tensor, secret_tensor):
        tensor = cv2.GaussianBlur(tensor, (5, 5), sigmaX=1.0, sigmaY=1.0)
        return tensor, secret_tensor

    def median_blur(tensor, secret_tensor):
        for i in range(bs):
            tensor[i] = cv2.medianBlur(tensor[i], 3)
        return tensor, secret_tensor

    def mean_blur(tensor, secret_tensor):
        for i in range(bs):
            tensor[i] = cv2.blur(tensor[i], (3, 3))

        return tensor, secret_tensor

    def gaussian_noised(tensor, secret_tensor):
        mean = np.random.randint(0, 5)
        tensor = np.clip(
            tensor + np.random.normal(scale=mean, size=tensor.shape), 0, 255).astype(np.uint8)
        return tensor, secret_tensor

    def sp_noised(tensor, secret_tensor):
        p = np.random.uniform(0, 0.05)

        tensor = (skimage.util.random_noise(
            tensor, mode='s&p', amount=p) * 255).astype(np.uint8)
        # flipped = np.random.choice([True, False], size=tensor.shape,
        #                            p=[p, 1 - p])
        # salted = np.random.choice([True, False], size=tensor.shape,
        #                           p=[q, 1 - q])
        return tensor, secret_tensor

    def jpeg_encoding(tensor, secret_tensor):
        quality = np.random.randint(70, 95)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        result, encimg = cv2.imencode('.jpg', tensor, encode_param)
        decimg = cv2.imdecode(encimg, 1)
        return decimg, secret_tensor

    if mode == 'test':
        # stego_noise, secret_noise = rotate(tensor, secret_tensor, angle=math.radians(10))
        # stego_noise, secret_noise = rotate(tensor, secret_tensor, angle=math.radians(1))
        # stego_noise, secret_noise = translate(tensor, secret_tensor)
        #         stego_noise, secret_noise = median_blur(tensor, secret_tensor)
        stego_noise, secret_noise = crop(tensor, secret_tensor, 1 / 4.0)
#         stego_noise, secret_noise = mean_blur(tensor, secret_tensor)
        return stego_noise, secret_noise

    if mode == 'train':
        stego_out = np.zeros_like(tensor)
        secret_out = np.zeros_like(secret_tensor)
        for i in range(bs):

            stego_noise, secret_noise = tensor[i], secret_tensor[i]

            if np.random.random() > 0.5:
                index = np.random.randint(1, 4)
                if index == 1:
                    stego_noise, secret_noise = rotate(
                        stego_noise, secret_noise)
                elif index == 2:
                    stego_noise, secret_noise = translate(
                        stego_noise, secret_noise)
                else:
                    stego_noise, secret_noise = scale(
                        stego_noise, secret_noise)

            if np.random.random() > 0.5:
                index = np.random.randint(1, 5)
                if index == 1:
                    stego_noise, secret_noise = gaussian_noised(
                        stego_noise, secret_noise)
                elif index == 2:
                    stego_noise, secret_noise = sp_noised(
                        stego_noise, secret_noise)
            # if np.random.random() > 0.6:
            #     index = np.random.randint(1, 3)
                elif index == 3:
                    stego_noise, secret_noise = gaussian_blur(
                        stego_noise, secret_noise)
                elif index == 4:
                    stego_noise, secret_noise = jpeg_encoding(
                        stego_noise, secret_noise)

                       stego_out[i], secret_out[i] = stego_noise, secret_noise

        stego_out = stego_out.astype(np.float32) / 255.
        secret_out = (secret_out > 128).astype(np.float32)
        secret_out = np.expand_dims(secret_out, -1)
        return stego_out, secret_out

    if mode == 'None':
        return tensor, secret_tensor


def debug_py_func():
    secret_tensor = tf.placeholder(
        shape=[None, None, None, 1], dtype=tf.float32, name="secret_tensor")
    cover_tensor = tf.placeholder(
        shape=[None, None, None, 3], dtype=tf.float32, name="cover_tensor")
    stego_noise, secret_noise = tf.py_func(
        attack_py_func,
        [cover_tensor, secret_tensor],
        [tf.float32, tf.float32])

    image, secret_image = cv2.imread(
        'img/cover.png'), cv2.imread('img/watermark_demo.JPEG', 0)

    image = np.expand_dims(image, 0)
    image = image.astype(np.float32) / 255.
    image = np.concatenate([image, image], axis=0)

    secret_image = cv2.resize(secret_image, (512, 512))
    secret_image = (np.expand_dims(secret_image, 0) > 128).astype(np.uint8)
    secret_image = np.expand_dims(secret_image, -1)
    secret_image = np.concatenate([secret_image, secret_image], axis=0)

    # a, b  = attack_py_func(image, secret_image)

    with tf.Session() as sess:
        out_image, out_mask = sess.run([stego_noise, secret_noise], feed_dict={
                                       cover_tensor: image, secret_tensor: secret_image})
        for i in range(out_image.shape[0]):
            cv2.imwrite('img/%d_image.png' %
                        i, (image[i] * 255).astype(np.uint8))
            cv2.imwrite('img/%d_mask.png' %
                        i, (secret_image[i] * 255).squeeze().astype(np.uint8))

            cv2.imwrite('img/%d_image_noied.png' %
                        i, (out_image[i] * 255).astype(np.uint8))
            cv2.imwrite('img/%d_mask_noised.png' %
                        i, (out_mask[i] * 255).squeeze().astype(np.uint8))

        print out_image, out_mask


def random_transform(tensor, secret_tensor, std=0.01, mode='train'):

    # print 'add noise using mode ', mode

    cols, rows, _ = tensor.shape

    def scale(tensor, secret_tensor):
        scales = [0.8, 0.9, 1.0, 1.1, 1.2]
        index = np.random.randint(0, 5)
        matrix = cv2.getRotationMatrix2D(
            (cols / 2, rows / 2), angle=0, scale=scales[index])
        tensor = cv2.warpAffine(tensor, matrix, (cols, rows))
        secret_tensor = cv2.warpAffine(secret_tensor, matrix, (cols, rows))

        return tensor, secret_tensor

    def rotate(tensor, secret_tensor, angle=0.):
        if angle == 0.:
            angle = np.random.randint(0, 6)
        matrix = cv2.getRotationMatrix2D(
            (cols / 2, rows / 2), angle=angle, scale=1)
        tensor = cv2.warpAffine(tensor, matrix, (cols, rows))
        secret_tensor = cv2.warpAffine(secret_tensor, matrix, (cols, rows))

        return tensor, secret_tensor

    def translate(tensor, secret_tensor):

        offset = np.random.randint(0, 11)
        tform = transform.AffineTransform(
            scale=(1, 1), rotation=0, translation=(offset, 0))
        tensor = cv2.warpPerspective(tensor, tform.params, (cols, rows))
        secret_tensor = cv2.warpPerspective(
            secret_tensor, tform.params, (cols, rows))
        return tensor, secret_tensor

    if mode == 'test':
        # stego_noise, secret_noise = rotate(tensor, secret_tensor, angle=math.radians(10))
        # stego_noise, secret_noise = rotate(tensor, secret_tensor, angle=math.radians(1))
        # stego_noise, secret_noise = translate(tensor, secret_tensor)
        stego_noise, secret_noise = scale(tensor, secret_tensor)

        return stego_noise, secret_noise

    if mode == 'train':
        stego_noise = tensor
        secret_noise = secret_tensor
        stego_noise, secret_noise = rotate(stego_noise, secret_noise) if np.random.random(
        ) > 0.5 else (stego_noise, secret_noise)
        stego_noise, secret_noise = translate(
            stego_noise, secret_noise) if np.random.random() > 0.5 else (stego_noise, secret_noise)
        stego_noise, secret_noise = scale(stego_noise, secret_noise) if np.random.random(
        ) > 0.5 else (stego_noise, secret_noise)
        return stego_noise, secret_noise

    if mode == 'None':
        return tensor, secret_tensor



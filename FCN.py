import logging
import sys
import tensorflow as tf
import numpy as np
import math
import inspect
import os


learning_rate = 0.00001
dropped_rate = 0.4


class FCN16:
    def __init__(self, path):
        logging.debug('init')

        # if not path:
        #     path = inspect.getfile(FCN16)
        #     logging.debug('path = %s' % path)
        #     logging.debug('pardir = %s' % os.pardir)
        #     path = os.path.abspath(os.path.join(path, os.pardir, ))
        #     logging.debug('path = %s' % path)
        #     path = os.path.join(path, 'vgg16.npy')
        #     logging.debug(path)
        
        self.data_dict = np.load(path, encoding = 'latin1').item()
        self.wd = 5e-4

        logging.debug('npy file loaded')

    def build(self, rgb, y, is_train = False, n_class = 20, debug = False):
        # convert rgb to bgr why???
        with tf.name_scope('rgb2bgr'):
            # why does the nn have to use bgr instead of rgb ??
            # rgb shape = [batch, h, w, channel]
            # split the channel dimension (axis 3) into 3 parts
            r, g, b = tf.split(rgb, 3, 3)

            # why does the sample minus a mean value ???
            bgr = tf.concat([
                b,
                g,
                r
            ], 3)

            if debug:
                tf.Print(bgr, [tf.shape(bgr)],
                        message = 'shape of input image',
                        summarize = 4, first_n = 1)

        # to create the layers
        # self.conv1_1 = self.conv_layer(bgr, 'conv1_1')
        # self.conv1_2 = self.conv_layer(self.conv1_1, 'conv1_2')
        # self.pool1 = self.max_pool(self.conv1_2, 'pool1', debug)

        # self.conv2_1 = self.conv_layer(self.pool1, 'conv2_1')
        # self.conv2_2 = self.conv_layer(self.conv2_1, 'conv2_2')
        # self.pool2 = self.max_pool(self.conv2_2, 'pool2', debug)

        # self.conv3_1 = self.conv_layer(self.pool2, 'conv3_1')
        # self.conv3_2 = self.conv_layer(self.conv3_1, 'conv3_2')
        # self.conv3_3 = self.conv_layer(self.conv3_2, 'conv3_3')
        # self.pool3 = self.max_pool(self.conv3_3, 'pool3', debug)

        # self.conv4_1 = self.conv_layer(self.pool3, 'conv4_1')
        # self.conv4_2 = self.conv_layer(self.conv4_1, 'conv4_2')
        # self.conv4_3 = self.conv_layer(self.conv4_2, 'conv4_3')
        # self.pool4 = self.max_pool(self.conv4_3, 'pool4', debug)

        # self.conv5_1 = self.conv_layer(self.pool4, 'conv5_1')
        # self.conv5_2 = self.conv_layer(self.conv5_1, 'conv5_2')
        # self.conv5_3 = self.conv_layer(self.conv5_2, 'conv5_3')
        # # self.pool5 = self.max_pool(self.conv5_3, 'pool5', debug)
        # self.deconv5_4 = self.deconv_layer(self.conv5_3, shape = None,
        #                                     out_n_feature = self.conv5_3.shape[3].value / 2,
        #                                     name = 'deconv5_4', debug, stride = 2)

        conv_f_h = 3; conv_f_w = 3; conv_stride = 1
        pool_f_h = 2; pool_f_w = 2; pool_stride = 2

        # filter_shape = [f_h, f_w, in_ch, out_ch]
        # self.conv1_1 = self.conv_layer(bgr, [conv_f_h, conv_f_w, 3, 4], conv_stride, 'conv1_1')
        # self.conv1_2 = self.conv_layer(self.conv1_1, [conv_f_h, conv_f_w, 4, 8], conv_stride, 'conv1_2')
        # self.pool1 = self.max_pool(self.conv1_2, [1, pool_f_h, pool_f_w, 1], pool_stride, 'pool1', debug)

        # self.conv2_1 = self.conv_layer(self.pool1, [conv_f_h, conv_f_w, 8, 16], conv_stride, 'conv2_1')
        # self.conv2_2 = self.conv_layer(self.conv2_1, [conv_f_h, conv_f_w, 16, 32], conv_stride, 'conv2_2')
        # self.pool2 = self.max_pool(self.conv2_2, [1, pool_f_h, pool_f_w, 1], pool_stride, 'pool2', debug)

        # self.conv3_1 = self.conv_layer(self.pool2, [conv_f_h, conv_f_w, 32, 64], conv_stride, 'conv3_1')
        # self.conv3_2 = self.conv_layer(self.conv3_1, [conv_f_h, conv_f_w, 64, 128], conv_stride, 'conv3_2')
        # self.conv3_3 = self.conv_layer(self.conv3_2, [conv_f_h, conv_f_w, 128, 256], conv_stride, 'conv3_3')
        # self.pool3 = self.max_pool(self.conv3_3, [1, pool_f_h, pool_f_w, 1], pool_stride, 'pool3', debug)

        # self.conv4_1 = self.conv_layer(self.pool3, [conv_f_h, conv_f_w, 256, 512], conv_stride, 'conv4_1')
        # self.conv4_2 = self.conv_layer(self.conv4_1, [conv_f_h, conv_f_w, 512, 1024], conv_stride, 'conv4_2')
        # self.conv4_3 = self.conv_layer(self.conv4_2, [conv_f_h, conv_f_w, 1024, 2048], conv_stride, 'conv4_3')
        # self.pool4 = self.max_pool(self.conv4_3, [1, pool_f_h, pool_f_w, 1], pool_stride, 'pool4', debug)

        # self.conv5_1 = self.conv_layer(self.pool4, [conv_f_h, conv_f_w, 2048, 4096], conv_stride, 'conv5_1')
        # self.conv5_2 = self.conv_layer(self.conv5_1, [conv_f_h, conv_f_w, 4096, 8192], conv_stride, 'conv5_2')


        # self.conv1_1 = self.conv_layer(bgr, [conv_f_h, conv_f_w, 3, 4], conv_stride, 'conv1_1', debug)
        # self.pool1 = self.max_pool(self.conv1_1, [1, pool_f_h, pool_f_w, 1], pool_stride, 'pool1', debug)

        # self.conv2_1 = self.conv_layer(self.pool1, [conv_f_h, conv_f_w, 4, 8], conv_stride, 'conv2_1', debug)
        # self.pool2 = self.max_pool(self.conv2_1, [1, pool_f_h, pool_f_w, 1], pool_stride, 'pool2', debug)

        # self.conv3_1 = self.conv_layer(self.pool2, [conv_f_h, conv_f_w, 8, 16], conv_stride, 'conv3_1', debug)
        # self.conv3_2 = self.conv_layer(self.conv3_1, [conv_f_h, conv_f_w, 16, 32], conv_stride, 'conv3_2', debug)
        # self.pool3 = self.max_pool(self.conv3_2, [1, pool_f_h, pool_f_w, 1], pool_stride, 'pool3', debug)

        # # self.conv4_1 = self.conv_layer(self.pool3, [conv_f_h, conv_f_w, 32, 64], conv_stride, 'conv4_1', debug)
        # # self.conv4_2 = self.conv_layer(self.conv4_1, [conv_f_h, conv_f_w, 64, 128], conv_stride, 'conv4_2', debug)
        # # self.pool4 = self.max_pool(self.conv4_2, [1, pool_f_h, pool_f_w, 1], pool_stride, 'pool4', debug)

        # # self.conv5_1 = self.conv_layer(self.pool4, [conv_f_h, conv_f_w, 128, 256], conv_stride, 'conv5_1', debug)
        # # self.conv5_2 = self.conv_layer(self.conv5_1, [conv_f_h, conv_f_w, 256, 512], conv_stride, 'conv5_2', debug)
        
        # self.conv5_1 = self.conv_layer(self.pool3, [conv_f_h, conv_f_w, 32, 64], conv_stride, 'conv5_1', debug)
        # self.conv5_2 = self.conv_layer(self.conv5_1, [conv_f_h, conv_f_w, 64, 128], conv_stride, 'conv5_2', debug)

        # self.conv5_3 = self.conv_layer(self.conv5_2, [conv_f_h, conv_f_w, 8192, 16384], conv_stride, 'conv5_3')
        # # self.pool5 = self.max_pool(self.conv5_3, 'pool5', debug)
        # self.deconv5_4 = self.deconv_layer(self.conv5_3, shape = None,
        #                                     out_n_feature = self.conv5_3.shape[3].value,
        #                                     name = 'deconv5_4', debug, stride = 2)


        # VGG16 structure
        self.conv1_1 = self.conv_layer(bgr, None, conv_stride, 'conv1_1', debug)
        self.conv1_2 = self.conv_layer(self.conv1_1, None, conv_stride, 'conv1_2', debug)
        self.pool1 = self.max_pool(self.conv1_2, [1, pool_f_h, pool_f_w, 1], pool_stride, 'pool1', debug)

        self.conv2_1 = self.conv_layer(self.pool1, None, conv_stride, 'conv2_1', debug)
        self.conv2_2 = self.conv_layer(self.conv2_1, None, conv_stride, 'conv2_2', debug)
        self.pool2 = self.max_pool(self.conv2_2, [1, pool_f_h, pool_f_w, 1], pool_stride, 'pool2', debug)

        self.conv3_1 = self.conv_layer(self.pool2, None, conv_stride, 'conv3_1', debug)
        self.conv3_2 = self.conv_layer(self.conv3_1, None, conv_stride, 'conv3_2', debug)
        self.conv3_3 = self.conv_layer(self.conv3_2, None, conv_stride, 'conv3_3', debug)
        self.pool3 = self.max_pool(self.conv3_3, [1, pool_f_h, pool_f_w, 1], pool_stride, 'pool3', debug)

        self.conv4_1 = self.conv_layer(self.pool3, None, conv_stride, 'conv4_1', debug)
        self.conv4_2 = self.conv_layer(self.conv4_1, None, conv_stride, 'conv4_2', debug)
        self.conv4_3 = self.conv_layer(self.conv4_2, None, conv_stride, 'conv4_3', debug)
        self.pool4 = self.max_pool(self.conv4_3, [1, pool_f_h, pool_f_w, 1], pool_stride, 'pool4', debug)

        self.conv5_1 = self.conv_layer(self.pool4, None, conv_stride, 'conv5_1', debug)
        self.conv5_2 = self.conv_layer(self.conv5_1, None, conv_stride, 'conv5_2', debug)
        self.conv5_3 = self.conv_layer(self.conv5_2, None, conv_stride, 'conv5_3', debug)
        # get rid of the last max pool layer to connnect the deconv layer
        # self.pool5 = self.max_pool(self.conv5_3, [1, pool_f_h, pool_f_w, 1], pool_stride, 'pool5', debug)


        # self.deconv5_3 = self.deconv_layer(self.conv5_2, None, int(self.conv5_2.shape[3].value / 2), 'deconv5_3', debug)
        # self.deconv5_4 = self.deconv_layer(self.deconv5_3, None, self.deconv5_3.shape[3].value, 'deconv5_4', debug, stride = 2)

        # self.deconv6_1 = self.deconv_layer(self.deconv5_4, None, int(self.deconv5_4.shape[3].value / 2), 'deconv6_1', debug)
        # self.deconv6_2 = self.deconv_layer(self.deconv6_1, None, int(self.deconv6_1.shape[3].value / 2), 'deconv6_2', debug)
        # self.deconv6_3 = self.deconv_layer(self.deconv6_2, None, int(self.deconv6_2.shape[3].value / 2), 'deconv6_3', debug)
        # self.deconv6_4 = self.deconv_layer(self.deconv6_3, None, self.deconv6_3.shape[3].value, 'deconv6_4', debug, stride = 2)

        # self.deconv7_1 = self.deconv_layer(self.deconv6_4, None, int(self.deconv6_4.shape[3].value / 2), 'deconv7_1', debug)
        # self.deconv7_2 = self.deconv_layer(self.deconv7_1, None, int(self.deconv7_1.shape[3].value / 2), 'deconv7_2', debug)
        # self.deconv7_3 = self.deconv_layer(self.deconv7_2, None, int(self.deconv7_2.shape[3].value / 2), 'deconv7_3', debug)
        # self.deconv7_4 = self.deconv_layer(self.deconv7_3, None, self.deconv7_3.shape[3].value, 'deconv7_4', debug, stride = 2)

        # self.deconv8_1 = self.deconv_layer(self.deconv7_4, None, int(self.deconv7_4.shape[3].value / 2), 'deconv8_1', debug)
        # self.deconv8_2 = self.deconv_layer(self.deconv8_1, None, int(self.deconv8_1.shape[3].value / 2), 'deconv8_2', debug)
        # self.deconv8_3 = self.deconv_layer(self.deconv8_2, None, self.deconv8_2.shape[3].value, 'deconv8_3', debug, stride = 2)

        # self.deconv9_1 = self.deconv_layer(self.deconv8_3, None, self.deconv8_3.shape[3].value / 2, 'deconv9_1', debug)
        
        # self.deconv5_3 = self.deconv_layer(self.conv5_2, None, 256, 512, 'deconv5_3', debug)
        # self.deconv5_4 = self.deconv_layer(self.deconv5_3, None, 256, 256, 'deconv5_4', debug, stride = 2)

        # self.deconv6_1 = self.deconv_layer(self.deconv5_4, None, 128, 256, 'deconv6_1', debug)
        # self.deconv6_2 = self.deconv_layer(self.deconv6_1, None, 1024, 2048, 'deconv6_2', debug)
        # self.deconv6_3 = self.deconv_layer(self.deconv6_2, None, 512, 1024, 'deconv6_3', debug)
        # self.deconv6_4 = self.deconv_layer(self.deconv6_3, None, 512, 512, 'deconv6_4', debug, stride = 2)

        # self.deconv7_1 = self.deconv_layer(self.deconv6_4, None, 256, 512, 'deconv7_1', debug)
        # self.deconv7_2 = self.deconv_layer(self.deconv7_1, None, 128, 256, 'deconv7_2', debug)
        # self.deconv7_3 = self.deconv_layer(self.deconv7_2, None, 64, 128, 'deconv7_3', debug)
        # self.deconv7_4 = self.deconv_layer(self.deconv7_3, None, 64, 64, 'deconv7_4', debug, stride = 2)

        # self.deconv8_1 = self.deconv_layer(self.deconv7_4, None, 32, 64, 'deconv8_1', debug)
        # self.deconv8_2 = self.deconv_layer(self.deconv8_1, None, 16, 32, 'deconv8_2', debug)
        # self.deconv8_3 = self.deconv_layer(self.deconv8_2, None, 16, 16, 'deconv8_3', debug, stride = 2)

        # self.deconv9_1 = self.deconv_layer(self.deconv8_3, None, 1, 16, 'deconv9_1', debug)
        # self.conv9_2 = self.conv_layer(self.deconv9_1, [1, 1, 1, 1], conv_stride, 'conv9_2')
        
        # self.sig9 = tf.sigmoid(self.conv9_2, 'sig9')

        # self.deconv5_3 = self.deconv_layer(self.conv5_2, None, 512, 512, conv_f_h, conv_f_w, 'deconv5_3', debug, stride = 2)

        # self.deconv6_1 = self.deconv_layer(self.deconv5_3, None, 256, 512, conv_f_h, conv_f_w, 'deconv6_1', debug)
        # self.deconv6_2 = self.deconv_layer(self.deconv6_1, None, 128, 256, conv_f_h, conv_f_w, 'deconv6_2', debug)
        # self.deconv6_3 = self.deconv_layer(self.deconv6_2, None, 128, 128, conv_f_h, conv_f_w, 'deconv6_3', debug, stride = 2)

        # self.deconv7_1 = self.deconv_layer(self.deconv6_3, None, 64, 128, conv_f_h, conv_f_w, 'deconv7_1', debug)
        # self.deconv7_2 = self.deconv_layer(self.deconv7_1, None, 32, 64, conv_f_h, conv_f_w, 'deconv7_2', debug)
        # self.deconv7_3 = self.deconv_layer(self.deconv7_2, None, 32, 32, conv_f_h, conv_f_w, 'deconv7_3', debug, stride = 2)

        # self.deconv8_1 = self.deconv_layer(self.deconv7_3, None, 16, 32, conv_f_h, conv_f_w, 'deconv8_1', debug)
        # self.deconv8_2 = self.deconv_layer(self.deconv8_1, None, 16, 16, conv_f_h, conv_f_w, 'deconv8_2', debug, stride = 2)

        # self.deconv9_1 = self.deconv_layer(self.deconv8_2, None, 1, 16, conv_f_h, conv_f_w, 'deconv9_1', debug)
        # self.conv9_2 = self.conv_layer(self.deconv9_1, [1, 1, 1, 1], conv_stride, 'conv9_2', debug)
        
        # self.sig9 = tf.sigmoid(self.conv9_2, 'sig9')

        # self.deconv5_3 = self.deconv_layer(self.conv5_2, None, 128, 128, conv_f_h, conv_f_w, 'deconv5_3', debug, stride = 2)

        # self.deconv6_1 = self.deconv_layer(self.deconv5_3, None, 64, 128, conv_f_h, conv_f_w, 'deconv6_1', debug)
        # self.deconv6_2 = self.deconv_layer(self.deconv6_1, None, 32, 64, conv_f_h, conv_f_w, 'deconv6_2', debug)
        # self.deconv6_3 = self.deconv_layer(self.deconv6_2, None, 32, 32, conv_f_h, conv_f_w, 'deconv6_3', debug, stride = 2)

        # self.deconv7_1 = self.deconv_layer(self.deconv6_3, None, 16, 32, conv_f_h, conv_f_w, 'deconv7_1', debug)
        # self.deconv7_2 = self.deconv_layer(self.deconv7_1, None, 8, 16, conv_f_h, conv_f_w, 'deconv7_2', debug)
        # self.deconv7_3 = self.deconv_layer(self.deconv7_2, None, 8, 8, conv_f_h, conv_f_w, 'deconv7_3', debug, stride = 2)

        # # self.deconv8_1 = self.deconv_layer(self.deconv7_3, None, 4, 8, conv_f_h, conv_f_w, 'deconv8_1', debug)
        # # self.deconv8_2 = self.deconv_layer(self.deconv8_1, None, 2, 4, conv_f_h, conv_f_w, 'deconv8_2', debug, stride = 2)

        # self.deconv9_1 = self.deconv_layer(self.deconv7_3, None, 1, 8, conv_f_h, conv_f_w, 'deconv9_1', debug)
        # self.conv9_2 = self.conv_layer(self.deconv9_1, [1, 1, 1, 1], conv_stride, 'conv9_2', debug)
        
        # self.sig9 = tf.sigmoid(self.conv9_2, 'sig9')

        
        # deconv layers for VGG16 structure
        self.deconv5_4 = self.deconv_layer(self.conv5_3, None, FCN16.n_filters[0], 512, conv_f_h, conv_f_w, 'deconv5_4', debug, stride = 2, conv_shape = tf.shape(self.conv4_3))

        # self.deconv6_1 = self.deconv_layer(self.deconv5_4, None, FCN16.n_filters[0], FCN16.n_filters[0], conv_f_h, conv_f_w, 'deconv6_1', debug)
        # self.deconv6_2 = self.deconv_layer(self.deconv6_1, None, FCN16.n_filters[0], FCN16.n_filters[0], conv_f_h, conv_f_w, 'deconv6_2', debug)
        # self.deconv6_3 = self.deconv_layer(self.deconv6_2, None, FCN16.n_filters[0], FCN16.n_filters[0], conv_f_h, conv_f_w, 'deconv6_3', debug)
        # self.deconv6_4 = self.deconv_layer(self.deconv6_3, None, FCN16.n_filters[0], FCN16.n_filters[0], conv_f_h, conv_f_w, 'deconv6_4', debug, stride = 2)

        # self.deconv6_1 = self.deconv_layer(self.deconv5_4, None, FCN16.n_filters[0], FCN16.n_filters[0], conv_f_h, conv_f_w, 'deconv6_1', debug)
        # self.deconv6_2 = self.deconv_layer(self.deconv6_1, None, FCN16.n_filters[0], FCN16.n_filters[0], conv_f_h, conv_f_w, 'deconv6_2', debug)
        # self.deconv6_3 = self.deconv_layer(self.deconv6_2, None, FCN16.n_filters[1], FCN16.n_filters[0], conv_f_h, conv_f_w, 'deconv6_3', debug)
        # self.deconv6_4 = self.deconv_layer(self.deconv6_3, None, FCN16.n_filters[1], FCN16.n_filters[1], conv_f_h, conv_f_w, 'deconv6_4', debug, stride = 2, conv_shape = tf.shape(self.conv3_3))

        # self.deconv6_1 = self.deconv_layer(self.deconv5_4, None, FCN16.n_filters[0], FCN16.n_filters[0], conv_f_h, conv_f_w, 'deconv6_1', debug)
        # self.deconv6_2 = self.deconv_layer(self.deconv6_1, None, FCN16.n_filters[0], FCN16.n_filters[0], conv_f_h, conv_f_w, 'deconv6_2', debug)
        self.deconv6_3 = self.deconv_layer(self.deconv5_4, None, FCN16.n_filters[1], FCN16.n_filters[0], conv_f_h, conv_f_w, 'deconv6_3', debug)
        self.deconv6_4 = self.deconv_layer(self.deconv6_3, None, FCN16.n_filters[1], FCN16.n_filters[1], conv_f_h, conv_f_w, 'deconv6_4', debug, stride = 2, conv_shape = tf.shape(self.conv3_3))
        if is_train:
            self.deconv6_4 = tf.layers.dropout(self.deconv6_4, dropped_rate, name = 'dropout6')

        # self.deconv7_1 = self.deconv_layer(self.deconv6_4, None, FCN16.n_filters[1], FCN16.n_filters[1], conv_f_h, conv_f_w, 'deconv7_1', debug)
        # self.deconv7_2 = self.deconv_layer(self.deconv7_1, None, FCN16.n_filters[1], FCN16.n_filters[1], conv_f_h, conv_f_w, 'deconv7_2', debug)
        # self.deconv7_3 = self.deconv_layer(self.deconv7_2, None, FCN16.n_filters[2], FCN16.n_filters[1], conv_f_h, conv_f_w, 'deconv7_3', debug)
        # self.deconv7_4 = self.deconv_layer(self.deconv7_3, None, FCN16.n_filters[2], FCN16.n_filters[2], conv_f_h, conv_f_w, 'deconv7_4', debug, stride = 2, conv_shape = tf.shape(self.conv2_2))

        # self.deconv7_1 = self.deconv_layer(self.deconv6_4, None, FCN16.n_filters[1], FCN16.n_filters[1], conv_f_h, conv_f_w, 'deconv7_1', debug)
        # self.deconv7_2 = self.deconv_layer(self.deconv7_1, None, FCN16.n_filters[1], FCN16.n_filters[1], conv_f_h, conv_f_w, 'deconv7_2', debug)
        self.deconv7_3 = self.deconv_layer(self.deconv6_4, None, FCN16.n_filters[2], FCN16.n_filters[1], conv_f_h, conv_f_w, 'deconv7_3', debug)
        self.deconv7_4 = self.deconv_layer(self.deconv7_3, None, FCN16.n_filters[2], FCN16.n_filters[2], conv_f_h, conv_f_w, 'deconv7_4', debug, stride = 2, conv_shape = tf.shape(self.conv2_2))
        if is_train:
            self.deconv7_4 = tf.layers.dropout(self.deconv7_4, dropped_rate, name = 'dropout7')
        
        # self.deconv8_1 = self.deconv_layer(self.deconv7_4, None, FCN16.n_filters[2], FCN16.n_filters[2], conv_f_h, conv_f_w, 'deconv8_1', debug)
        # self.deconv8_2 = self.deconv_layer(self.deconv8_1, None, FCN16.n_filters[3], FCN16.n_filters[2], conv_f_h, conv_f_w, 'deconv8_2', debug)
        # self.deconv8_3 = self.deconv_layer(self.deconv8_2, None, FCN16.n_filters[3], FCN16.n_filters[3], conv_f_h, conv_f_w, 'deconv8_3', debug, stride = 2, conv_shape = tf.shape(self.conv1_2))

        # self.deconv8_1 = self.deconv_layer(self.deconv7_4, None, FCN16.n_filters[2], FCN16.n_filters[2], conv_f_h, conv_f_w, 'deconv8_1', debug)
        self.deconv8_2 = self.deconv_layer(self.deconv7_4, None, FCN16.n_filters[3], FCN16.n_filters[2], conv_f_h, conv_f_w, 'deconv8_2', debug)
        self.deconv8_3 = self.deconv_layer(self.deconv8_2, None, FCN16.n_filters[3], FCN16.n_filters[3], conv_f_h, conv_f_w, 'deconv8_3', debug, stride = 2, conv_shape = tf.shape(self.conv1_2))
        if is_train:
            self.deconv8_3 = tf.layers.dropout(self.deconv8_3, dropped_rate, name = 'dropout8')
        
        self.deconv9_1 = self.deconv_layer(self.deconv8_3, None, FCN16.n_filters[3], FCN16.n_filters[3], conv_f_h, conv_f_w, 'deconv9_1', debug)
        self.conv9_2 = self.conv_layer(self.deconv9_1, [1, 1, FCN16.n_filters[3], FCN16.n_filters[4]], conv_stride, 'conv9_2', True, activation = 'sigmoid')

        # self.deconv9_2 = self.deconv_layer(self.deconv9_1, None, 1, 64, conv_f_h, conv_f_w, 'deconv9_2', debug)
        # self.deconv10_3 = self.deconv_layer(self.deconv9_2, None, 1, 1, conv_f_h, conv_f_w, 'deconv10_3', debug, stride = 2)
        # self.sig10 = self.sigmoid(self.deconv10_3, 'sig10', debug)

        # self.sig10 = self.sig10 * 255.0
        # self.sig10 = tf.reshape(self.sig10)
        # self.conv9_2 = self.conv9_2 * 255.0

        # if debug:
        #     self.sig9 = tf.Print(self.sig9, [self.sig9],
        #                             message = 'value of sig9 = ')

        # save the final tensor
        # tf.add_to_collection('prediction_tensor', self.sig9)
        tf.add_to_collection('prediction_tensor', self.conv9_2)

        # set accuracy function
        with tf.name_scope('accuracy') as scope:
            # accuracy = tf.reduce_mean(tf.cast(tf.abs(tf.reshape(self.conv9_2, [-1, 1]) - y), tf.float32))
            #  accuracy = tf.reduce_mean(tf.cast(tf.abs(self.conv9_2 - y), tf.float32))
            
            # what is update_op???
            accuracy1, accuracy = tf.metrics.mean_squared_error(y, self.conv9_2)
            # accuracy, update_op = tf.metrics.mean_squared_error(y, self.conv9_2)

        tf.add_to_collection('accuracy', accuracy)

        if is_train:
            # set loss function
            # softmax_cross_entropy is not reasonable
            with tf.name_scope('loss') as scope:
                # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.conv9_2, labels = y))
                # loss = tf.losses.softmax_cross_entropy()
                loss = tf.losses.mean_squared_error(labels = y, predictions = self.conv9_2)
                # tf.summary.scalar('loss', loss)

                # how ??
                # batch_size = tf.shape(y)[0]
                # losses = []
                # for i in range(batch_size):
                #     loss = tf.losses.mean_squared_error(labels = y[i], predictions = self.conv9_2[i])
                #     losses.append(loss)
                
                # loss = tf.reduce_mean(tf.concat(losses, 0))
                # y_shape = tf.shape(y)
                # re_shape = [y_shape[0], y_shape[1], y_shape[2]]

                # loss = tf.losses.mean_squared_error(labels = tf.reshape(y, re_shape), predictions = tf.reshape(self.conv9_2, re_shape))
                
                tf.summary.scalar('loss', loss)

            # set optimizer
            # adam is a kind of gradient descent
            with tf.name_scope('optimizer') as scope:
                optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

            return optimizer, accuracy, loss
        else:
            return accuracy

    # def conv_layer(self, bottom, name):
    def conv_layer(self, bottom, filter_shape, stride, name, debug, activation = 'relu'):
        with tf.variable_scope(name) as scope:
            filter = self.get_conv_filter(filter_shape, name, debug)
            conv = tf.nn.conv2d(bottom, filter, [1, stride, stride, 1], padding = 'SAME')
            
            # bias = self.get_bias(name, filter_shape[3])
            bias = self.get_bias(name, debug, filter_shape)
            conv_bias = tf.nn.bias_add(conv, bias)
            
            if debug:
                conv_bias = self.print_tensor(conv_bias, name + '_conv_bias')
            
            if activation == 'relu':
                activation = tf.nn.relu(conv_bias)
            elif activation == 'sigmoid':
                activation = tf.sigmoid(conv_bias)
            else:
                logging.error('unexpected activation mode = %s in conv layer = %s' % (activation, name))
                exit()

            # why ??
            activation_summary(activation)

            if debug:
                # activation = tf.Print(activation, [tf.shape(activation)],
                #                 message = 'shape of %s = ' % name,
                #                 summarize = 4, first_n = 1)

                # activation = tf.Print(activation, [activation],
                #                     message = 'value of %s = ' % name)

                activation = self.print_tensor(activation, name)

        return activation

    def max_pool(self, bottom, filter_shape, stride, name, debug):
        pool = tf.nn.max_pool(bottom, ksize = filter_shape, strides = [1, stride, stride, 1],
                                padding = 'SAME', name = name)
        
        if debug:
            # pool = tf.Print(pool, [tf.shape(pool)],
            #                 message = 'shape of %s = ' % name,
            #                 summarize = 4, first_n = 1)

            # pool = tf.Print(pool, [pool],
            #                     message = 'value of %s = ' % name)

            pool = self.print_tensor(pool, name)

        return pool

    def deconv_layer(self, bottom, shape, out_n_feature, in_n_feature, o_f_h, o_f_w, name, debug, f_h = 3, f_w = 3, stride = 1, conv_shape = None):
        strides = [1, stride, stride, 1]
        # strides = [1, 1, 1, 1]
        # bot_shape = tf.shape(bottom)
        # deconv_stride = bot_shape[1] + 1 / bot_shape[1] - 1

        with tf.variable_scope(name):
            # in_n_feature = bottom.shape[3].value

            if shape is None:
                # Compute the shape out of bottom
                # bot_shape = bottom.shape
                bot_shape = tf.shape(bottom)

                # out = (i' - 1) * stride + k
                # temp
                if stride == 1:
                    # out_h = bot_shape[1]
                    # out_w = bot_shape[2]
                    # out_h = ((bot_shape[1] - 1) * stride) + o_f_h
                    # out_w = ((bot_shape[2] - 1) * stride) + o_f_w
                    out_h = bot_shape[1]
                    out_w = bot_shape[2]
                else:
                    # out_h = ((bot_shape[1] - 1) * stride) + 1
                    # out_w = ((bot_shape[2] - 1) * stride) + 1
                    # out_h = ((bot_shape[1] - 1) * stride) + o_f_h - 1
                    # out_w = ((bot_shape[2] - 1) * stride) + o_f_w - 1
                    out_h = bot_shape[1] * stride
                    out_w = bot_shape[2] * stride
                
                # out_h = ((bot_shape[1] - 1) * stride) + o_f_h
                # out_w = ((bot_shape[2] - 1) * stride) + o_f_w

                # out_n_feature <==> n_class
                new_shape = [bot_shape[0], out_h, out_w, out_n_feature]
            else:
                new_shape = [shape[0], shape[1], shape[2], out_n_feature]

            # for i in range(len(new_shape)):
            #     logging.debug(type(new_shape[i]))

            out_shape = tf.stack(new_shape)
            # out_shape = new_shape

            # logging.debug('layer: %s, in_n_feature = %d' % (name, in_n_feature))

            f_shape = [f_h, f_w, out_n_feature, in_n_feature]

            # # create, why ??
            # num_input = f_h * f_w * in_n_feature / stride
            # stddev = (2 / num_input) ** 0.5
            # logging.debug('stddev = %f' % stddev)

            weights = self.get_deconv_weights(f_shape, name, debug)
            deconv = tf.nn.conv2d_transpose(bottom, weights, out_shape,
                                            strides = strides, padding = 'SAME')

            # t0 = tf.constant(stride)
            # t1 = tf.constant(2)

            # if stride == 2:
            #     deconv = tf.slice(deconv, [0, 0, 0, 0], conv_shape)

            # deconv = tf.cond(stride == 2, lambda: tf.slice(deconv, [0, 0, 0, 0], conv_shape), lambda: deconv)
            # deconv = tf.cond(tf.equal(t0, t1), lambda: tf.slice(deconv, [0, 0, 0, 0], conv_shape), lambda: deconv)
            # tf.slice(deconv, [0, 0, 0, 0], conv_shape)

            # if stride == 1:
            #     # the same size as input
            #     pre_padding = o_f_h - 1
            # else:
            #     # twice
            #     pre_padding = 1

            # deconv = deconv[pre_padding: len(deconv) - pre_padding, pre_padding: len(deconv[0]) - pre_padding]

            # d_shape = tf.shape(deconv)
            # if stride == 1:
            #     deconv = tf.slice(deconv, [0, 1, 1, 0], [d_shape[0], d_shape[1] - 2, d_shape[2] - 2, d_shape[3]])
            # else:
            #     deconv = tf.slice(deconv, )

            # if stride == 1:
            #     deconv = tf.slice(deconv, [0, 1, 1, 0], [d_shape[0], d_shape[1] - 2, d_shape[2] - 2, d_shape[3]])

            if debug:
                # deconv = tf.Print(deconv, [tf.shape(deconv)],
                #                     message = 'shape of %s = ' % name,
                #                     summarize = 4, first_n = 1)
                
                # deconv = tf.Print(deconv, [deconv],
                #                     message = 'value of %s = ' % name)
            
                deconv = self.print_tensor(deconv, name)

        activation_summary(deconv)

        return deconv

    def get_conv_filter(self, f_shape, name, debug):
        # init = tf.constant_initializer(value = self.data_dict[name][0],
        #                                 dtype = tf.float32)
        # # test
        # # shape = self.data_dict[name][0].shape
        # shape = init.shape
        # logging.debug('layer name: %s' % name)
        # logging.debug('layer shape: %s' % str(shape))

        # var = tf.get_variable(name = 'filter', initializer = init, shape = shape)

        # # why ??
        # if not tf.get_variable_scope().resue:
        #     weight_decay = tf.multiply(tf.nn.l2_loss(var), self.wd,
        #                                 name = 'weight_loss')
        #     tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
        #                         weight_decay)
        
        # return var

        n = name + '_conv_filter'

        if f_shape:
            # h = f_shape[0]
            # w = f_shape[1]

            # f = math.ceil(h / 2.0)
            # c = (2 * f - 1 - f % 2) / (2.0 * f)
            # bilinear = np.zeros([h, w])
            # for y in range(h):
            #     for x in range(w):
            #         value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
            #         bilinear[x, y] = value
            
            # filter = np.zeros(f_shape)

            # shape_len = f_shape[2] if f_shape[2] < f_shape[3] else f_shape[3]
            # for i in range(shape_len):
            #     filter[:, :, i, i] = bilinear

            # init = tf.constant_initializer(value = filter, dtype = tf.float32)

            # return tf.get_variable(name = n, initializer = init, shape = filter.shape)

            # random initialization
            # dtype = tf.float32 and use tf.glorot_unifrom_initializer by default
            return tf.get_variable(name = n, shape = f_shape, dtype = tf.float32)
        else:
            init = tf.constant_initializer(value = self.data_dict[name][0], dtype = tf.float32)

            shape = self.data_dict[name][0].shape
            logging.debug('layer name = %s and shape = %s' % (name, str(shape)))

            var = tf.get_variable(name = n, shape = shape, initializer = init)

            if not tf.get_variable_scope().reuse:
                # weight decay useful ??
                weight_decay = tf.multiply(tf.nn.l2_loss(var), self.wd, name = name + '_weight_loss')
                
                tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, weight_decay)
            
            if debug:
                # weight_decay = tf.Print(weight_decay, [tf.shape(weight_decay)],
                #                         message = 'shape of %s = ' % name,
                #                         summarize = 4, first_n = 1)
                
                # weight_decay = tf.Print(weight_decay, [weight_decay]
                #                         message = 'value of %s = ' % name)
            
                var = self.print_tensor(var, n)
                weight_decay = self.print_tensor(weight_decay, name + '_weight_decay')
            
            return var


    # def get_bias(self, name, debug, n_class = None):
    def get_bias(self, name, debug, f_shape = None):
        n = name + '_bias'

        if name in self.data_dict:
            bias = self.data_dict[name][1]
            # test
            # shape = self.data_dict[name][1]
            shape = bias.shape

            # if name == 'fc8':

            init = tf.constant_initializer(value = bias, dtype = tf.float32)
            # init = tf.constant_initializer(value = self.data_dict[name][1],
            #                                 dtype = tf.float32)

            var = tf.get_variable(name = n, initializer = init, shape = shape)
        else:
            shape = [f_shape[3]]

            # init = tf.constant_initializer(value = np.zeros(shape), dtype = tf.float32)
        
            # var = tf.get_variable(name = n, initializer = init, shape = shape)
            var = tf.get_variable(name = n, shape = shape, dtype = tf.float32)

        if debug:
            var = self.print_tensor(var, n)
        
        return var

    def get_deconv_weights(self, f_shape, name, debug):
        n = name + '_deconv_weights'

        # initialize the weights ??
        
        h = f_shape[0]
        w = f_shape[1]

        f = math.ceil(h / 2.0)
        c = (2 * f - 1 - f % 2) / (2.0 * f)
        bilinear = np.zeros([h, w])
        for y in range(h):
            for x in range(w):
                value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
                bilinear[x, y] = value
        
        weights = np.zeros(f_shape)
        for i in range(f_shape[2]):
            weights[:, :, i, i] = bilinear

        init = tf.constant_initializer(value = weights, dtype = tf.float32)

        var = tf.get_variable(name = n, shape = weights.shape, initializer = init)

        if debug:
            var = self.print_tensor(var, n)

        return var

    def print_tensor(self, tensor, name):
        tensor = tf.Print(tensor, [tf.shape(tensor)],
                                message = 'shape of %s = ' % name,
                                        summarize = 4, first_n = 1)
                
        tensor = tf.Print(tensor, [tensor],
                                message = 'value of %s = ' % name)
        
        return tensor
    
    n_filters = [128, 64, 32, 8, 1]
    deconv_activation = 'relu'
    dropped_rate = 0.4

def activation_summary(x):
    name = x.op.name

    tf.summary.histogram(name + '/activations', x)
    tf.summary.scalar(name + '/sparsity', tf.nn.zero_fraction(x))
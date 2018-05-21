import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
import pdb
import utils_wgan

def LeakyReLU(x, alpha=0.3):
    return tf.maximum(alpha*x, x)

def LeakyReLU2(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

def Batchnorm(inputs, is_training, name=None, data_format='NHWC'):
    bn = tf.contrib.layers.batch_norm(inputs, 
                                      center=True, scale=True, 
                                      is_training=is_training,
                                      scope=name, 
                                      data_format=data_format)
    return bn

## Ref code: https://github.com/tensorflow/models/blob/master/slim/nets/resnet_v2.py
def ResBottleNeckBlock(x, n1, n2, n3, data_format, activation_fn=LeakyReLU):
    if n1 != n3:
        shortcut = slim.conv2d(x, n3, 1, 1, activation_fn=None, data_format=data_format)
    else:
        shortcut = x
    x = slim.conv2d(x, n2, 1, 1, activation_fn=activation_fn, data_format=data_format)
    x = slim.conv2d(x, n2, 3, 1, activation_fn=activation_fn, data_format=data_format)
    x = slim.conv2d(x, n3, 1, 1, activation_fn=None, data_format=data_format)
    out = activation_fn(shortcut + x)
    return out

def ResBlock(x, n1, n2, n3, data_format, activation_fn=LeakyReLU):
    if n1 != n3:
        shortcut = slim.conv2d(x, n3, 1, 1, activation_fn=None, data_format=data_format)
    else:
        shortcut = x
    x = slim.conv2d(x, n2, 3, 1, activation_fn=activation_fn, data_format=data_format)
    x = slim.conv2d(x, n3, 3, 1, activation_fn=None, data_format=data_format)
    out = activation_fn(shortcut + x)
    return out

def LuNet(x, input_H, input_W, is_train_tensor, data_format='NHWC', activation_fn=LeakyReLU, reuse=False):
    input_shape = x.get_shape().as_list()
    if len(input_shape) != 4:
        raise ValueError('Invalid input tensor rank, expected 4, was: %d' % len(input_shape))

    with tf.variable_scope("CNN",reuse=reuse) as vs:
        #
        x = slim.conv2d(x, 128, 7, 1, data_format=data_format, activation_fn=activation_fn)
        x = ResBottleNeckBlock(x, 128, 32, 128, data_format=data_format, activation_fn=activation_fn)
        x = tf.contrib.layers.max_pool2d(x, [3, 3], [2, 2], padding='SAME')
        input_H = input_H/2
        input_W = input_W/2
        #
        x = ResBottleNeckBlock(x, 128, 32, 128, data_format=data_format, activation_fn=activation_fn)
        x = ResBottleNeckBlock(x, 128, 32, 128, data_format=data_format, activation_fn=activation_fn)
        x = ResBottleNeckBlock(x, 128, 64, 256, data_format=data_format, activation_fn=activation_fn)
        x = tf.contrib.layers.max_pool2d(x, [3, 3], [2, 2], padding='SAME')
        input_H = input_H/2
        input_W = input_W/2
        #
        x = ResBottleNeckBlock(x, 256, 64, 256, data_format=data_format, activation_fn=activation_fn)
        x = ResBottleNeckBlock(x, 256, 64, 256, data_format=data_format, activation_fn=activation_fn)
        x = tf.contrib.layers.max_pool2d(x, [3, 3], [2, 2], padding='SAME')
        input_H = input_H/2
        input_W = input_W/2
        #
        x = ResBottleNeckBlock(x, 256, 64, 256, data_format=data_format, activation_fn=activation_fn)
        x = ResBottleNeckBlock(x, 256, 64, 256, data_format=data_format, activation_fn=activation_fn)
        x = ResBottleNeckBlock(x, 256, 128, 512, data_format=data_format, activation_fn=activation_fn)
        x = tf.contrib.layers.max_pool2d(x, [3, 3], [2, 2], padding='SAME')
        input_H = input_H/2
        input_W = input_W/2
        #
        x = ResBottleNeckBlock(x, 512, 128, 512, data_format=data_format, activation_fn=activation_fn)
        x = ResBottleNeckBlock(x, 512, 128, 512, data_format=data_format, activation_fn=activation_fn)
        x = tf.contrib.layers.max_pool2d(x, [3, 3], [2, 2], padding='SAME')
        input_H = input_H/2
        input_W = input_W/2
        #
        # x = slim.dropout(x, keep_prob=0.6)
        x = ResBlock(x, 512, 512, 128, data_format=data_format, activation_fn=activation_fn)
        x = tf.reshape(x, [-1, 128*input_H*input_W])
        print('dim:%d'%(128*input_H*input_W))
        x = slim.fully_connected(x, 512, activation_fn=None)
        x = Batchnorm(x, is_train_tensor, 'LuNet.BN', data_format='NHWC')
        # x = tf.nn.relu(x)
        x = activation_fn(x)
        # x = selu(x) ## Relpace BN+ReLU
        out = slim.fully_connected(x, 128, activation_fn=None)

    variables = tf.contrib.framework.get_variables(vs)
    return out, variables


################################################################
#######################       GAN       ########################
def GeneratorCNN_Pose_UAEAfterResidual(x, pose_target, input_channel, z_num, repeat_num, hidden_num, data_format, activation_fn=tf.nn.elu, min_fea_map_H=8, noise_dim=0, reuse=False):
    with tf.variable_scope("G") as vs:
        if pose_target is not None:
            if data_format == 'NCHW':
                x = tf.concat([x, pose_target], 1)
            elif data_format == 'NHWC':
                x = tf.concat([x, pose_target], 3)

        # Encoder
        encoder_layer_list = []
        x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=activation_fn, data_format=data_format)

        for idx in range(repeat_num):
            channel_num = hidden_num * (idx + 1)
            # channel_num = x.get_shape()[-1]
            res = x
            x = slim.conv2d(x, channel_num, 3, 1, activation_fn=activation_fn, data_format=data_format)
            x = slim.conv2d(x, channel_num, 3, 1, activation_fn=activation_fn, data_format=data_format)
            x = x + res
            encoder_layer_list.append(x)
            if idx < repeat_num - 1:
                x = slim.conv2d(x, hidden_num * (idx + 2), 3, 2, activation_fn=activation_fn, data_format=data_format)
                #x = tf.contrib.layers.max_pool2d(x, [2, 2], [2, 2], padding='VALID')

        x = tf.reshape(x, [-1, np.prod([min_fea_map_H, min_fea_map_H/2, channel_num])])
        z = x = slim.fully_connected(x, z_num, activation_fn=None)
        if noise_dim>0:
            noise = tf.random_uniform(
                (tf.shape(z)[0], noise_dim), minval=-1.0, maxval=1.0)
            z = tf.concat([z, noise], 1)

        # Decoder
        x = slim.fully_connected(z, np.prod([min_fea_map_H, min_fea_map_H/2, hidden_num]), activation_fn=None)
        x = reshape(x, min_fea_map_H, min_fea_map_H/2, hidden_num, data_format)
        
        for idx in range(repeat_num):
            # pdb.set_trace()
            x = tf.concat([x, encoder_layer_list[repeat_num-1-idx]], axis=-1)
            res = x
            # channel_num = hidden_num * (repeat_num-idx)
            channel_num = x.get_shape()[-1]
            x = slim.conv2d(x, channel_num, 3, 1, activation_fn=activation_fn, data_format=data_format)
            x = slim.conv2d(x, channel_num, 3, 1, activation_fn=activation_fn, data_format=data_format)
            x = x + res
            if idx < repeat_num - 1:
                # x = slim.layers.conv2d_transpose(x, hidden_num * (repeat_num-idx-1), 3, 2, activation_fn=activation_fn, data_format=data_format)
                x = upscale(x, 2, data_format)
                x = slim.conv2d(x, hidden_num * (repeat_num-idx-1), 1, 1, activation_fn=activation_fn, data_format=data_format)


        out = slim.conv2d(x, input_channel, 3, 1, activation_fn=None, data_format=data_format)

    variables = tf.contrib.framework.get_variables(vs)
    return out, z, variables

def UAE_noFC_After2Noise(x, input_channel, z_num, repeat_num, hidden_num, data_format, activation_fn=tf.nn.elu, noise_dim=64, reuse=False):
    with tf.variable_scope("G") as vs:
        # Encoder
        encoder_layer_list = []
        x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=activation_fn, data_format=data_format)

        prev_channel_num = hidden_num
        for idx in range(repeat_num):
            channel_num = hidden_num * (idx + 1)
            x = slim.conv2d(x, channel_num, 3, 1, activation_fn=activation_fn, data_format=data_format)
            x = slim.conv2d(x, channel_num, 3, 1, activation_fn=activation_fn, data_format=data_format)
            if idx > 0:
                encoder_layer_list.append(x)
            if idx < repeat_num - 1:
                x = slim.conv2d(x, channel_num, 3, 2, activation_fn=activation_fn, data_format=data_format)
                #x = tf.contrib.layers.max_pool2d(x, [2, 2], [2, 2], padding='VALID')
        
        if noise_dim>0:
            # pdb.set_trace()
            noise = tf.random_uniform(
                (tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], noise_dim), minval=-1.0, maxval=1.0)
            x = tf.concat([x, noise], -1)

        for idx in range(repeat_num):
            if idx < repeat_num - 1:
                x = tf.concat([x,encoder_layer_list[-1-idx]], axis=-1)
            x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=activation_fn, data_format=data_format)
            x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=activation_fn, data_format=data_format)
            if idx < repeat_num - 1:
                x = upscale(x, 2, data_format)

        out = slim.conv2d(x, input_channel, 3, 1, activation_fn=None, data_format=data_format)

    variables = tf.contrib.framework.get_variables(vs)
    return out, variables


def UAE_noFC_AfterNoise(x, input_channel, z_num, repeat_num, hidden_num, data_format, activation_fn=tf.nn.elu, noise_dim=64, reuse=False):
    with tf.variable_scope("G") as vs:
        # Encoder
        encoder_layer_list = []
        x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=activation_fn, data_format=data_format)

        prev_channel_num = hidden_num
        for idx in range(repeat_num):
            channel_num = hidden_num * (idx + 1)
            x = slim.conv2d(x, channel_num, 3, 1, activation_fn=activation_fn, data_format=data_format)
            x = slim.conv2d(x, channel_num, 3, 1, activation_fn=activation_fn, data_format=data_format)
            encoder_layer_list.append(x)
            if idx < repeat_num - 1:
                x = slim.conv2d(x, channel_num, 3, 2, activation_fn=activation_fn, data_format=data_format)
                #x = tf.contrib.layers.max_pool2d(x, [2, 2], [2, 2], padding='VALID')
        
        if noise_dim>0:
            # pdb.set_trace()
            noise = tf.random_uniform(
                (tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], noise_dim), minval=-1.0, maxval=1.0)
            x = tf.concat([x, noise], -1)

        for idx in range(repeat_num):
            x = tf.concat([x,encoder_layer_list[-1-idx]], axis=-1)
            x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=activation_fn, data_format=data_format)
            x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=activation_fn, data_format=data_format)
            if idx < repeat_num - 1:
                x = upscale(x, 2, data_format)

        out = slim.conv2d(x, input_channel, 3, 1, activation_fn=None, data_format=data_format)

    variables = tf.contrib.framework.get_variables(vs)
    return out, variables
    
def GeneratorCNN_Pose_UAEAfterResidual_UAEnoFCAfter2Noise(x, pose_target, input_channel, z_num, repeat_num, hidden_num, data_format, activation_fn=tf.nn.elu, noise_dim=64, reuse=False):
    with tf.variable_scope("Pose_AE") as vs1:
        out1, _, var1 = GeneratorCNN_Pose_UAEAfterResidual(x, pose_target, input_channel, z_num, repeat_num, hidden_num, data_format, activation_fn=activation_fn, noise_dim=0, reuse=False)
    with tf.variable_scope("UAEnoFC") as vs2:
        out2, var2 = UAE_noFC_After2Noise(tf.concat([out1,x],axis=-1), input_channel, z_num, repeat_num-2, hidden_num, data_format, noise_dim=noise_dim, activation_fn=activation_fn, reuse=False)
    return out1, out2, var1, var2


def int_shape(tensor):
    shape = tensor.get_shape().as_list()
    return [num if num is not None else -1 for num in shape]

def get_conv_shape(tensor, data_format):
    shape = int_shape(tensor)
    # always return [N, H, W, C]
    if data_format == 'NCHW':
        return [shape[0], shape[2], shape[3], shape[1]]
    elif data_format == 'NHWC':
        return shape

def nchw_to_nhwc(x):
    return tf.transpose(x, [0, 2, 3, 1])

def nhwc_to_nchw(x):
    return tf.transpose(x, [0, 3, 1, 2])

def reshape(x, h, w, c, data_format):
    if data_format == 'NCHW':
        x = tf.reshape(x, [-1, c, h, w])
    else:
        x = tf.reshape(x, [-1, h, w, c])
    return x

def resize_nearest_neighbor(x, new_size, data_format):
    if data_format == 'NCHW':
        x = nchw_to_nhwc(x)
        x = tf.image.resize_nearest_neighbor(x, new_size)
        x = nhwc_to_nchw(x)
    else:
        x = tf.image.resize_nearest_neighbor(x, new_size)
    return x

def upscale(x, scale, data_format):
    _, h, w, _ = get_conv_shape(x, data_format)
    return resize_nearest_neighbor(x, (h*scale, w*scale), data_format)
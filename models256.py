from models import *


def GeneratorCNN_Pose_UAEAfterResidual_256(x, pose_target, input_channel, z_num, repeat_num, hidden_num, data_format, activation_fn=tf.nn.elu, min_fea_map_H=8, noise_dim=0, reuse=False):
    with tf.variable_scope("G") as vs:
        if pose_target is not None:
            if data_format == 'NCHW':
                x = tf.concat([x, pose_target], 1)
            elif data_format == 'NHWC':
                x = tf.concat([x, pose_target], 3)

        # pdb.set_trace()
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

        x = tf.reshape(x, [-1, np.prod([min_fea_map_H, min_fea_map_H, channel_num])])
        z = x = slim.fully_connected(x, z_num, activation_fn=None)
        if noise_dim>0:
            noise = tf.random_uniform(
                (tf.shape(z)[0], noise_dim), minval=-1.0, maxval=1.0)
            z = tf.concat([z, noise], 1)

        # Decoder
        x = slim.fully_connected(z, np.prod([min_fea_map_H, min_fea_map_H, hidden_num]), activation_fn=None)
        x = reshape(x, min_fea_map_H, min_fea_map_H, hidden_num, data_format)
        
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


def GeneratorCNN_Pose_UAEAfterResidual_UAEnoFCAfterNoise_256(x, pose_target, input_channel, z_num, repeat_num, hidden_num, data_format, activation_fn=tf.nn.elu, noise_dim=64, reuse=False):
    with tf.variable_scope("Pose_AE") as vs1:
        out1, _, var1 = GeneratorCNN_Pose_UAEAfterResidual_256(x, pose_target, input_channel, z_num, repeat_num, hidden_num, data_format, activation_fn=activation_fn, noise_dim=0, reuse=False)
    with tf.variable_scope("UAEnoFC") as vs2:
        out2, var2 = UAE_noFC_AfterNoise(tf.concat([out1,x],axis=-1), input_channel, z_num, repeat_num-2, hidden_num, data_format, noise_dim=noise_dim, activation_fn=activation_fn, reuse=False)
    return out1, out2, var1, var2
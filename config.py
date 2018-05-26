#-*- coding: utf-8 -*-
import argparse

def str2bool(v):
    return v.lower() in ('true', '1')

arg_lists = []
parser = argparse.ArgumentParser()

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

# Network
net_arg = add_argument_group('Network')
# net_arg.add_argument('--input_scale_size', type=int, default=64,
#                      help='input image will be resized with the given value as width and height')
net_arg.add_argument('--img_H', type=int, default=128,
                     help='input image height')
net_arg.add_argument('--img_W', type=int, default=64,
                     help='input image width')
net_arg.add_argument('--conv_hidden_num', type=int, default=128,
                     choices=[64, 128],help='n in the paper')
# net_arg.add_argument('--z_num', type=int, default=64, choices=[64, 128])
net_arg.add_argument('--z_num', type=int, default=64)
# net_arg.add_argument('--noise_dim', type=int, default=10, choices=[10, 128])

# Data
data_arg = add_argument_group('Data')
data_arg.add_argument('--dataset', type=str, default='CelebA')
data_arg.add_argument('--split', type=str, default='train')
data_arg.add_argument('--batch_size', type=int, default=16)
data_arg.add_argument('--grayscale', type=str2bool, default=False)
data_arg.add_argument('--num_worker', type=int, default=4)

# Training / test parameters
train_arg = add_argument_group('Training')
train_arg.add_argument('--is_train', type=str2bool, default=True)
train_arg.add_argument('--test_one_by_one', type=str2bool, default=False)
train_arg.add_argument('--optimizer', type=str, default='adam')
train_arg.add_argument('--start_step', type=int, default=0)
data_arg.add_argument('--ckpt_path', type=str, default=None)
data_arg.add_argument('--pretrained_path', type=str, default=None)
train_arg.add_argument('--max_step', type=int, default=500000)
# train_arg.add_argument('--lr_update_step', type=int, default=100000, choices=[100000, 75000])
train_arg.add_argument('--lr_update_step', type=int, default=100000, choices=[50000, 100000])
train_arg.add_argument('--d_lr', type=float, default=0.00008)
train_arg.add_argument('--g_lr', type=float, default=0.00008)
train_arg.add_argument('--beta1', type=float, default=0.5)
train_arg.add_argument('--beta2', type=float, default=0.999)
train_arg.add_argument('--gamma', type=float, default=0.5)
train_arg.add_argument('--lambda_k', type=float, default=0.001)
train_arg.add_argument('--use_gpu', type=str2bool, default=True)
train_arg.add_argument('--gpu', type=int, default=-1)
train_arg.add_argument('--model', type=int, default=0)
train_arg.add_argument('--D_arch', type=str, default='DCGAN')  # 'DCGAN'  'noNormDCGAN'  'MultiplicativeDCGAN'  'tanhNonlinearDCGAN'  'resnet101'

# Misc
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--load_path', type=str, default='')
misc_arg.add_argument('--log_step', type=int, default=200)
misc_arg.add_argument('--save_model_secs', type=int, default=1000)
misc_arg.add_argument('--num_log_samples', type=int, default=3)
misc_arg.add_argument('--log_level', type=str, default='INFO', choices=['INFO', 'DEBUG', 'WARN'])
misc_arg.add_argument('--log_dir', type=str, default='logs')
misc_arg.add_argument('--model_dir', type=str, default=None)
misc_arg.add_argument('--data_dir', type=str, default='data')
misc_arg.add_argument('--test_data_path', type=str, default=None,
                      help='directory with images which will be used in test sample generation')
misc_arg.add_argument('--sample_per_image', type=int, default=64,
                      help='# of sample per image during test sample generation')
misc_arg.add_argument('--random_seed', type=int, default=123)

def get_config():
    config, unparsed = parser.parse_known_args()
    if config.use_gpu:
        data_format = 'NCHW'
    else:
        data_format = 'NHWC'
    setattr(config, 'data_format', data_format)
    return config, unparsed

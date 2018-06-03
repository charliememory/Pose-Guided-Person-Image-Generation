from __future__ import print_function

import os, pdb
import StringIO
import scipy.misc
import numpy as np
import glob
from tqdm import trange
from itertools import chain
from collections import deque
import pickle, shutil
from tqdm import tqdm

from tensorflow.python.ops import control_flow_ops

from models import *
from utils import save_image, _getPoseMask, _getSparsePose, _sparse2dense, _get_valid_peaks
import tflib as lib
from wgan_gp import *

def next(loader):
    return loader.next()[0].data.numpy()

def to_nhwc(image, data_format):
    if data_format == 'NCHW':
        new_image = nchw_to_nhwc(image)
    else:
        new_image = image
    return new_image

def to_nchw_numpy(image):
    if image.shape[3] in [1, 3]:
        new_image = image.transpose([0, 3, 1, 2])
    else:
        new_image = image
    return new_image

def norm_img(image, data_format=None):
    image = image/127.5 - 1.
    if data_format:
        image = to_nhwc(image, data_format)
    return image

def denorm_img(norm, data_format):
    return tf.clip_by_value(to_nhwc((norm + 1)*127.5, data_format), 0, 255)

def slerp(val, low, high):
    """Code from https://github.com/soumith/dcgan.torch/issues/14"""
    omega = np.arccos(np.clip(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)), -1, 1))
    so = np.sin(omega)
    if so == 0:
        return (1.0-val) * low + val * high # L'Hopital's rule/LERP
    return np.sin((1.0-val)*omega) / so * low + np.sin(val*omega) / so * high

from datasets import market1501, dataset_utils
import utils_wgan
from skimage.measure import compare_ssim as ssim
from skimage.color import rgb2gray
from PIL import Image
from tensorflow.python.ops import sparse_ops

class PG2(object):
    def _common_init(self, config):
        self.config = config
        self.data_loader = None
        self.dataset = config.dataset

        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.optimizer = config.optimizer
        self.batch_size = config.batch_size

        self.step = tf.Variable(config.start_step, name='step', trainable=False)

        self.g_lr = tf.Variable(config.g_lr, name='g_lr')
        self.d_lr = tf.Variable(config.d_lr, name='d_lr')

        self.g_lr_update = tf.assign(self.g_lr, self.g_lr * 0.5, name='g_lr_update')
        self.d_lr_update = tf.assign(self.d_lr, self.d_lr * 0.5, name='d_lr_update')

        self.gamma = config.gamma
        self.lambda_k = config.lambda_k

        self.z_num = config.z_num
        self.conv_hidden_num = config.conv_hidden_num
        self.img_H, self.img_W = config.img_H, config.img_W

        self.model_dir = config.model_dir
        self.load_path = config.load_path

        self.use_gpu = config.use_gpu
        self.data_format = config.data_format

        _, self.height, self.width, self.channel = self._get_conv_shape()
        self.repeat_num = int(np.log2(self.height)) - 2

        self.data_path = config.data_path
        self.pretrained_path = config.pretrained_path
        self.ckpt_path = config.ckpt_path
        self.start_step = config.start_step
        self.log_step = config.log_step
        self.max_step = config.max_step
        # self.save_model_secs = config.save_model_secs
        self.lr_update_step = config.lr_update_step

        self.is_train = config.is_train
        if self.is_train:
            self.num_threads = 4
            self.capacityCoff = 2
        else: # during testing to keep the order of the input data
            self.num_threads = 1
            self.capacityCoff = 1

    def __init__(self, config):
        # Trainer.__init__(self, config, data_loader=None)
        self._common_init(config)

        self.keypoint_num = 18
        self.D_arch = config.D_arch
        if 'market' in config.dataset.lower():
            if config.is_train:
                self.dataset_obj = market1501.get_split('train', config.data_path)
            else:
                self.dataset_obj = market1501.get_split('test', config.data_path)

        if config.test_one_by_one:
            self.x = tf.placeholder(tf.float32, shape=(None, self.img_H, self.img_W, 3))
            self.x_target = tf.placeholder(tf.float32, shape=(None, self.img_H, self.img_W, 3))
            self.pose = tf.placeholder(tf.float32, shape=(None, self.img_H, self.img_W, 18))
            self.pose_target = tf.placeholder(tf.float32, shape=(None, self.img_H, self.img_W, 18))
            self.mask = tf.placeholder(tf.float32, shape=(None, self.img_H, self.img_W, 1))
            self.mask_target = tf.placeholder(tf.float32, shape=(None, self.img_H, self.img_W, 1))
        else:
            self.x, self.x_target, self.pose, self.pose_target, self.mask, self.mask_target = self._load_batch_pair_pose(self.dataset_obj)

    def init_net(self):
        self.build_model()

        if self.pretrained_path is not None:
            var = tf.get_collection(tf.GraphKeys.VARIABLES, scope='Pose_AE')+tf.get_collection(tf.GraphKeys.VARIABLES, scope='UAEnoFC')
            self.saverPart = tf.train.Saver(var, max_to_keep=20)
            
        self.saver = tf.train.Saver(max_to_keep=20)
        self.summary_writer = tf.summary.FileWriter(self.model_dir)

        sv = tf.train.Supervisor(logdir=self.model_dir,
                                is_chief=True,
                                saver=None,
                                summary_op=None,
                                summary_writer=self.summary_writer,
                                global_step=self.step,
                                save_model_secs=0,
                                ready_for_local_init_op=None)

        gpu_options = tf.GPUOptions(allow_growth=True)
        sess_config = tf.ConfigProto(allow_soft_placement=True,
                                    gpu_options=gpu_options)
        self.sess = sv.prepare_or_wait_for_session(config=sess_config)
        # self.sess.run(tf.global_variables_initializer())
        if self.pretrained_path is not None:
            self.saverPart.restore(self.sess, self.pretrained_path)
            print('restored from pretrained_path:', self.pretrained_path)
        elif self.ckpt_path is not None:
            self.saver.restore(self.sess, self.ckpt_path)
            print('restored from ckpt_path:', self.ckpt_path)

    def _get_conv_shape(self):
        shape = [self.batch_size, self.img_H, self.img_W, 3]
        return shape

    def _getOptimizer(self, wgan_gp, gen_cost1, gen_cost2, disc_cost, G_var1, G_var2, D_var):
        clip_disc_weights = None
        if wgan_gp.MODE == 'wgan':
            gen_train_op1 = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(gen_cost1,
                                                 var_list=G_var1, colocate_gradients_with_ops=True)
            gen_train_op2 = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(gen_cost2,
                                                 var_list=G_var2, colocate_gradients_with_ops=True)
            disc_train_op = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(disc_cost,
                                                 var_list=D_var, colocate_gradients_with_ops=True)

            clip_ops = []
            for var in lib.params_with_name('Discriminator'):
                clip_bounds = [-.01, .01]
                clip_ops.append(tf.assign(var, tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])))
            clip_disc_weights = tf.group(*clip_ops)

        elif wgan_gp.MODE == 'wgan-gp':
            gen_train_op1 = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(gen_cost1,
                                              var_list=G_var1, colocate_gradients_with_ops=True)
            gen_train_op2 = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(gen_cost2,
                                              var_list=G_var2, colocate_gradients_with_ops=True)
            disc_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(disc_cost,
                                               var_list=D_var, colocate_gradients_with_ops=True)

        elif wgan_gp.MODE == 'dcgan':
            gen_train_op1 = tf.train.AdamOptimizer(learning_rate=2e-5, beta1=0.5).minimize(gen_cost1,
                                              var_list=G_var1, colocate_gradients_with_ops=True)
            gen_train_op2 = tf.train.AdamOptimizer(learning_rate=2e-5, beta1=0.5).minimize(gen_cost2,
                                              var_list=G_var2, colocate_gradients_with_ops=True)
            disc_train_op = tf.train.AdamOptimizer(learning_rate=2e-5, beta1=0.5).minimize(disc_cost,
                                               var_list=D_var, colocate_gradients_with_ops=True)

        elif wgan_gp.MODE == 'lsgan':
            gen_train_op1 = tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(gen_cost1,
                                                 var_list=G_var1, colocate_gradients_with_ops=True)
            gen_train_op2 = tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(gen_cost2,
                                                 var_list=G_var2, colocate_gradients_with_ops=True)
            disc_train_op = tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(disc_cost,
                                                  var_list=D_var, colocate_gradients_with_ops=True)
        else:
            raise Exception()
        return gen_train_op1, gen_train_op2, disc_train_op, clip_disc_weights

    def _getDiscriminator(self, wgan_gp, arch='DCGAN'):
        """
        Choose which generator and discriminator architecture to use by
        uncommenting one of these lines.
        """        
        if 'DCGAN'==arch:
            # Baseline (G: DCGAN, D: DCGAN)
            return wgan_gp.DCGANDiscriminator
        raise Exception('You must choose an architecture!')

    # def build_test_model(self):
    #     G1, DiffMap, self.G_var1, self.G_var2  = GeneratorCNN_Pose_UAEAfterResidual_UAEnoFCAfter2Noise(
    #             self.x, self.pose_target, 
    #             self.channel, self.z_num, self.repeat_num, self.conv_hidden_num, self.data_format, activation_fn=tf.nn.relu, noise_dim=0, reuse=False)

    #     G2 = G1 + DiffMap
    #     self.G1 = denorm_img(G1, self.data_format)
    #     self.G2 = denorm_img(G2, self.data_format)
    #     self.G = self.G2
    #     self.DiffMap = denorm_img(DiffMap, self.data_format)

    #     self.wgan_gp = WGAN_GP(DATA_DIR='', MODE='dcgan', DIM=64, BATCH_SIZE=self.batch_size, ITERS=200000, LAMBDA=10, G_OUTPUT_DIM=128*64*3)
    #     Dis = self._getDiscriminator(self.wgan_gp, arch=self.D_arch)

    def build_model(self):
        G1, DiffMap, self.G_var1, self.G_var2  = GeneratorCNN_Pose_UAEAfterResidual_UAEnoFCAfter2Noise(
                self.x, self.pose_target, 
                self.channel, self.z_num, self.repeat_num, self.conv_hidden_num, self.data_format, activation_fn=tf.nn.relu, noise_dim=0, reuse=False)

        G2 = G1 + DiffMap
        self.G1 = denorm_img(G1, self.data_format)
        self.G2 = denorm_img(G2, self.data_format)
        self.G = self.G2
        self.DiffMap = denorm_img(DiffMap, self.data_format)
        self.wgan_gp = WGAN_GP(DATA_DIR='', MODE='dcgan', DIM=64, BATCH_SIZE=self.batch_size, ITERS=200000, LAMBDA=10, G_OUTPUT_DIM=128*64*3)
        
        Dis = self._getDiscriminator(self.wgan_gp, arch=self.D_arch)
        triplet = tf.concat([self.x_target, self.x, G1, G2], 0)

        ## WGAN-GP code uses NCHW
        self.D_z = Dis(tf.transpose( triplet, [0,3,1,2] ), input_dim=3)
        self.D_var = lib.params_with_name('Discriminator.')

        D_z_pos_x_target, D_z_neg_x, D_z_neg_g1, D_z_neg_g2 = tf.split(self.D_z, 4)

        self.PoseMaskLoss1 = tf.reduce_mean(tf.abs(G1 - self.x_target) * (self.mask_target))
        self.g_loss1 = tf.reduce_mean(tf.abs(G1-self.x_target)) + self.PoseMaskLoss1

        self.g_loss2, self.d_loss, self.g2_g1_loss = self._gan_loss(self.wgan_gp, Dis, D_z_pos_x_target, D_z_neg_x, D_z_neg_g1, D_z_neg_g2, arch=self.D_arch)
        self.PoseMaskLoss2 = tf.reduce_mean(tf.abs(G2 - self.x_target) * (self.mask_target))
        self.L1Loss2 = tf.reduce_mean(tf.abs(G2 - self.x_target)) + self.PoseMaskLoss2
        self.g_loss2 += self.L1Loss2 * 10

        self.g_optim1, self.g_optim2, self.d_optim, self.clip_disc_weights = self._getOptimizer(self.wgan_gp, 
                                self.g_loss1, self.g_loss2, self.d_loss, self.G_var1,self.G_var2, self.D_var)
        self.summary_op = tf.summary.merge([
            tf.summary.image("G1", self.G1),
            tf.summary.image("G2", self.G2),
            tf.summary.image("DiffMap", self.DiffMap),
            tf.summary.scalar("loss/PoseMaskLoss1", self.PoseMaskLoss1),
            tf.summary.scalar("loss/PoseMaskLoss2", self.PoseMaskLoss2),
            tf.summary.scalar("loss/L1Loss2", self.L1Loss2),
            tf.summary.scalar("loss/g_loss1", self.g_loss1),
            tf.summary.scalar("loss/g_loss2", self.g_loss2),
            tf.summary.scalar("loss/d_loss", self.d_loss),
            tf.summary.scalar("loss/g2_g1_loss", self.g2_g1_loss),
            tf.summary.scalar("misc/d_lr", self.d_lr),
            tf.summary.scalar("misc/g_lr", self.g_lr),
        ])

    def _gan_loss(self, wgan_gp, Discriminator, disc_real, disc_fake_x, disc_fake_g1, disc_fake_g2, arch='DCGAN'):
        if wgan_gp.MODE == 'dcgan':
            if 'DCGAN'==arch:
                g2_g1_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake_g2-disc_fake_g1, labels=tf.ones_like(disc_fake_g2-disc_fake_g1)))
                
                gen_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake_g2, labels=tf.ones_like(disc_fake_g2)))
                # gen_cost = 0.5*tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake_g2, labels=tf.ones_like(disc_fake_g2))) \
                #             + 0.5*g2_g1_cost*10
                # gen_cost = g2_g1_cost
                            
                disc_cost = 0.25*tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake_x, labels=tf.zeros_like(disc_fake_x))) \
                            + 0.25*tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake_g2, labels=tf.zeros_like(disc_fake_g2)))
                disc_cost += 0.5*tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_real, labels=tf.ones_like(disc_real)))

        return gen_cost, disc_cost, g2_g1_cost

    def train(self):
        x_fixed, x_target_fixed, pose_fixed, pose_target_fixed, mask_fixed, mask_target_fixed = self.get_image_from_loader()
        save_image(x_fixed, '{}/x_fixed.png'.format(self.model_dir))
        save_image(x_target_fixed, '{}/x_target_fixed.png'.format(self.model_dir))
        save_image((np.amax(pose_fixed, axis=-1, keepdims=True)+1.0)*127.5, '{}/pose_fixed.png'.format(self.model_dir))
        save_image((np.amax(pose_target_fixed, axis=-1, keepdims=True)+1.0)*127.5, '{}/pose_target_fixed.png'.format(self.model_dir))
        save_image(mask_fixed, '{}/mask_fixed.png'.format(self.model_dir))
        save_image(mask_target_fixed, '{}/mask_target_fixed.png'.format(self.model_dir))

        for step in trange(self.start_step, self.max_step):
            if step < 22000:
                self.sess.run(self.g_optim1)
            else:
                # Train generator
                if step > 0:
                    self.sess.run(self.g_optim2)

                # Train critic
                if (self.wgan_gp.MODE == 'dcgan') or (self.wgan_gp.MODE == 'lsgan'):
                    disc_ITERS = 1
                else:
                    disc_ITERS = self.wgan_gp.CRITIC_ITERS
                for i in xrange(disc_ITERS):
                    self.sess.run(self.d_optim)
                    if self.wgan_gp.MODE == 'wgan':
                        self.sess.run(self.clip_disc_weights)

            fetch_dict = {}
            if step % self.log_step == self.log_step-1:
                fetch_dict.update({
                    "summary": self.summary_op
                })
                    # "k_t": self.k_t,
            result = self.sess.run(fetch_dict)

            if step % self.log_step == self.log_step-1:
                self.summary_writer.add_summary(result['summary'], step)
                self.summary_writer.flush()

            if step % (self.log_step * 3) == (self.log_step * 3)-1:
                # if self.data_format == 'NCHW':
                #     x = x_fixed.transpose([0, 3, 1, 2])
                # else:
                #     x = x_fixed
                x = utils_wgan.process_image(x_fixed, 127.5, 127.5)
                x_target = utils_wgan.process_image(x_target_fixed, 127.5, 127.5)
                self.generate(x, x_target, pose_target_fixed, self.model_dir, idx=step)

            if step % self.lr_update_step == self.lr_update_step - 1:
                self.sess.run([self.g_lr_update, self.d_lr_update])

            if step % (self.log_step * 30) == (self.log_step * 30)-1:
                self.saver.save(self.sess, os.path.join(self.model_dir, 'model.ckpt'), global_step=step)

    def test(self):
        test_result_dir = os.path.join(self.model_dir, 'test_result')
        test_result_dir_x = os.path.join(test_result_dir, 'x')
        test_result_dir_x_target = os.path.join(test_result_dir, 'x_target')
        test_result_dir_G = os.path.join(test_result_dir, 'G')
        test_result_dir_pose = os.path.join(test_result_dir, 'pose')
        test_result_dir_pose_target = os.path.join(test_result_dir, 'pose_target')
        test_result_dir_mask = os.path.join(test_result_dir, 'mask')
        test_result_dir_mask_target = os.path.join(test_result_dir, 'mask_target')
        if not os.path.exists(test_result_dir):
            os.makedirs(test_result_dir)
        if not os.path.exists(test_result_dir_x):
            os.makedirs(test_result_dir_x)
        if not os.path.exists(test_result_dir_x_target):
            os.makedirs(test_result_dir_x_target)
        if not os.path.exists(test_result_dir_G):
            os.makedirs(test_result_dir_G)
        if not os.path.exists(test_result_dir_pose):
            os.makedirs(test_result_dir_pose)
        if not os.path.exists(test_result_dir_pose_target):
            os.makedirs(test_result_dir_pose_target)
        if not os.path.exists(test_result_dir_mask):
            os.makedirs(test_result_dir_mask)
        if not os.path.exists(test_result_dir_mask_target):
            os.makedirs(test_result_dir_mask_target)

        for i in xrange(400):
            x_fixed, x_target_fixed, pose_fixed, pose_target_fixed, mask_fixed, mask_target_fixed = self.get_image_from_loader()
            x = utils_wgan.process_image(x_fixed, 127.5, 127.5)
            x_target = utils_wgan.process_image(x_target_fixed, 127.5, 127.5)
            if 0==i:
                x_fake = self.generate(x, x_target, pose_target_fixed, test_result_dir, idx=self.start_step, save=True)
            else:
                x_fake = self.generate(x, x_target, pose_target_fixed, test_result_dir, idx=self.start_step, save=False)
            p = (np.amax(pose_fixed, axis=-1, keepdims=False)+1.0)*127.5
            pt = (np.amax(pose_target_fixed, axis=-1, keepdims=False)+1.0)*127.5
            for j in xrange(self.batch_size):
                idx = i*self.batch_size+j
                im = Image.fromarray(x_fixed[j,:].astype(np.uint8))
                im.save('%s/%05d.png'%(test_result_dir_x, idx))
                im = Image.fromarray(x_target_fixed[j,:].astype(np.uint8))
                im.save('%s/%05d.png'%(test_result_dir_x_target, idx))
                im = Image.fromarray(x_fake[j,:].astype(np.uint8))
                im.save('%s/%05d.png'%(test_result_dir_G, idx))
                im = Image.fromarray(p[j,:].astype(np.uint8))
                im.save('%s/%05d.png'%(test_result_dir_pose, idx))
                im = Image.fromarray(pt[j,:].astype(np.uint8))
                im.save('%s/%05d.png'%(test_result_dir_pose_target, idx))
                im = Image.fromarray(mask_fixed[j,:].squeeze().astype(np.uint8))
                im.save('%s/%05d.png'%(test_result_dir_mask, idx))
                im = Image.fromarray(mask_target_fixed[j,:].squeeze().astype(np.uint8))
                im.save('%s/%05d.png'%(test_result_dir_mask_target, idx))
            if 0==i:
                save_image(x_fixed, '{}/x_fixed.png'.format(test_result_dir))
                save_image(x_target_fixed, '{}/x_target_fixed.png'.format(test_result_dir))
                save_image(mask_fixed, '{}/mask_fixed.png'.format(test_result_dir))
                save_image(mask_target_fixed, '{}/mask_target_fixed.png'.format(test_result_dir))
                save_image((np.amax(pose_fixed, axis=-1, keepdims=True)+1.0)*127.5, '{}/pose_fixed.png'.format(test_result_dir))
                save_image((np.amax(pose_target_fixed, axis=-1, keepdims=True)+1.0)*127.5, '{}/pose_target_fixed.png'.format(test_result_dir))

    def generate(self, x_fixed, x_target_fixed, pose_target_fixed, root_path=None, path=None, idx=None, save=True):
        G = self.sess.run(self.G, {self.x: x_fixed, self.pose_target: pose_target_fixed})
        ssim_G_x_list = []
        # x_0_255 = utils_wgan.unprocess_image(x_target_fixed, 127.5, 127.5)
        for i in xrange(G.shape[0]):
            # G_gray = rgb2gray((G[i,:]/127.5-1).clip(min=-1,max=1))
            # x_target_gray = rgb2gray((x_target_fixed[i,:]).clip(min=-1,max=1))
            G_gray = rgb2gray((G[i,:]).clip(min=0,max=255).astype(np.uint8))
            x_target_gray = rgb2gray(((x_target_fixed[i,:]+1)*127.5).clip(min=0,max=255).astype(np.uint8))
            ssim_G_x_list.append(ssim(G_gray, x_target_gray, data_range=x_target_gray.max() - x_target_gray.min(), multichannel=False))
        ssim_G_x_mean = np.mean(ssim_G_x_list)
        if path is None and save:
            path = os.path.join(root_path, '{}_G_ssim{}.png'.format(idx,ssim_G_x_mean))
            save_image(G, path)
            print("[*] Samples saved: {}".format(path))
        return G

    def _load_batch_pair_pose(self, dataset):
        data_provider = slim.dataset_data_provider.DatasetDataProvider(dataset, common_queue_capacity=32, common_queue_min=8)
        image_raw_0, image_raw_1, label, pose_0, pose_1, mask_0, mask_1  = data_provider.get([
            'image_raw_0', 'image_raw_1', 'label', 'pose_sparse_r4_0', 'pose_sparse_r4_1', 'pose_mask_r4_0', 'pose_mask_r4_1'])
        pose_0 = sparse_ops.sparse_tensor_to_dense(pose_0, default_value=0, validate_indices=False)
        pose_1 = sparse_ops.sparse_tensor_to_dense(pose_1, default_value=0, validate_indices=False)

        image_raw_0 = tf.reshape(image_raw_0, [128, 64, 3])        
        image_raw_1 = tf.reshape(image_raw_1, [128, 64, 3]) 
        pose_0 = tf.cast(tf.reshape(pose_0, [128, 64, self.keypoint_num]), tf.float32)
        pose_1 = tf.cast(tf.reshape(pose_1, [128, 64, self.keypoint_num]), tf.float32)
        mask_0 = tf.cast(tf.reshape(mask_0, [128, 64, 1]), tf.float32)
        mask_1 = tf.cast(tf.reshape(mask_1, [128, 64, 1]), tf.float32)

        images_0, images_1, poses_0, poses_1, masks_0, masks_1 = tf.train.batch([image_raw_0, image_raw_1, pose_0, pose_1, mask_0, mask_1], 
                    batch_size=self.batch_size, num_threads=self.num_threads, capacity=self.capacityCoff * self.batch_size)

        images_0 = utils_wgan.process_image(tf.to_float(images_0), 127.5, 127.5)
        images_1 = utils_wgan.process_image(tf.to_float(images_1), 127.5, 127.5)
        poses_0 = poses_0*2-1
        poses_1 = poses_1*2-1
        return images_0, images_1, poses_0, poses_1, masks_0, masks_1

    def get_image_from_loader(self):
        x, x_target, pose, pose_target, mask, mask_target = self.sess.run([self.x, self.x_target, self.pose, self.pose_target, self.mask, self.mask_target])
        x = utils_wgan.unprocess_image(x, 127.5, 127.5)
        x_target = utils_wgan.unprocess_image(x_target, 127.5, 127.5)
        mask = mask*255
        mask_target = mask_target*255
        return x, x_target, pose, pose_target, mask, mask_target
from trainer import *
from models256 import *
from datasets import deepfashion

class PG2_256(PG2):
    def __init__(self, config):
        self._common_init(config)
        self.keypoint_num = 18
        self.D_arch = config.D_arch

        if ('deepfashion' in config.dataset.lower()) or ('df' in config.dataset.lower()):
            if config.is_train:
                self.dataset_obj = deepfashion.get_split('train', config.data_path, data_name='DeepFashion')
            else:
                self.dataset_obj = deepfashion.get_split('test', config.data_path, data_name='DeepFashion')

        self.x, self.x_target, self.pose, self.pose_target, self.mask, self.mask_target = self._load_batch_pair_pose(self.dataset_obj)

    def _getDiscriminator(self, wgan_gp, arch='DCGAN'):
        """
        Choose which generator and discriminator architecture to use by
        uncommenting one of these lines.
        """        
        if 'DCGAN'==arch:
            # Baseline (G: DCGAN, D: DCGAN)
            return wgan_gp.DCGANDiscriminator_256
        raise Exception('You must choose an architecture!')

    def _gan_loss(self, wgan_gp, Discriminator, disc_real, disc_fake, arch='DCGAN'):
        if wgan_gp.MODE == 'dcgan':
            if 'DCGAN'==arch:
                gen_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake,
                                                                                  labels=tf.ones_like(disc_fake)))
                disc_cost =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake,
                                                                                    labels=tf.zeros_like(disc_fake)))
                disc_cost += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_real,
                                                                                    labels=tf.ones_like(disc_real)))                    
                disc_cost /= 2.
        else:
            raise Exception()
        return gen_cost, disc_cost

    def build_model(self):
        G1, DiffMap, self.G_var1, self.G_var2  = GeneratorCNN_Pose_UAEAfterResidual_UAEnoFCAfterNoise_256(
                self.x, self.pose_target, 
                self.channel, self.z_num, self.repeat_num, self.conv_hidden_num, self.data_format, activation_fn=tf.nn.relu, noise_dim=0, reuse=False)
        G2 = G1 + DiffMap
        self.G1 = denorm_img(G1, self.data_format)
        self.G2 = denorm_img(G2, self.data_format)
        self.G = self.G2
        self.DiffMap = denorm_img(DiffMap, self.data_format)
        self.wgan_gp = WGAN_GP(DATA_DIR='', MODE='dcgan', DIM=64, BATCH_SIZE=self.batch_size, ITERS=200000, LAMBDA=10, G_OUTPUT_DIM=256*256*3)
        Dis = self._getDiscriminator(self.wgan_gp, arch=self.D_arch)

        triplet = tf.concat([self.x_target, G2, self.x], 0)

        ## WGAN-GP code uses NCHW
        self.D_z = Dis(tf.transpose( triplet, [0,3,1,2] ), input_dim=3)
        self.D_var = lib.params_with_name('Discriminator.')

        D_z_pos_x_target, D_z_neg_g2, D_z_neg_x = tf.split(self.D_z, 3)
        D_z_pos = D_z_pos_x_target
        D_z_neg = tf.concat([D_z_neg_g2, D_z_neg_x], 0)


        self.PoseMaskLoss1 = tf.reduce_mean(tf.abs(G1 - self.x_target) * (self.mask_target))
        self.g_loss1 = tf.reduce_mean(tf.abs(G1-self.x_target)) + self.PoseMaskLoss1

        self.g_loss2, self.d_loss = self._gan_loss(self.wgan_gp, Dis, D_z_pos, D_z_neg, arch=self.D_arch)
        self.PoseMaskLoss2 = tf.reduce_mean(tf.abs(G2 - self.x_target) * (self.mask_target))
        self.L1Loss2 = tf.reduce_mean(tf.abs(G2 - self.x_target)) + self.PoseMaskLoss2
        self.g_loss2 += self.L1Loss2 * 50

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
            tf.summary.scalar("misc/d_lr", self.d_lr),
            tf.summary.scalar("misc/g_lr", self.g_lr),
        ])

    def _load_batch_pair_pose(self, dataset, mode='coordSolid'):
        data_provider = slim.dataset_data_provider.DatasetDataProvider(dataset, common_queue_capacity=32, common_queue_min=8)

        image_raw_0, image_raw_1, label, pose_0, pose_1, mask_0, mask_1 = data_provider.get([
            'image_raw_0', 'image_raw_1', 'label', 'pose_sparse_r4_0', 'pose_sparse_r4_1', 'pose_mask_r4_0', 'pose_mask_r4_1'])

        pose_0 = sparse_ops.sparse_tensor_to_dense(pose_0, default_value=0, validate_indices=False)
        pose_1 = sparse_ops.sparse_tensor_to_dense(pose_1, default_value=0, validate_indices=False)

        image_raw_0 = tf.reshape(image_raw_0, [256, 256, 3])        
        image_raw_1 = tf.reshape(image_raw_1, [256, 256, 3]) 
        pose_0 = tf.cast(tf.reshape(pose_0, [256, 256, self.keypoint_num]), tf.float32)
        pose_1 = tf.cast(tf.reshape(pose_1, [256, 256, self.keypoint_num]), tf.float32)
        mask_0 = tf.cast(tf.reshape(mask_0, [256, 256, 1]), tf.float32)
        mask_1 = tf.cast(tf.reshape(mask_1, [256, 256, 1]), tf.float32)

        images_0, images_1, poses_0, poses_1, masks_0, masks_1 = tf.train.batch([image_raw_0, image_raw_1, pose_0, pose_1, mask_0, mask_1], 
                    batch_size=self.batch_size, num_threads=self.num_threads, capacity=self.capacityCoff * self.batch_size)

        images_0 = utils_wgan.process_image(tf.to_float(images_0), 127.5, 127.5)
        images_1 = utils_wgan.process_image(tf.to_float(images_1), 127.5, 127.5)
        poses_0 = poses_0*2-1
        poses_1 = poses_1*2-1
        return images_0, images_1, poses_0, poses_1, masks_0, masks_1



r"""Downloads and converts Market1501 data to TFRecords of TF-Example protos.

This module downloads the Market1501 data, uncompresses it, reads the files
that make up the Market1501 data and creates two TFRecord datasets: one for train
and one for test. Each TFRecord dataset is comprised of a set of TF-Example
protocol buffers, each of which contain a single image and label.

The script should take about a minute to run.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys

import tensorflow as tf

try:
    import dataset_utils
except:
    from datasets import dataset_utils
import numpy as np
import pickle
import pdb
import glob

# The URL where the Market1501 data can be downloaded.
# _DATA_URL = 'xxxxx'

# The number of images in the validation set.
# _NUM_VALIDATION = 350

# Seed for repeatability.
_RANDOM_SEED = 0
random.seed(_RANDOM_SEED)

# The number of shards per dataset split.
_NUM_SHARDS = 1

_IMG_PATTERN = '.jpg'


class ImageReader(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

    def read_image_dims(self, sess, image_data):
        image = self.decode_jpeg(sess, image_data)
        return image.shape[0], image.shape[1]

    def decode_jpeg(self, sess, image_data):
        image = sess.run(self._decode_jpeg,
                                         feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image


def _get_folder_path(dataset_dir, split_name):
    if split_name == 'train':
        folder_path = os.path.join(dataset_dir, 'bounding_box_train')
    elif split_name == 'train_flip':
        folder_path = os.path.join(dataset_dir, 'bounding_box_train_flip')
    elif split_name == 'test':
        folder_path = os.path.join(dataset_dir, 'bounding_box_test')
    elif split_name == 'test_samples':
        folder_path = os.path.join(dataset_dir, 'bounding_box_test_samples')
    elif split_name == 'all':
        folder_path = os.path.join(dataset_dir, 'bounding_box_all')
    elif split_name == 'query':
        folder_path = os.path.join(dataset_dir, 'query')
    assert os.path.isdir(folder_path)
    return folder_path


def _get_image_file_list(dataset_dir, split_name):
    folder_path = _get_folder_path(dataset_dir, split_name)
    if split_name == 'train' or split_name == 'train_flip' or split_name == 'test_samples' or split_name == 'query' or split_name == 'all':
        filelist = sorted(os.listdir(folder_path))
        # filelist = glob.glob(os.path.join(folder_path, _IMG_PATTERN)) # glob will return full path
        # pdb.set_trace()
        filelist = sorted(filelist)
    elif split_name == 'test':
        filelist = sorted(os.listdir(folder_path))[6617:]  # before 6617 are junk detections
        # filelist = glob.glob(os.path.join(folder_path, _IMG_PATTERN))
        # filelist = sorted(filelist)[6617:]
    elif split_name == 'test_clean':
        filelist = sorted(os.listdir(folder_path))  # before 6617 are junk detections

    # Remove non-jpg files
    valid_filelist = []
    for i in xrange(0, len(filelist)):
        if filelist[i].endswith('.jpg') or filelist[i].endswith('.png'):
            valid_filelist.append(filelist[i])

    return valid_filelist


def _get_dataset_filename(dataset_dir, out_dir, split_name, shard_id):
    output_filename = 'Market1501_%s_%05d-of-%05d.tfrecord' % (
            split_name.split('_')[0], shard_id, _NUM_SHARDS)
    return os.path.join(out_dir, output_filename)


def _get_train_all_pn_pairs(dataset_dir, out_dir, split_name='train', augment_ratio=1, mode='diff_cam',add_switch_pair=True):
    """Returns a list of pair image filenames.

    Args:
        dataset_dir: A directory containing person images.

    Returns:
        p_pairs: A list of positive pairs.
        n_pairs: A list of negative pairs.
    """
    assert split_name in {'train', 'train_flip', 'test', 'test_samples', 'all'}
    if split_name=='train_flip':
        p_pairs_path = os.path.join(out_dir, 'p_pairs_train_flip.p')
        n_pairs_path = os.path.join(out_dir, 'n_pairs_train_flip.p')
    else:
        p_pairs_path = os.path.join(out_dir, 'p_pairs_'+split_name.split('_')[0]+'.p')
        n_pairs_path = os.path.join(out_dir, 'n_pairs_'+split_name.split('_')[0]+'.p')
    if os.path.exists(p_pairs_path):
        with open(p_pairs_path,'r') as f:
            p_pairs = pickle.load(f)
        with open(n_pairs_path,'r') as f:
            n_pairs = pickle.load(f)
    else:
        filelist = _get_image_file_list(dataset_dir, split_name)
        filenames = []
        p_pairs = []
        n_pairs = []
        if 'diff_cam'==mode:
            for i in xrange(0, len(filelist)):
                id_i = filelist[i][0:4]
                cam_i = filelist[i][6]
                for j in xrange(i+1, len(filelist)):
                    id_j = filelist[j][0:4]
                    cam_j = filelist[j][6]
                    if id_j == id_i and cam_j != cam_i:
                        p_pairs.append([filelist[i],filelist[j]])
                        # p_pairs.append([filelist[j],filelist[i]])  # two streams share the same weights, no need switch
                        if len(p_pairs)%100000==0:
                                print(len(p_pairs))
                    elif j%10==0 and id_j != id_i and cam_j != cam_i:  # limit the neg pairs to 1/10, otherwise it cost too much time
                        n_pairs.append([filelist[i],filelist[j]])
                        # n_pairs.append([filelist[j],filelist[i]])  # two streams share the same weights, no need switch
                        if len(n_pairs)%100000==0:
                                print(len(n_pairs))
        elif 'same_cam'==mode:
            for i in xrange(0, len(filelist)):
                id_i = filelist[i][0:4]
                cam_i = filelist[i][6]
                for j in xrange(i+1, len(filelist)):
                    id_j = filelist[j][0:4]
                    cam_j = filelist[j][6]
                    if id_j == id_i and cam_j == cam_i:
                        p_pairs.append([filelist[i],filelist[j]])
                        # p_pairs.append([filelist[j],filelist[i]])  # two streams share the same weights, no need switch
                        if len(p_pairs)%100000==0:
                                print(len(p_pairs))
                    elif j%10==0 and id_j != id_i and cam_j == cam_i:  # limit the neg pairs to 1/10, otherwise it cost too much time
                        n_pairs.append([filelist[i],filelist[j]])
                        # n_pairs.append([filelist[j],filelist[i]])  # two streams share the same weights, no need switch
                        if len(n_pairs)%100000==0:
                                print(len(n_pairs))
        elif 'same_diff_cam'==mode:
            for i in xrange(0, len(filelist)):
                id_i = filelist[i][0:4]
                cam_i = filelist[i][6]
                for j in xrange(i+1, len(filelist)):
                    id_j = filelist[j][0:4]
                    cam_j = filelist[j][6]
                    if id_j == id_i:
                        p_pairs.append([filelist[i],filelist[j]])
                        if add_switch_pair:
                            p_pairs.append([filelist[j],filelist[i]])  # if two streams share the same weights, no need switch
                        if len(p_pairs)%100000==0:
                                print(len(p_pairs))
                    elif j%2000==0 and id_j != id_i:  # limit the neg pairs to 1/40, otherwise it cost too much time
                        n_pairs.append([filelist[i],filelist[j]])
                        # n_pairs.append([filelist[j],filelist[i]])  # two streams share the same weights, no need switch
                        if len(n_pairs)%100000==0:
                                print(len(n_pairs))

        print('repeat positive pairs augment_ratio times and cut down negative pairs to balance data ......')
        p_pairs = p_pairs * augment_ratio  
        random.shuffle(n_pairs)
        n_pairs = n_pairs[:len(p_pairs)]
        print('p_pairs length:%d' % len(p_pairs))
        print('n_pairs length:%d' % len(n_pairs))
        print('save p_pairs and n_pairs ......')
        with open(p_pairs_path,'w') as f:
            pickle.dump(p_pairs,f)
        with open(n_pairs_path,'w') as f:
            pickle.dump(n_pairs,f)

    print('_get_train_all_pn_pairs finish ......')
    print('p_pairs length:%d' % len(p_pairs))
    print('n_pairs length:%d' % len(n_pairs))

    print('save pn_pairs_num ......')
    pn_pairs_num = len(p_pairs) + len(n_pairs)
    if split_name=='train_flip':
        fpath = os.path.join(out_dir, 'pn_pairs_num_train_flip.p')
    else:
        fpath = os.path.join(out_dir, 'pn_pairs_num_'+split_name.split('_')[0]+'.p')
    with open(fpath,'w') as f:
        pickle.dump(pn_pairs_num,f)

    return p_pairs, n_pairs



##################### one_pair_rec ###############
import scipy.io
import scipy.stats
import skimage.morphology
from skimage.morphology import square, dilation, erosion
from PIL import Image
def _getPoseMask(peaks, height, width, radius=4, var=4, mode='Solid'):
    ## MSCOCO Pose part_str = [nose, neck, Rsho, Relb, Rwri, Lsho, Lelb, Lwri, Rhip, Rkne, Rank, Lhip, Lkne, Lank, Leye, Reye, Lear, Rear, pt19]
    # find connection in the specified sequence, center 29 is in the position 15
    # limbSeq = [[2,3], [2,6], [3,4], [4,5], [6,7], [7,8], [2,9], [9,10], \
    #            [10,11], [2,12], [12,13], [13,14], [2,1], [1,15], [15,17], \
    #            [1,16], [16,18], [3,17], [6,18]]
    # limbSeq = [[2,3], [2,6], [3,4], [4,5], [6,7], [7,8], [2,9], [9,10], \
    #            [10,11], [2,12], [12,13], [13,14], [2,1], [1,15], [15,17], \
    #            [1,16], [16,18]] # , [9,12]
    # limbSeq = [[3,4], [4,5], [6,7], [7,8], [9,10], \
    #            [10,11], [12,13], [13,14], [2,1], [1,15], [15,17], \
    #            [1,16], [16,18]] # 
    limbSeq = [[2,3], [2,6], [3,4], [4,5], [6,7], [7,8], [2,9], [9,10], \
                         [10,11], [2,12], [12,13], [13,14], [2,1], [1,15], [15,17], \
                         [1,16], [16,18], [2,17], [2,18], [9,12], [12,6], [9,3], [17,18]] #
    indices = []
    values = []
    for limb in limbSeq:
        p0 = peaks[limb[0] -1]
        p1 = peaks[limb[1] -1]
        if 0!=len(p0) and 0!=len(p1):
            r0 = p0[0][1]
            c0 = p0[0][0]
            r1 = p1[0][1]
            c1 = p1[0][0]
            ind, val = _getSparseKeypoint(r0, c0, 0, height, width, radius, var, mode)
            indices.extend(ind)
            values.extend(val)
            ind, val = _getSparseKeypoint(r1, c1, 0, height, width, radius, var, mode)
            indices.extend(ind)
            values.extend(val)
        
            distance = np.sqrt((r0-r1)**2 + (c0-c1)**2)
            sampleN = int(distance/radius)
            # sampleN = 0
            if sampleN>1:
                for i in xrange(1,sampleN):
                    r = r0 + (r1-r0)*i/sampleN
                    c = c0 + (c1-c0)*i/sampleN
                    ind, val = _getSparseKeypoint(r, c, 0, height, width, radius, var, mode)
                    indices.extend(ind)
                    values.extend(val)

    shape = [height, width, 1]
    ## Fill body
    dense = np.squeeze(_sparse2dense(indices, values, shape))
    ## TODO
    # im = Image.fromarray((dense*255).astype(np.uint8))
    # im.save('xxxxx.png')
    # pdb.set_trace()
    dense = dilation(dense, square(5))
    dense = erosion(dense, square(5))
    return dense


Ratio_0_4 = 1.0/scipy.stats.norm(0, 4).pdf(0)
Gaussian_0_4 = scipy.stats.norm(0, 4)
def _getSparseKeypoint(r, c, k, height, width, radius=4, var=4, mode='Solid'):
    r = int(r)
    c = int(c)
    k = int(k)
    indices = []
    values = []
    for i in range(-radius, radius+1):
        for j in range(-radius, radius+1):
            distance = np.sqrt(float(i**2+j**2))
            if r+i>=0 and r+i<height and c+j>=0 and c+j<width:
                if 'Solid'==mode and distance<=radius:
                    indices.append([r+i, c+j, k])
                    values.append(1)
                elif 'Gaussian'==mode and distance<=radius:
                    indices.append([r+i, c+j, k])
                    if 4==var:
                        values.append( Gaussian_0_4.pdf(distance) * Ratio_0_4  )
                    else:
                        assert 'Only define Ratio_0_4  Gaussian_0_4 ...'
    return indices, values

def _getSparsePose(peaks, height, width, channel, radius=4, var=4, mode='Solid'):
    indices = []
    values = []
    for k in range(len(peaks)):
        p = peaks[k]
        if 0!=len(p):
            r = p[0][1]
            c = p[0][0]
            ind, val = _getSparseKeypoint(r, c, k, height, width, radius, var, mode)
            indices.extend(ind)
            values.extend(val)
    shape = [height, width, channel]
    return indices, values, shape

def _oneDimSparsePose(indices, shape):
    ind_onedim = []
    for ind in indices:
        # idx = ind[2]*shape[0]*shape[1] + ind[1]*shape[0] + ind[0]
        idx = ind[0]*shape[2]*shape[1] + ind[1]*shape[2] + ind[2]
        ind_onedim.append(idx)
    shape = np.prod(shape)
    return ind_onedim, shape

def _sparse2dense(indices, values, shape):
    dense = np.zeros(shape)
    for i in range(len(indices)):
        r = indices[i][0]
        c = indices[i][1]
        k = indices[i][2]
        dense[r,c,k] = values[i]
    return dense

def _get_valid_peaks(all_peaks, subsets):
    try:
        subsets = subsets.tolist()
        valid_idx = -1
        valid_score = -1
        for i, subset in enumerate(subsets):
            score = subset[-2]
            # for s in subset:
            #   if s > -1:
            #     cnt += 1
            if score > valid_score:
                valid_idx = i
                valid_score = score
        if valid_idx>=0:
            peaks = []
            cand_id_list = subsets[valid_idx][:18]

            for ap in all_peaks:
                valid_p = []
                for p in ap:
                    if p[-1] in cand_id_list:
                        valid_p = p
                if len(valid_p)>0: # use the same structure with all_peaks
                    peaks.append([(valid_p)])
                else:
                    peaks.append([])
            return peaks
        else:
            return all_peaks ## Avoid to return None
        #     return None
    except Exception as e:
        print("Unexpected error:")
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        # fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_tb.tb_lineno)
        # pdb.set_trace()
        return None

import matplotlib.pyplot as plt 
import scipy.misc
def _visualizePose(pose, img):
    # pdb.set_trace()
    if 3==len(pose.shape):
        pose = pose.max(axis=-1, keepdims=True)
        pose = np.tile(pose, (1,1,3))
    elif 2==len(pose.shape):
        pose = np.expand_dims(pose, -1)
        pose = np.tile(pose, (1,1,3))

    imgShow = ((pose.astype(np.float)+1)/2.0*img.astype(np.float)).astype(np.uint8)
    plt.imshow(imgShow)
    plt.show()


def _format_data(sess, image_reader, folder_path, pairs, idx, labels, id_map, attr_onehot_mat, attr_w2v25_mat, 
                    attr_w2v50_mat, attr_w2v100_mat, attr_w2v150_mat, id_map_attr, all_peaks_dic, subsets_dic, 
                    seg_data_dir, FiltOutMissRegion=False, FLIP=False):
    # Read the filename:
    img_path_0 = os.path.join(folder_path, pairs[idx][0])
    img_path_1 = os.path.join(folder_path, pairs[idx][1])

    id_0 = pairs[idx][0][0:4]
    id_1 = pairs[idx][1][0:4]
    cam_0 = pairs[idx][0][6]
    cam_1 = pairs[idx][1][6]

    image_raw_0 = tf.gfile.FastGFile(img_path_0, 'r').read()
    image_raw_1 = tf.gfile.FastGFile(img_path_1, 'r').read()
    height, width = image_reader.read_image_dims(sess, image_raw_0)

    ########################## Attribute ##########################
    attrs_0 = []
    attrs_1 = []
    attrs_w2v25_0 = []
    attrs_w2v25_1 = []
    attrs_w2v50_0 = []
    attrs_w2v50_1 = []
    attrs_w2v100_0 = []
    attrs_w2v100_1 = []
    attrs_w2v150_0 = []
    attrs_w2v150_1 = []
    idx_0 = id_map_attr[id_0]
    idx_1 = id_map_attr[id_1]
    # pdb.set_trace()
    if attr_onehot_mat is not None:
        for name in attr_onehot_mat.dtype.names:
            attrs_0.append(attr_onehot_mat[(name)][0][0][0][idx_0])
            attrs_1.append(attr_onehot_mat[(name)][0][0][0][idx_1])
    if attr_w2v25_mat is not None:
        for i in xrange(attr_w2v25_mat[0].shape[0]):
            attrs_w2v25_0 = attrs_w2v25_0 + attr_w2v25_mat[0][i][idx_0].tolist()
            attrs_w2v25_1 = attrs_w2v25_1 + attr_w2v25_mat[0][i][idx_1].tolist()
    if attr_w2v50_mat is not None:
        for i in xrange(attr_w2v50_mat[0].shape[0]):
            attrs_w2v50_0 = attrs_w2v50_0 + attr_w2v50_mat[0][i][idx_0].tolist()
            attrs_w2v50_1 = attrs_w2v50_1 + attr_w2v50_mat[0][i][idx_1].tolist()
    if attr_w2v100_mat is not None:
        for i in xrange(attr_w2v100_mat[0].shape[0]):
            attrs_w2v100_0 = attrs_w2v100_0 + attr_w2v100_mat[0][i][idx_0].tolist()
            attrs_w2v100_1 = attrs_w2v100_1 + attr_w2v100_mat[0][i][idx_1].tolist()
    if attr_w2v150_mat is not None:
        for i in xrange(attr_w2v150_mat[0].shape[0]):
            attrs_w2v150_0 = attrs_w2v150_0 + attr_w2v150_mat[0][i][idx_0].tolist()
            attrs_w2v150_1 = attrs_w2v150_1 + attr_w2v150_mat[0][i][idx_1].tolist()

    ########################## Segment ##########################
    seg_0 = np.zeros([128,64])
    seg_1 = np.zeros([128,64])
    if seg_data_dir:
        path_0 = os.path.join(seg_data_dir, pairs[idx][0])
        path_1 = os.path.join(seg_data_dir, pairs[idx][1])
        if os.exists(path_0) and os.exists(path_1):
            seg_0 = scipy.misc.imread(path_0)
            seg_1 = scipy.misc.imread(path_1)
            if FLIP:
                # pdb.set_trace()
                seg_0 = np.fliplr(seg_0)
                seg_1 = np.fliplr(seg_1)
        else:
            return None

    ########################## Pose 16x8 & Pose coodinate (for 128x64(Solid) 128x64(Gaussian))##########################
    ## Pose 16x8
    w_unit = width/8
    h_unit = height/16
    pose_peaks_0 = np.zeros([16,8,18])
    pose_peaks_1 = np.zeros([16,8,18])
    ## Pose coodinate
    pose_peaks_0_rcv = np.zeros([18,3]) ## Row, Column, Visibility
    pose_peaks_1_rcv = np.zeros([18,3])
    #
    pose_subs_0 = []
    pose_subs_1 = []
    # pdb.set_trace()
    if (all_peaks_dic is not None) and (pairs[idx][0] in all_peaks_dic) and (pairs[idx][1] in all_peaks_dic):
        ###### Pose 0 ######
        peaks = _get_valid_peaks(all_peaks_dic[pairs[idx][0]], subsets_dic[pairs[idx][0]])
        indices_r4_0, values_r4_0, shape = _getSparsePose(peaks, height, width, 18, radius=4, mode='Solid')
        indices_r4_0, shape_0 = _oneDimSparsePose(indices_r4_0, shape)
        pose_mask_r4_0 = _getPoseMask(peaks, height, width, radius=4, mode='Solid')
        pose_mask_r7_0 = _getPoseMask(peaks, height, width, radius=7, mode='Solid')
        for ii in range(len(peaks)):
            p = peaks[ii]
            if 0!=len(p):
                pose_peaks_0[int(p[0][1]/h_unit), int(p[0][0]/w_unit), ii] = 1
                pose_peaks_0_rcv[ii][0] = p[0][1]
                pose_peaks_0_rcv[ii][1] = p[0][0]
                pose_peaks_0_rcv[ii][2] = 1
        ## Generate body region proposals
        # part_bbox_list_0, visibility_list_0 = get_part_bbox7(peaks, img_path_0, radius=6, idx=idx)
        part_bbox_list_0, visibility_list_0 = get_part_bbox37(peaks, img_path_0, radius=6)
        if FiltOutMissRegion and (0 in visibility_list_0):
            return None

        ###### Pose 1 ######
        peaks = _get_valid_peaks(all_peaks_dic[pairs[idx][1]], subsets_dic[pairs[idx][1]])
        indices_r4_1, values_r4_1, shape = _getSparsePose(peaks, height, width, 18, radius=4, mode='Solid')
        indices_r4_1, shape_1 = _oneDimSparsePose(indices_r4_1, shape)
        pose_mask_r4_1 = _getPoseMask(peaks, height, width, radius=4, mode='Solid')
        pose_mask_r7_1 = _getPoseMask(peaks, height, width, radius=7, mode='Solid')
        ## Generate body region proposals
        # part_bbox_list_1, visibility_list_1 = get_part_bbox7(peaks, img_path_1, radius=7)
        part_bbox_list_1, visibility_list_1 = get_part_bbox37(peaks, img_path_0, radius=6)
        if FiltOutMissRegion and (0 in visibility_list_1):
            return None

        ###### Visualize ######
        # dense = _sparse2dense(indices_r4_0, values_r4_0, shape)
        # _visualizePose(pose_mask_r4_0, scipy.misc.imread(img_path_0))
        # _visualizePose(pose_mask_r7_0, scipy.misc.imread(img_path_0))
        # pdb.set_trace()

        for ii in range(len(peaks)):
            p = peaks[ii]
            if 0!=len(p):
                pose_peaks_1[int(p[0][1]/h_unit), int(p[0][0]/w_unit), ii] = 1
                pose_peaks_1_rcv[ii][0] = p[0][1]
                pose_peaks_1_rcv[ii][1] = p[0][0]
                pose_peaks_1_rcv[ii][2] = 1
        pose_subs_0 = subsets_dic[pairs[idx][0]][0].tolist()
        pose_subs_1 = subsets_dic[pairs[idx][1]][0].tolist()
    else:
        return None


    example = tf.train.Example(features=tf.train.Features(feature={
            'image_name_0': dataset_utils.bytes_feature(pairs[idx][0]),
            'image_name_1': dataset_utils.bytes_feature(pairs[idx][1]),
            'image_raw_0': dataset_utils.bytes_feature(image_raw_0),
            'image_raw_1': dataset_utils.bytes_feature(image_raw_1),
            'label': dataset_utils.int64_feature(labels[idx]),
            'id_0': dataset_utils.int64_feature(id_map[id_0]),
            'id_1': dataset_utils.int64_feature(id_map[id_1]),
            'cam_0': dataset_utils.int64_feature(int(cam_0)),
            'cam_1': dataset_utils.int64_feature(int(cam_1)),
            'image_format': dataset_utils.bytes_feature('jpg'),
            'image_height': dataset_utils.int64_feature(height),
            'image_width': dataset_utils.int64_feature(width),
            'real_data': dataset_utils.int64_feature(1),
            'attrs_0': dataset_utils.int64_feature(attrs_0),
            'attrs_1': dataset_utils.int64_feature(attrs_1),
            'attrs_w2v25_0': dataset_utils.float_feature(attrs_w2v25_0),
            'attrs_w2v25_1': dataset_utils.float_feature(attrs_w2v25_1),
            'attrs_w2v50_0': dataset_utils.float_feature(attrs_w2v50_0),
            'attrs_w2v50_1': dataset_utils.float_feature(attrs_w2v50_1),
            'attrs_w2v100_0': dataset_utils.float_feature(attrs_w2v100_0),
            'attrs_w2v100_1': dataset_utils.float_feature(attrs_w2v100_1),
            'attrs_w2v150_0': dataset_utils.float_feature(attrs_w2v150_0),
            'attrs_w2v150_1': dataset_utils.float_feature(attrs_w2v150_1),
            'pose_peaks_0': dataset_utils.float_feature(pose_peaks_0.flatten().tolist()),
            'pose_peaks_1': dataset_utils.float_feature(pose_peaks_1.flatten().tolist()),
            'pose_peaks_0_rcv': dataset_utils.float_feature(pose_peaks_0_rcv.flatten().tolist()),
            'pose_peaks_1_rcv': dataset_utils.float_feature(pose_peaks_1_rcv.flatten().tolist()),
            'pose_mask_r4_0': dataset_utils.int64_feature(pose_mask_r4_0.astype(np.int64).flatten().tolist()),
            'pose_mask_r4_1': dataset_utils.int64_feature(pose_mask_r4_1.astype(np.int64).flatten().tolist()),
            'pose_mask_r6_0': dataset_utils.int64_feature(pose_mask_r7_0.astype(np.int64).flatten().tolist()),
            'pose_mask_r6_1': dataset_utils.int64_feature(pose_mask_r7_1.astype(np.int64).flatten().tolist()),
            'seg_0': dataset_utils.int64_feature(seg_0.astype(np.int64).flatten().tolist()),
            'seg_1': dataset_utils.int64_feature(seg_1.astype(np.int64).flatten().tolist()),

            'shape': dataset_utils.int64_feature(shape_0),
            
            'indices_r4_0': dataset_utils.int64_feature(np.array(indices_r4_0).astype(np.int64).flatten().tolist()),
            'values_r4_0': dataset_utils.float_feature(np.array(values_r4_0).astype(np.float).flatten().tolist()),
            'indices_r4_1': dataset_utils.int64_feature(np.array(indices_r4_1).astype(np.int64).flatten().tolist()),
            'values_r4_1': dataset_utils.float_feature(np.array(values_r4_1).astype(np.float).flatten().tolist()),

            'pose_subs_0': dataset_utils.float_feature(pose_subs_0),
            'pose_subs_1': dataset_utils.float_feature(pose_subs_1),

            'part_bbox_0': dataset_utils.int64_feature(np.array(part_bbox_list_0).astype(np.int64).flatten().tolist()),
            'part_bbox_1': dataset_utils.int64_feature(np.array(part_bbox_list_1).astype(np.int64).flatten().tolist()),
            'part_vis_0': dataset_utils.int64_feature(np.array(visibility_list_0).astype(np.int64).flatten().tolist()),
            'part_vis_1': dataset_utils.int64_feature(np.array(visibility_list_1).astype(np.int64).flatten().tolist()),
    }))

    return example

def get_part_bbox7(peaks, img_path=None, radius=7, idx=None):
    ## Generate body region proposals
    ## MSCOCO Pose part_str = [nose, neck, Rsho, Relb, Rwri, Lsho, Lelb, Lwri, Rhip, Rkne, Rank, Lhip, Lkne, Lank, Leye, Reye, Lear, Rear, pt19]
    ## part1: nose, neck, Rsho, Lsho, Leye, Reye, Lear, Rear [0,1,2,5,14,15,16,17]
    ## part2: Rsho, Relb, Rwri, Lsho, Lelb, Lwri, Rhip, Lhip [2,3,4,5,6,7,8,11]
    ## part3: Rhip, Rkne, Rank, Lhip, Lkne, Lank [8,9,10,11,12,13]
    ## part4: Lsho, Lelb, Lwri [3,6,7]
    ## part5: Rsho, Relb, Rwri [2,4,5]
    ## part6: Lhip, Lkne, Lank [11,12,13]
    ## part7: Rhip, Rkne, Rank [8,9,10]
    part_idx_list_all = [ [0,1,2,5,14,15,16,17], ## part1: nose, neck, Rsho, Lsho, Leye, Reye, Lear, Rear
                        [2,3,4,5,6,7,8,11], ## part2: Rsho, Relb, Rwri, Lsho, Lelb, Lwri, Rhip, Lhip
                        [8,9,10,11,12,13], ## part3: Rhip, Rkne, Rank, Lhip, Lkne, Lank
                        [5,6,7], ## part4: Lsho, Lelb, Lwri
                        [2,3,4], ## part5: Rsho, Relb, Rwri
                        [11,12,13], ## part6: Lhip, Lkne, Lank
                        [8,9,10] ] ## part7: Rhip, Rkne, Rank
    part_idx_list = part_idx_list_all ## select all
    part_bbox_list = [] ## bbox: normalized coordinates [y1, x1, y2, x2]
    visibility_list = []
    r = radius
    r_single = 10
    for ii in range(len(part_idx_list)):
        part_idx = part_idx_list[ii]
        xs = []
        ys = []
        select_peaks = [peaks[i] for i in part_idx]
        for p in select_peaks:
            if 0!=len(p):
                xs.append(p[0][0])
                ys.append(p[0][1])
        if len(xs)==0:
            # print('miss peaks')
            visibility_list.append(0)
            part_bbox_list.append([0,0,1,1])
            # return None
        else:
            visibility_list.append(1)
            y1 = np.array(ys).min()
            x1 = np.array(xs).min()
            y2 = np.array(ys).max()
            x2 = np.array(xs).max()
            if len(xs)>1:
                y1 = max(0,y1-r)
                x1 = max(0,x1-r)
                y2 = min(127,y2+r)
                x2 = min(63,x2+r)
            else:
                y1 = max(0,y1-r_single)
                x1 = max(0,x1-r_single)
                y2 = min(127,y2+r_single)
                x2 = min(63,x2+r_single)
            part_bbox_list.append([y1, x1, y2, x2])
        if idx is not None:
            img = scipy.misc.imread(img_path)
            scipy.misc.imsave('%04d_part%d.jpg'%(idx,ii+1), img[y1:y2,x1:x2,:])

    if idx is not None:
        scipy.misc.imsave('%04d_part_whole.jpg'%idx, img)

    return part_bbox_list, visibility_list

def get_part_bbox37(peaks, img_path=None, radius=7, idx=None):
    ## Generate body region proposals
    ## MSCOCO Pose part_str = [nose, neck, Rsho, Relb, Rwri, Lsho, Lelb, Lwri, Rhip, Rkne, Rank, Lhip, Lkne, Lank, Leye, Reye, Lear, Rear, pt19]
    ## part1: nose, neck, Rsho, Lsho, Leye, Reye, Lear, Rear [0,1,2,5,14,15,16,17]
    ## part2: Rsho, Relb, Rwri, Lsho, Lelb, Lwri, Rhip, Lhip [2,3,4,5,6,7,8,11]
    ## part3: Rhip, Rkne, Rank, Lhip, Lkne, Lank [8,9,10,11,12,13]
    ## part4: Lsho, Lelb, Lwri [3,6,7]
    ## part5: Rsho, Relb, Rwri [2,4,5]
    ## part6: Lhip, Lkne, Lank [11,12,13]
    ## part7: Rhip, Rkne, Rank [8,9,10]
    ###################################
    ## part8: Rsho, Lsho, Rhip, Lhip [2,5,8,11]
    ## part9: Lsho, Lelb [5,6]
    ## part10: Lelb, Lwri [6,7]
    ## part11: Rsho, Relb [2,3]
    ## part12: Relb, Rwri [3,4]
    ## part13: Lhip, Lkne [11,12]
    ## part14: Lkne, Lank [12,13]
    ## part15: Rhip, Rkne [8,9]
    ## part16: Rkne, Rank [9,10]
    ## part17: WholeBody range(0,18)
    ## part18-36: single key point [0],...,[17]
    ## part36: Rsho, Relb, Rwri, Rhip, Rkne, Rank [2,3,4,8,9,10]
    ## part37: Lsho, Lelb, Lwri, Lhip, Lkne, Lank [5,6,7,11,12,13]
    part_idx_list_all = [ [0,1,2,5,14,15,16,17], ## part1: nose, neck, Rsho, Lsho, Leye, Reye, Lear, Rear
                        [2,3,4,5,6,7,8,11], ## part2: Rsho, Relb, Rwri, Lsho, Lelb, Lwri, Rhip, Lhip
                        [8,9,10,11,12,13], ## part3: Rhip, Rkne, Rank, Lhip, Lkne, Lank
                        [5,6,7], ## part4: Lsho, Lelb, Lwri
                        [2,3,4], ## part5: Rsho, Relb, Rwri
                        [11,12,13], ## part6: Lhip, Lkne, Lank
                        [8,9,10], ## part7: Rhip, Rkne, Rank
                        [2,5,8,11], ## part8: Rsho, Lsho, Rhip, Lhip
                        [5,6], ## part9: Lsho, Lelb
                        [6,7], ## part10: Lelb, Lwri
                        [2,3], ## part11: Rsho, Relb
                        [3,4], ## part12: Relb, Rwri
                        [11,12], ## part13: Lhip, Lkne
                        [12,13], ## part14: Lkne, Lank
                        [8,9], ## part15: Rhip, Rkne
                        [9,10], ## part16: Rkne, Rank
                        range(0,18) ] ## part17: WholeBody
    part_idx_list_all.extend([[i] for i in range(0,18)]) ## part18-35: single key point
    part_idx_list_all.extend([ [2,3,4,8,9,10], ## part36: Rsho, Relb, Rwri, Rhip, Rkne, Rank
                        [5,6,7,11,12,13]]) ## part37: Lsho, Lelb, Lwri, Lhip, Lkne, Lank
    # part_idx_list = [part_idx_list_all[i] for i in [0,1,2,3,4,5,6,7,8,16]] ## select >3 keypoints
    part_idx_list = part_idx_list_all ## select all
    part_bbox_list = [] ## bbox: normalized coordinates [y1, x1, y2, x2]
    visibility_list = []
    r = radius
    r_single = 10
    for ii in range(len(part_idx_list)):
        part_idx = part_idx_list[ii]
        xs = []
        ys = []
        select_peaks = [peaks[i] for i in part_idx]
        for p in select_peaks:
            if 0!=len(p):
                xs.append(p[0][0])
                ys.append(p[0][1])
        if len(xs)==0:
            # print('miss peaks')
            visibility_list.append(0)
            part_bbox_list.append([0,0,1,1])
            # return None
        else:
            visibility_list.append(1)
            y1 = np.array(ys).min()
            x1 = np.array(xs).min()
            y2 = np.array(ys).max()
            x2 = np.array(xs).max()
            if len(xs)>1:
                y1 = max(0,y1-r)
                x1 = max(0,x1-r)
                y2 = min(127,y2+r)
                x2 = min(63,x2+r)
            else:
                y1 = max(0,y1-r_single)
                x1 = max(0,x1-r_single)
                y2 = min(127,y2+r_single)
                x2 = min(63,x2+r_single)
            part_bbox_list.append([y1, x1, y2, x2])
        if idx is not None:
            img = scipy.misc.imread(img_path)
            scipy.misc.imsave('%04d_part%d.jpg'%(idx,ii+1), img[y1:y2,x1:x2,:])

    if idx is not None:
        scipy.misc.imsave('%04d_part_whole.jpg'%idx, img)

    return part_bbox_list, visibility_list


def _convert_dataset_one_pair_rec_withFlip(out_dir, split_name, split_name_flip, pairs, pairs_flip, labels, labels_flip, dataset_dir, 
            attr_onehot_mat_path=None, attr_w2v_dir=None, pose_peak_path=None, pose_sub_path=None, pose_peak_path_flip=None, 
            pose_sub_path_flip=None, seg_dir=None, tf_record_pair_num=np.inf):
    """Converts the given pairs to a TFRecord dataset.

    Args:
        split_name: The name of the dataset, either 'train' or 'validation'.
        pairs: A list of image name pairs.
        labels: label list to indicate positive(1) or negative(0)
        dataset_dir: The directory where the converted datasets are stored.
    """
    if split_name_flip is None:
        USE_FLIP = False
    else:
        USE_FLIP = True

    # num_shards = _NUM_SHARDS
    num_shards = 1
    assert split_name in ['train', 'test', 'test_samples', 'all']
    num_per_shard = int(math.ceil(len(pairs) / float(num_shards)))
    folder_path = _get_folder_path(dataset_dir, split_name)
    if USE_FLIP:
        folder_path_flip = _get_folder_path(dataset_dir, split_name_flip)

    # Load attr mat file
    attr_onehot_mat = None
    attr_w2v_mat = None
    if attr_onehot_mat_path or attr_w2v_dir:
        assert split_name in ['train', 'test', 'test_samples']
        id_cnt = 0
        id_map_attr = {}
        filelist = _get_image_file_list(dataset_dir, split_name)
        filelist.sort()
        # pdb.set_trace()
        for i in xrange(0, len(filelist)):
            id_i = filelist[i][0:4]
            if not id_map_attr.has_key(id_i):
                id_map_attr[id_i] = id_cnt
                id_cnt += 1
        print('id_map_attr length:%d' % len(id_map_attr))

    if attr_onehot_mat_path:
        if 'test_samples'==split_name:
            attr_onehot_mat = scipy.io.loadmat(attr_onehot_mat_path)['market_attribute']['test'][0][0]
        else:
            attr_onehot_mat = scipy.io.loadmat(attr_onehot_mat_path)['market_attribute'][split_name][0][0]
    if attr_w2v_dir:
        if split_name in ['test_samples', 'test']:
            attr_w2v25_mat_path = os.path.join(attr_w2v_dir, 'test_att_wordvec_dim25.mat')
            attr_w2v25_mat = scipy.io.loadmat(attr_w2v25_mat_path)['test_att'] 
            attr_w2v50_mat_path = os.path.join(attr_w2v_dir, 'test_att_wordvec_dim50.mat')
            attr_w2v50_mat = scipy.io.loadmat(attr_w2v50_mat_path)['test_att'] 
            attr_w2v100_mat_path = os.path.join(attr_w2v_dir, 'test_att_wordvec_dim100.mat')
            attr_w2v100_mat = scipy.io.loadmat(attr_w2v100_mat_path)['test_att'] 
            attr_w2v150_mat_path = os.path.join(attr_w2v_dir, 'test_att_wordvec_dim150.mat')
            attr_w2v150_mat = scipy.io.loadmat(attr_w2v150_mat_path)['test_att'] 
        else:
            attr_w2v25_mat_path = os.path.join(attr_w2v_dir, 'train_att_wordvec_dim25.mat')
            attr_w2v25_mat = scipy.io.loadmat(attr_w2v25_mat_path)['train_att'] 
            attr_w2v50_mat_path = os.path.join(attr_w2v_dir, 'train_att_wordvec_dim50.mat')
            attr_w2v50_mat = scipy.io.loadmat(attr_w2v50_mat_path)['train_att'] 
            attr_w2v100_mat_path = os.path.join(attr_w2v_dir, 'train_att_wordvec_dim100.mat')
            attr_w2v100_mat = scipy.io.loadmat(attr_w2v100_mat_path)['train_att'] 
            attr_w2v150_mat_path = os.path.join(attr_w2v_dir, 'train_att_wordvec_dim150.mat')
            attr_w2v150_mat = scipy.io.loadmat(attr_w2v150_mat_path)['train_att'] 

    seg_data_dir = None
    if seg_dir:
        if split_name in ['test_samples', 'test']:
            seg_data_dir = os.path.join(seg_dir, 'person_seg_test')
        else:
            seg_data_dir = os.path.join(seg_dir, 'person_seg_train')

    # Load pose pickle file 
    all_peaks_dic = None
    subsets_dic = None
    all_peaks_dic_flip = None
    subsets_dic_flip = None
    with open(pose_peak_path, 'r') as f:
        all_peaks_dic = pickle.load(f)
    with open(pose_sub_path, 'r') as f:
        subsets_dic = pickle.load(f)
    if USE_FLIP:
        with open(pose_peak_path_flip, 'r') as f:
            all_peaks_dic_flip = pickle.load(f)
        with open(pose_sub_path_flip, 'r') as f:
            subsets_dic_flip = pickle.load(f)
    # Transform ids to [0, ..., num_of_ids]
    id_cnt = 0
    id_map = {}
    for i in range(0, len(pairs)):
        id_0 = pairs[i][0][0:4]
        id_1 = pairs[i][1][0:4]
        if not id_map.has_key(id_0):
            id_map[id_0] = id_cnt
            id_cnt += 1
        if not id_map.has_key(id_1):
            id_map[id_1] = id_cnt
            id_cnt += 1
    print('id_map length:%d' % len(id_map))
    if USE_FLIP:
        id_cnt = 0
        id_map_flip = {}
        for i in range(0, len(pairs_flip)):
            id_0 = pairs_flip[i][0][0:4]
            id_1 = pairs_flip[i][1][0:4]
            if not id_map_flip.has_key(id_0):
                id_map_flip[id_0] = id_cnt
                id_cnt += 1
            if not id_map_flip.has_key(id_1):
                id_map_flip[id_1] = id_cnt
                id_cnt += 1
        print('id_map_flip length:%d' % len(id_map_flip))

    with tf.Graph().as_default():
        image_reader = ImageReader()

        with tf.Session('') as sess:

            for shard_id in range(num_shards):
                output_filename = _get_dataset_filename(
                        dataset_dir, out_dir, split_name, shard_id)

                with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                    cnt = 0

                    if USE_FLIP:
                        start_ndx = shard_id * num_per_shard
                        end_ndx = min((shard_id+1) * num_per_shard, len(pairs_flip))
                        for i in range(start_ndx, end_ndx):
                            sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                                    i+1, len(pairs_flip), shard_id))
                            sys.stdout.flush()
                            example = _format_data(sess, image_reader, folder_path_flip, pairs_flip, i, labels_flip, id_map_flip, attr_onehot_mat, 
                                attr_w2v25_mat, attr_w2v50_mat, attr_w2v100_mat, attr_w2v150_mat, id_map_attr, all_peaks_dic_flip, subsets_dic_flip, seg_data_dir, FLIP=True)
                            if None==example:
                                continue
                            tfrecord_writer.write(example.SerializeToString())
                            cnt += 1
                            if cnt==tf_record_pair_num:
                                break

                    start_ndx = shard_id * num_per_shard
                    end_ndx = min((shard_id+1) * num_per_shard, len(pairs))
                    for i in range(start_ndx, end_ndx):
                        sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                                i+1, len(pairs), shard_id))
                        sys.stdout.flush()
                        example = _format_data(sess, image_reader, folder_path, pairs, i, labels, id_map, attr_onehot_mat, 
                            attr_w2v25_mat, attr_w2v50_mat, attr_w2v100_mat, attr_w2v150_mat, id_map_attr, all_peaks_dic, subsets_dic, seg_data_dir, FLIP=False)
                        if None==example:
                            continue
                        tfrecord_writer.write(example.SerializeToString())
                        cnt += 1
                        if cnt==tf_record_pair_num:
                            break

    sys.stdout.write('\n')
    sys.stdout.flush()
    print('cnt:',cnt)
    with open(os.path.join(out_dir,'tf_record_pair_num.txt'),'w') as f:
        f.write('cnt:%d' % cnt)

def run_one_pair_rec(dataset_dir, out_dir, split_name):
    # if not tf.gfile.Exists(dataset_dir):
    #     tf.gfile.MakeDirs(dataset_dir)
    
    if split_name.lower()=='train':
        # ================ Prepare training set ================
        attr_onehot_mat_path = os.path.join(dataset_dir,'Market-1501_Attribute','market_attribute.mat')
        attr_w2v_dir = os.path.join(dataset_dir,'Market-1501_Attribute','word2vec')
        pose_peak_path = os.path.join(dataset_dir,'Market-1501_PoseFiltered','all_peaks_dic_Market-1501_train.p')
        pose_sub_path = os.path.join(dataset_dir,'Market-1501_PoseFiltered','subsets_dic_Market-1501_train.p')
        pose_peak_path_flip = os.path.join(dataset_dir,'Market-1501_PoseFiltered','all_peaks_dic_Market-1501_train_Flip.p')
        pose_sub_path_flip = os.path.join(dataset_dir,'Market-1501_PoseFiltered','subsets_dic_Market-1501_train_Flip.p')
        seg_dir = os.path.join(dataset_dir,'Market-1501_Segment','seg')

        p_pairs, n_pairs = _get_train_all_pn_pairs(dataset_dir, out_dir,
                                                split_name=split_name,
                                                augment_ratio=1, 
                                                mode='same_diff_cam')
        p_labels = [1]*len(p_pairs)
        n_labels = [0]*len(n_pairs)
        pairs = p_pairs
        labels = p_labels
        combined = list(zip(pairs, labels))
        random.shuffle(combined)
        pairs[:], labels[:] = zip(*combined)

        split_name_flip='train_flip'
        p_pairs_flip, n_pairs_flip = _get_train_all_pn_pairs(dataset_dir, out_dir,
                                                split_name=split_name_flip,
                                                augment_ratio=1, 
                                                mode='same_diff_cam')
        p_labels_flip = [1]*len(p_pairs_flip)
        n_labels_flip = [0]*len(n_pairs_flip)
        pairs_flip = p_pairs_flip
        labels_flip = p_labels_flip
        combined = list(zip(pairs_flip, labels_flip))
        random.shuffle(combined)
        pairs_flip[:], labels_flip[:] = zip(*combined)

        # print('os.remove pn_pairs_num_train_flip.p')
        # os.remove(os.path.join(out_dir, 'pn_pairs_num_train_flip.p'))
        
        _convert_dataset_one_pair_rec_withFlip(out_dir, split_name, split_name_flip, pairs, pairs_flip, labels, labels_flip, dataset_dir, attr_onehot_mat_path=attr_onehot_mat_path,
            attr_w2v_dir=attr_w2v_dir, pose_peak_path=pose_peak_path, pose_sub_path=pose_sub_path, pose_peak_path_flip=pose_peak_path_flip, pose_sub_path_flip=pose_sub_path_flip)

        print('\nTrain convert Finished !')

    elif split_name.lower()=='test':
        #================ Prepare testing set ================
        attr_onehot_mat_path = os.path.join(dataset_dir,'Market-1501_Attribute','market_attribute.mat')
        attr_w2v_dir = os.path.join(dataset_dir,'Market-1501_Attribute','word2vec')
        pose_peak_path = os.path.join(dataset_dir,'Market-1501_PoseFiltered','all_peaks_dic_Market-1501_test_clean.p')
        pose_sub_path = os.path.join(dataset_dir,'Market-1501_PoseFiltered','subsets_dic_Market-1501_test_clean.p')
        seg_dir = os.path.join(dataset_dir,'Market-1501_Segment','seg')

        p_pairs, n_pairs = _get_train_all_pn_pairs(dataset_dir, out_dir, 
                                                split_name=split_name,
                                                augment_ratio=1, 
                                                mode='same_diff_cam')
        p_labels = [1]*len(p_pairs)
        n_labels = [0]*len(n_pairs)
        pairs = p_pairs
        labels = p_labels
        combined = list(zip(pairs, labels))
        random.shuffle(combined)
        pairs[:], labels[:] = zip(*combined)

        ## Test will not use flip
        split_name_flip = None   
        pairs_flip = None
        labels_flip = None
        _convert_dataset_one_pair_rec_withFlip(out_dir, split_name, split_name_flip, pairs, pairs_flip, labels, labels_flip, dataset_dir, attr_onehot_mat_path=attr_onehot_mat_path,
            attr_w2v_dir=attr_w2v_dir, pose_peak_path=pose_peak_path, pose_sub_path=pose_sub_path, tf_record_pair_num=12800)

        print('\nTest convert Finished !')

    elif split_name.lower()=='test_samples':
        #================ Prepare testing sample set ================
        attr_onehot_mat_path = os.path.join(dataset_dir,'Market-1501_Attribute','market_attribute.mat')
        attr_w2v_dir = os.path.join(dataset_dir,'Market-1501_Attribute','word2vec')
        pose_peak_path = os.path.join(dataset_dir,'Market-1501_PoseFiltered','all_peaks_dic_Market-1501_test_samples.p')
        pose_sub_path = os.path.join(dataset_dir,'Market-1501_PoseFiltered','subsets_dic_Market-1501_test_samples.p')
        seg_dir = os.path.join(dataset_dir,'Market-1501_Segment','seg')

        p_pairs, n_pairs = _get_train_all_pn_pairs(dataset_dir, out_dir, 
                                                split_name=split_name,
                                                augment_ratio=1, 
                                                mode='same_diff_cam')
        p_labels = [1]*len(p_pairs)
        n_labels = [0]*len(n_pairs)
        pairs = p_pairs
        labels = p_labels

        ## Test will not use flip
        split_name_flip = None
        pairs_flip = None
        labels_flip = None        
        _convert_dataset_one_pair_rec_withFlip(out_dir, split_name, split_name_flip, pairs, pairs_flip, labels, labels_flip, dataset_dir, attr_onehot_mat_path=attr_onehot_mat_path,
            attr_w2v_dir=attr_w2v_dir, pose_peak_path=pose_peak_path, pose_sub_path=pose_sub_path)

        print('\nTest_sample convert Finished !')

  

if __name__ == '__main__':
    dataset_dir = sys.argv[1]
    split_name = sys.argv[2]   ## 'train', 'test', 'test_samples'
    out_dir = os.path.join(dataset_dir, 'Market_%s_data'%split_name)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    run_one_pair_rec(dataset_dir, out_dir, split_name)

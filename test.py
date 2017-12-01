# -*- coding: utf-8 -*-
# @Time    : 17-11-21 上午11:42
# @Author  : Fei Xue
# @Email   : feixue@pku.edu.cn
# @File    : test.py
# @Software: PyCharm Community Edition


import numpy as np
from lib.binvox_rw import *
from lib.voxel import *
import theano
import theano.tensor as tensor
import scipy.io as sio



def downsample_tensor(stride=2):
    data = tensor.ones([32, 32, 32])
    shape = data.shape
    out = data[0:shape[0]:stride, 0:shape[1]:stride, 0:shape[2]:stride]
    return out


def downsample(data, stride=2):
    shape = data.shape
    out = data[0:shape[0]:stride, 0:shape[1]:stride, 0:shape[2]:stride]
    return out


def load_voxel(filename="/home/fei/Research/Dataset/ShapeNet/ShapeNetVox32/02828884/1a40eaf5919b1b3f3eaa2b95b99dae6/model.binvox"):
    with open(filename, 'rb') as f:
        voxel = read_as_3d_array(f)
    # voxel_array = np.array(voxel.data)

    voxel_downsample2 = downsample(voxel.data, stride=2)
    voxel_downsample4 = downsample(voxel.data, stride=4)
    voxel_downsample8 = downsample(voxel.data, stride=8)

    voxel2obj("test2.obj", voxel.data)
    # voxel2obj("model_downsample2.obj", voxel_downsample2)
    # voxel2obj("model_downsample4.obj", voxel_downsample4)
    # voxel2obj("model_downsample8.obj", voxel_downsample8)

    # print(voxel_downsampled.shape)
    return voxel


def show_result():
    data = sio.loadmat('show.mat')
    pred = data['pred']
    voxel = data['label']
    idxes = data['id'][0]

    # print(pred[0, ...])
    print(idxes)
    for i in range(9):
        raw_voxel = voxel[i, :,  1, :, :]
        raw_pred = pred[i, :, 1, :, :] < 2
        id = int(idxes[i])

        voxel2obj('pred_%d' % id + '.obj', raw_pred)
        voxel2obj('voxel_%d' % id + '.obj', raw_voxel)


if __name__ == '__main__':
    # show_result()
    load_voxel()
    # out = downsample_tensor()
    # print(out)

    # a = tensor.ones((4, 4, 4, 4))
    # b = tensor.ones((4, 4, 1, 1))
    #
    # c = a * b
    # print(c.shape)
    #
    # results = {'prob': np.zeros((5, 32, 32, 32))}
    #
    # x = np.random.normal((32, 2, 32, 32))
    #
    # # print(list(x)[:, 0, :, :, :])
    #
    # for i in range(5):
    #     x = np.random.normal(0, 2, (32, 2, 32, 32))
    #     # print(x)
    #     # print(x.shape)
    #     results['prob'][i, ...] = x[:, 1, :, :]
    #
    # print(results)
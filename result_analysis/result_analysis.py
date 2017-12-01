# -*- coding: utf-8 -*-
# @Time    : 17-11-22 下午3:42
# @Author  : Fei Xue
# @Email   : feixue@pku.edu.cn
# @File    : result_analysis.py
# @Software: PyCharm Community Edition

import numpy as np
import os
import scipy.io as sio

category_ids = ['02691156', '02828884', '02933112', '02958343', '03001627',
                   '03211117', '03636649', '03691459', '04090263', '04256520',
                   '04379243', '04401088', '04530566']
category_names = ['plane', 'bench', 'cabinet', 'car', 'chair',
                  'display', 'lamp', 'loudspeaker', 'rifle', 'sofa',
                  'table', 'telephone', 'vessel']
category_samples = [809, 364, 315, 1500, 1356, 219, 464, 324, 475, 635, 1702, 211, 388]
thresh = [0.3, 0.4, 0.45, 0.5, 0.55, 0.6]


def load_result(fn):
    data = sio.loadmat(fn)

    return data


def divide_data(data, start, len):
    return data[start:start+len]


def divide_IoU(data, start, len):
    return data[start:start+len, 0, :]

def average_data(cost, mAP):
    mean_cost = np.mean(cost)
    std_cost = np.sqrt(np.cov(cost))

    mean_ap = np.mean(mAP)
    std_ap = np.sqrt(np.mean(mAP))

    return mean_cost, std_cost, mean_ap, std_ap


def compute_IoU(intersection, union):
    return intersection/union


def average_IoU(IoU):
    mean_IoU = np.mean(IoU)
    std_IoU  = np.sqrt(np.cov(IoU))

    return mean_IoU, std_IoU


def extract_IoU(data, th):
    tmp = data[str(th)]
    return  th[:, 0, :][:, 1] / th[:, 0, :][:, 2]


def main():
    data = load_result('../output/default/test/seresgru2d3d_result_40000.mat')
    cost = data['cost'][0]
    mAP =  data['mAP']
    IoU = data['IoU']

    # IoUs = []
    # n_threshold = len(thresh)
    # for i, th in enumerate(thresh):
    #     IoUs.append(data[str(th)])

    # print(len(IoU))
    # print(IoU[0, 0, :])
    # print(IoU[1, 0, 0])
    # exit(0)

    n_category = len(category_ids)

    start = 0
    th = 0.4
    # IoU = data[str(th)][:, 0, :][:, 1] / data[str(0.4)][:, 0, :][:, 2]
    num_threshold = 5
    thresholds = [0.35, 0.4, 0.45, 0.5, 0.55]
    for i in range(n_category):
        len_i = category_samples[i]
        cost_i = divide_data(cost, start, len_i)
        mAP_i = divide_data(mAP, start, len_i)
        IoU_i = divide_IoU(IoU, start, len_i)

        mean_cost, std_cost, mean_ap, std_ap = average_data(cost_i, mAP_i)

        print("%s, mean_cost: %f, std_cost: %f, mean_ap: %f, std_ap: %f" %
              (category_names[i], mean_cost, std_cost, mean_ap, std_ap))
        for j in range(num_threshold):
            mean_IoU, std_IoU = average_IoU(IoU_i[:, j])
            print("mean_IoU: %f, std_IoU: %f" % (mean_IoU, std_IoU))

        start += len_i



if __name__ == '__main__':
    main()
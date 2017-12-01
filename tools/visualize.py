# -*- coding: utf-8 -*-
# @Time    : 17-11-22 上午11:17
# @Author  : Fei Xue
# @Email   : feixue@pku.edu.cn
# @File    : visualize.py
# @Software: PyCharm Community Edition

import numpy as np
import matplotlib.pyplot as plt
import os


def load_data(fn):
    with open(fn, "r") as f:
        data = f.readlines()

    return data


def plot_data(data, title=None, xlabel=None, ylabel=None, savename=None, stride=1):
    plt.figure()
    sample_data = data[0:len(data):stride]
    plt.plot(sample_data, 'r-')
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if title is not None:
        plt.title(title)

    if savename is not None:
        plt.savefig(savename)


    plt.show()



if __name__ == '__main__':
    data = load_data("../loss.60000.txt")

    print(len(data))
    plot_data(data[5000:20000], stride=5)
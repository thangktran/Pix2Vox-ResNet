# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import cv2
import matplotlib.pyplot as plt
import os

# from mpl_toolkits.mplot3d import Axes3D


def get_volume_views(volume, save_dir, n_itr, remark):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    volume = volume.squeeze().__ge__(0.5)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_aspect('auto')
    ax.voxels(volume, facecolors="grey", edgecolors="black")

    ax.axes.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.xaxis.set_ticks([])
    ax.axes.xaxis.set_visible(False)

    ax.axes.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.axes.yaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticks([])
    ax.axes.yaxis.set_visible(False)

    ax.axes.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.axes.zaxis.set_ticklabels([])
    ax.axes.zaxis.set_ticks([])
    ax.axes.zaxis.set_visible(False)

    ax.axes.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.axes.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.axes.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)

    plt.grid(False)
    plt.axis('off')

    save_path = os.path.join(save_dir, 'voxels-%06d_%s.png' % (n_itr, remark))
    plt.savefig(save_path, transparent=True)
    plt.close()
    return cv2.imread(save_path)

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
    # ax = fig.gca(projection=Axes3D.name)
    ax = fig.gca(projection='3d')
    ax.set_aspect('auto')
    ax.voxels(volume, edgecolor="k")

    save_path = os.path.join(save_dir, 'voxels-%06d_%s.png' % (n_itr, remark))
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    return cv2.imread(save_path)

# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

from easydict import EasyDict as edict
import pathlib

CURRENT_FILE_DIR = str(pathlib.Path(__file__).parent.resolve())

__C                                         = edict()
cfg                                         = __C

__C.DATASET                                 = edict()
__C.DATASET.MEAN                            = [0.5, 0.5, 0.5]
__C.DATASET.STD                             = [0.5, 0.5, 0.5]
__C.DATASET.TRAIN_DATASET                   = 'ShapeNet' # Pix3D, Pascal3D
__C.DATASET.TEST_DATASET                    = 'ShapeNet'

__C.DATASETS                                = edict()
__C.DATASETS.SHAPENET                       = edict()
__C.DATASETS.SHAPENET.TAXONOMY_FILE_PATH    = CURRENT_FILE_DIR + '/data/DataSets/ShapeNet_gen.json'
__C.DATASETS.SHAPENET.RENDERING_PATH        = CURRENT_FILE_DIR + '/data/DataSets/ShapeNetRendering/%s/%s/rendering/%02d.png'
__C.DATASETS.SHAPENET.VOXEL_PATH            = CURRENT_FILE_DIR + '/data/DataSets/ShapeNetVox32/%s/%s/model.binvox'


__C.TRAIN                                   = edict()
__C.TRAIN.NUM_EPOCHES                       = 350 ##STEPS

#
# Common
#
__C.CONST                                   = edict()
__C.CONST.RNG_SEED                          = 0
__C.CONST.DEVICE                            = None # '0'
__C.CONST.BATCH_SIZE                        = 1
__C.CONST.N_VIEWS_RENDERING                 = 1
__C.CONST.WEIGHTS                           = '/home/thangktran/projects/github/M.Sc.-Robotics-Cognition-Intelligence/sem2/M.Sc.-IN2392_ML3D/mvcnn_with_enet_reconstruction/output/best-ckpt.pth'
__C.CONST.IMG_W                             = 224       # Image width for input
__C.CONST.IMG_H                             = 224       # Image height for input
__C.CONST.CROP_IMG_W                        = 128       # Dummy property for Pascal 3D
__C.CONST.CROP_IMG_H                        = 128       # Dummy property for Pascal 3D
__C.CONST.N_VOX                             = 32

#
# Directories
#
__C.DIR                                     = edict()
__C.DIR.OUT_PATH                            = CURRENT_FILE_DIR + '/output_gen'

#
# Network
#
__C.NETWORK                                 = edict()
__C.NETWORK.LEAKY_VALUE                     = .2
__C.NETWORK.TCONV_USE_BIAS                  = False

#
# Testing options
#
__C.TEST                                    = edict()
__C.TEST.RANDOM_BG_COLOR_RANGE              = [[240, 240], [240, 240], [240, 240]]
# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

from easydict import EasyDict as edict
import pathlib

CURRENT_FILE_DIR = str(pathlib.Path(__file__).parent.resolve())

__C                                         = edict()
cfg                                         = __C

#
# Dataset Config
#
__C.DATASETS                                = edict()
__C.DATASETS.SHAPENET                       = edict()
__C.DATASETS.SHAPENET.TAXONOMY_FILE_PATH    = CURRENT_FILE_DIR + '/data/DataSets/ShapeNet_overfit.json'
__C.DATASETS.SHAPENET.RENDERING_PATH        = CURRENT_FILE_DIR + '/data/DataSets/ShapeNetRendering/%s/%s/rendering/%02d.png'
__C.DATASETS.SHAPENET.VOXEL_PATH            = CURRENT_FILE_DIR + '/data/DataSets/ShapeNetVox32/%s/%s/model.binvox'
# __C.DATASETS.PASCAL3D                       = edict()
# __C.DATASETS.PASCAL3D.TAXONOMY_FILE_PATH    = CURRENT_FILE_DIR + '/data/DataSets/Pascal3D.json'
# __C.DATASETS.PASCAL3D.ANNOTATION_PATH       = CURRENT_FILE_DIR + '/data/DataSets/PASCAL3D/Annotations/%s_imagenet/%s.mat'
# __C.DATASETS.PASCAL3D.RENDERING_PATH        = CURRENT_FILE_DIR + '/data/DataSets/PASCAL3D/Images/%s_imagenet/%s.JPEG'
# __C.DATASETS.PASCAL3D.VOXEL_PATH            = CURRENT_FILE_DIR + '/data/DataSets/PASCAL3D/CAD/%s/%02d.binvox'
# __C.DATASETS.PIX3D                          = edict()
# __C.DATASETS.PIX3D.TAXONOMY_FILE_PATH       = CURRENT_FILE_DIR + '/data/DataSets/Pix3D.json'
# __C.DATASETS.PIX3D.ANNOTATION_PATH          = CURRENT_FILE_DIR + '/data/DataSets/Pix3D/pix3d.json'
# __C.DATASETS.PIX3D.RENDERING_PATH           = CURRENT_FILE_DIR + '/data/DataSets/Pix3D/img/%s/%s.%s'
# __C.DATASETS.PIX3D.VOXEL_PATH               = CURRENT_FILE_DIR + '/data/DataSets/Pix3D/model/%s/%s/%s.binvox'

#
# Dataset
#
__C.DATASET                                 = edict()
__C.DATASET.MEAN                            = [0.5, 0.5, 0.5]
__C.DATASET.STD                             = [0.5, 0.5, 0.5]
__C.DATASET.TRAIN_DATASET                   = 'ShapeNet' # Pix3D, Pascal3D
__C.DATASET.TEST_DATASET                    = 'ShapeNet'

#
# Common
#
__C.CONST                                   = edict()
__C.CONST.DEVICE                            = '0'
__C.CONST.RNG_SEED                          = 0
__C.CONST.IMG_W                             = 224       # Image width for input
__C.CONST.IMG_H                             = 224       # Image height for input
__C.CONST.N_VOX                             = 32
__C.CONST.BATCH_SIZE                        = 64
__C.CONST.N_VIEWS_RENDERING                 = 3         # Dummy property for Pascal 3D ##STEPS
__C.CONST.CROP_IMG_W                        = 128       # Dummy property for Pascal 3D
__C.CONST.CROP_IMG_H                        = 128       # Dummy property for Pascal 3D
__C.CONST.WEIGHTS                           = '/cluster/51/tt/proj/output/checkpoints/2022-07-17T10:28:42.362870/ckpt-epoch-0240.pth' ##STEPS

#
# Directories
#
__C.DIR                                     = edict()
__C.DIR.OUT_PATH                            = CURRENT_FILE_DIR + '/output'
__C.DIR.RANDOM_BG_PATH                      = CURRENT_FILE_DIR + '/data/DataSets/SUN2012/JPEGImages'

#
# Network
#
__C.NETWORK                                 = edict()
__C.NETWORK.LEAKY_VALUE                     = .2
__C.NETWORK.TCONV_USE_BIAS                  = False
__C.NETWORK.USE_REFINER                     = True
__C.NETWORK.USE_MERGER                      = True

#
# Training
#
__C.TRAIN                                   = edict()
__C.TRAIN.RESUME_TRAIN                      = True ##STEPS
__C.TRAIN.NUM_WORKER                        = 4             # number of data workers
__C.TRAIN.NUM_EPOCHES                       = 350 ##STEPS
__C.TRAIN.BRIGHTNESS                        = .4
__C.TRAIN.CONTRAST                          = .4
__C.TRAIN.SATURATION                        = .4
__C.TRAIN.NOISE_STD                         = .1
__C.TRAIN.RANDOM_BG_COLOR_RANGE             = [[225, 255], [225, 255], [225, 255]]
__C.TRAIN.POLICY                            = 'adam'        # available options: sgd, adam
__C.TRAIN.EPOCH_START_USE_REFINER           = 251 ##STEPS
__C.TRAIN.EPOCH_START_USE_MERGER            = 251 ##STEPS
__C.TRAIN.ENCODER_LEARNING_RATE             = 1e-3
__C.TRAIN.DECODER_LEARNING_RATE             = 1e-3
__C.TRAIN.REFINER_LEARNING_RATE             = 1e-3
__C.TRAIN.MERGER_LEARNING_RATE              = 1e-4
__C.TRAIN.ENCODER_LR_MILESTONES             = [150]
__C.TRAIN.DECODER_LR_MILESTONES             = [150]
__C.TRAIN.REFINER_LR_MILESTONES             = [150]
__C.TRAIN.MERGER_LR_MILESTONES              = [150]
__C.TRAIN.BETAS                             = (.9, .999)
__C.TRAIN.MOMENTUM                          = .9
__C.TRAIN.GAMMA                             = .5
__C.TRAIN.SAVE_FREQ                         = 10            # weights will be overwritten every save_freq epoch
__C.TRAIN.UPDATE_N_VIEWS_RENDERING          = True ##STEPS

#
# Testing options
#
__C.TEST                                    = edict()
__C.TEST.RANDOM_BG_COLOR_RANGE              = [[240, 240], [240, 240], [240, 240]]
__C.TEST.VOXEL_THRESH                       = [.2, .3, .4, .5]
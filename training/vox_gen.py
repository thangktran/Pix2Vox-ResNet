# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import json
import numpy as np
import os
import torch
import torch.backends.cudnn
import torch.utils.data


import data.data_loaders
import utils.data_transforms
import utils.binvox_visualization
import utils.network_utils

from datetime import datetime as dt

import models.encoder
import models.decoder
import models.refiner
import models.merger

import models_original.encoder
import models_original.decoder
import models_original.refiner
import models_original.merger


def vox_gen(cfg,
             epoch_idx=-1,
             output_dir=None,
             test_data_loader=None,
             test_writer=None):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    # Load taxonomies of dataset
    taxonomies = []
    with open(cfg.DATASETS[cfg.DATASET.TEST_DATASET.upper()].TAXONOMY_FILE_PATH, encoding='utf-8') as file:
        taxonomies = json.loads(file.read())
    taxonomies = {t['taxonomy_id']: t for t in taxonomies}

    if not output_dir:
        output_dir = os.path.join(cfg.DIR.OUT_PATH, '%s', dt.now().isoformat())

    # Set up data loader
    if test_data_loader is None:
        # Set up data augmentation

        IMG_SIZE = cfg.CONST.IMG_H, cfg.CONST.IMG_W
        CROP_SIZE = cfg.CONST.CROP_IMG_H, cfg.CONST.CROP_IMG_W
        test_transforms = utils.data_transforms.Compose([
            utils.data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
            utils.data_transforms.RandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE),
            utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
            utils.data_transforms.ToTensor(),
        ])

        dataset_loader = data.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
        test_data_loader = torch.utils.data.DataLoader(dataset=dataset_loader.get_dataset(
            data.data_loaders.DatasetType.TEST, cfg.CONST.N_VIEWS_RENDERING, test_transforms),
                                                       batch_size=1,
                                                       num_workers=1,
                                                       pin_memory=True,
                                                       shuffle=False)

    # Set up networks
    encoder_pix2vox_a = models_original.encoder.Encoder(cfg)
    decoder_pix2vox_a = models_original.decoder.Decoder(cfg)
    refiner_pix2vox_a = models_original.refiner.Refiner(cfg)
    merger_pix2vox_a = models_original.merger.Merger(cfg)

    encoder_resnet = models.encoder.Encoder(cfg)
    decoder_resnet = models.decoder.Decoder(cfg)
    refiner_resnet = models.refiner.Refiner(cfg)
    merger_resnet = models.merger.Merger(cfg)

    if torch.cuda.is_available() and cfg.CONST.DEVICE:
        encoder_pix2vox_a = torch.nn.DataParallel(encoder_pix2vox_a).cuda()
        decoder_pix2vox_a = torch.nn.DataParallel(decoder_pix2vox_a).cuda()
        refiner_pix2vox_a = torch.nn.DataParallel(refiner_pix2vox_a).cuda()
        merger_pix2vox_a = torch.nn.DataParallel(merger_pix2vox_a).cuda()

        encoder_resnet = torch.nn.DataParallel(encoder_resnet).cuda()
        decoder_resnet = torch.nn.DataParallel(decoder_resnet).cuda()
        refiner_resnet = torch.nn.DataParallel(refiner_resnet).cuda()
        merger_resnet = torch.nn.DataParallel(merger_resnet).cuda()

    print('[INFO] %s Loading weights from %s ...' % (dt.now(), cfg.CONST.WEIGHTS))
    checkpoint_pix2vox_a = None
    checkpoint_resnet = None

    pix2VoxPath = cfg.CONST.WEIGHTS + "Pix2Vox-A-ShapeNet.pth"

    if not cfg.CONST.DEVICE:
        checkpoint_pix2vox_a = torch.load(pix2VoxPath, map_location=torch.device('cpu'))
        checkpoint_resnet = torch.load(cfg.CONST.WEIGHTS, map_location=torch.device('cpu'))
    else:
        checkpoint_pix2vox_a = torch.load(pix2VoxPath)
        checkpoint_resnet = torch.load(cfg.CONST.WEIGHTS)
        
    def rm_mod(module):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in module.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        return new_state_dict


    encoder_pix2vox_a.load_state_dict(rm_mod(checkpoint_pix2vox_a['encoder_state_dict']))
    decoder_pix2vox_a.load_state_dict(rm_mod(checkpoint_pix2vox_a['decoder_state_dict']))
    refiner_pix2vox_a.load_state_dict(rm_mod(checkpoint_pix2vox_a['refiner_state_dict']))
    merger_pix2vox_a.load_state_dict(rm_mod(checkpoint_pix2vox_a['merger_state_dict']))

    encoder_resnet.load_state_dict(rm_mod(checkpoint_resnet['encoder_state_dict']))
    decoder_resnet.load_state_dict(rm_mod(checkpoint_resnet['decoder_state_dict']))
    refiner_resnet.load_state_dict(rm_mod(checkpoint_resnet['refiner_state_dict']))
    merger_resnet.load_state_dict(rm_mod(checkpoint_resnet['merger_state_dict']))

            
    # Testing loop
    n_samples = len(test_data_loader)

    # Switch models to evaluation mode
    encoder_pix2vox_a.eval()
    decoder_pix2vox_a.eval()
    refiner_pix2vox_a.eval()
    merger_pix2vox_a.eval()

    encoder_resnet.eval()
    decoder_resnet.eval()
    refiner_resnet.eval()
    merger_resnet.eval()

    for sample_idx, (taxonomy_id, sample_name, rendering_images, ground_truth_volume) in enumerate(test_data_loader):
        taxonomy_id = taxonomy_id[0] if isinstance(taxonomy_id[0], str) else taxonomy_id[0].item()
        sample_name = sample_name[0]

        with torch.no_grad():
            # Get data from data loader
            rendering_images = utils.network_utils.var_or_cuda(rendering_images)
            ground_truth_volume = utils.network_utils.var_or_cuda(ground_truth_volume)

            # Test the encoder, decoder, refiner and merger
            image_features_pix2vox_a = encoder_pix2vox_a(rendering_images)
            raw_features_pix2vox_a, generated_volume_pix2vox_a = decoder_pix2vox_a(image_features_pix2vox_a)
            generated_volume_pix2vox_a = merger_pix2vox_a(raw_features_pix2vox_a, generated_volume_pix2vox_a)
            generated_volume_pix2vox_a = refiner_pix2vox_a(generated_volume_pix2vox_a)

            image_features_resnet = encoder_resnet(rendering_images)
            raw_features_resnet, generated_volume_resnet = decoder_resnet(image_features_resnet)
            generated_volume_resnet = merger_resnet(raw_features_resnet, generated_volume_resnet)
            generated_volume_resnet = refiner_resnet(generated_volume_resnet)

            img_dir = output_dir % 'images'

            key = str(taxonomy_id) + '_' + sample_name + '_' + str(sample_idx) + '_'

            # Volume Visualization
            gv_pix2vox_a = generated_volume_pix2vox_a.cpu().numpy()
            rendering_views = utils.binvox_visualization.get_volume_views(gv_pix2vox_a, os.path.join(img_dir, 'test'),
                                                                            epoch_idx, key + 'gv_pix2vox_a')
            gv_resnet = generated_volume_resnet.cpu().numpy()
            rendering_views = utils.binvox_visualization.get_volume_views(gv_resnet, os.path.join(img_dir, 'test'),
                                                                           epoch_idx, key + 'gv_resnet')
                                                                           
            gtv = ground_truth_volume.cpu().numpy()
            rendering_views = utils.binvox_visualization.get_volume_views(gtv, os.path.join(img_dir, 'test'),
                                                                            epoch_idx, key + 'gtv')
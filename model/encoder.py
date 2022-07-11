# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>
#
# References:
# - https://github.com/shawnxu1318/MVCNN-Multi-View-Convolutional-Neural-Networks/blob/master/mvcnn.py

import torch
from efficientnet_pytorch import EfficientNet


class Encoder(torch.nn.Module):
    def __init__(self, cfg):
        super(Encoder, self).__init__()
        self.cfg = cfg

        # Layer Definition
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b0')
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1280, 512, kernel_size=2, padding=1),
            torch.nn.BatchNorm2d(512)
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 256, kernel_size=1),
            torch.nn.BatchNorm2d(256),
        )

        # Don't update params in EfficientNet
        for param in self.efficientnet.parameters():
            param.requires_grad = False

    def forward(self, rendering_images):
        # print(rendering_images.size())  # torch.Size([batch_size, n_views, img_c, img_h, img_w])
        rendering_images = rendering_images.permute(1, 0, 2, 3, 4).contiguous()
        rendering_images = torch.split(rendering_images, 1, dim=0)
        image_features = []

        for img in rendering_images:
            features = self.efficientnet.extract_features(img.squeeze(dim=0))
            # print(features.size())    # torch.Size([batch_size, 1280, 7, 7])?
            features = self.layer1(features)
            # print(features.size())    # torch.Size([batch_size, 512, 8, 8])?
            features = self.layer2(features)
            # print(features.size())    # torch.Size([batch_size, 256, 8, 8])?
            image_features.append(features)

        image_features = torch.stack(image_features).permute(1, 0, 2, 3, 4).contiguous()
        # print(image_features.size())  # torch.Size([batch_size, n_views, 256, 8, 8])
        return image_features

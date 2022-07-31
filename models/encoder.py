# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>
#
# References:
# - https://github.com/shawnxu1318/MVCNN-Multi-View-Convolutional-Neural-Networks/blob/master/mvcnn.py

import torch
import torchvision.models as models
import utils.network_utils


class Encoder(torch.nn.Module):
    def __init__(self, cfg):
        super(Encoder, self).__init__()
        self.cfg = cfg

        # Layer Definition
        resnet152Full = models.resnet152(pretrained=True)
        self.resnet152 = torch.nn.Sequential(*list(resnet152Full.children())[:-4])

        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=3),
            torch.nn.BatchNorm2d(512),
            torch.nn.ELU(),
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=3),
            torch.nn.BatchNorm2d(512),
            torch.nn.ELU(),
            torch.nn.MaxPool2d(kernel_size=3)
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 256, kernel_size=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ELU()
        )

        # Don't update params in EfficientNet
        for param in self.resnet152.parameters():
            param.requires_grad = False

    def forward(self, rendering_images):
        # print(rendering_images.size())  # torch.Size([batch_size, n_views, img_c, img_h, img_w])
        rendering_images = rendering_images.permute(1, 0, 2, 3, 4).contiguous()
        rendering_images = torch.split(rendering_images, 1, dim=0)
        image_features = []

        for img in rendering_images:
            features = self.resnet152(img.squeeze(dim=0))
            # print('resnet')
            # print(features.shape)
            features = self.layer1(features)
            # print('l1')
            # print(features.shape)
            features = self.layer2(features)
            # print('l2')
            # print(features.shape)
            features = self.layer3(features)
            # print('l3')
            # print(features.shape)
            image_features.append(features)

        image_features = torch.stack(image_features).permute(1, 0, 2, 3, 4).contiguous()

        return utils.network_utils.var_or_cuda(image_features)

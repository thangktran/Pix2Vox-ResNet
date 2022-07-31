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
        self.enetB7 = EfficientNet.from_pretrained('efficientnet-b7')

        self.layer0 = torch.nn.Sequential(
            torch.nn.Conv2d(2560, 1280, kernel_size=1),
            torch.nn.BatchNorm2d(1280),
            torch.nn.ELU(),
        )
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1280, 512, kernel_size=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ELU(),
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=3),
            torch.nn.BatchNorm2d(512),
            torch.nn.ELU(),
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=3),
            torch.nn.BatchNorm2d(512),
            torch.nn.ELU(),
            torch.nn.MaxPool2d(kernel_size=3)
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 256, kernel_size=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ELU()
        )

        # =========================================

        # self.layer0 = torch.nn.Sequential(
        #     torch.nn.Conv2d(2560, 1280, kernel_size=1, padding=2),
        #     torch.nn.BatchNorm2d(1280),
        #     torch.nn.ELU(),
        # )
        # self.layer1 = torch.nn.Sequential(
        #     torch.nn.Conv2d(1280, 512, kernel_size=1, padding=2),
        #     torch.nn.BatchNorm2d(512),
        #     torch.nn.ELU(),
        # )
        # self.layer2 = torch.nn.Sequential(
        #     torch.nn.Conv2d(512, 512, kernel_size=1, padding=3),
        #     torch.nn.BatchNorm2d(512),
        #     torch.nn.ELU(),
        # )
        # self.layer3 = torch.nn.Sequential(
        #     torch.nn.Conv2d(512, 512, kernel_size=1, padding=2),
        #     torch.nn.BatchNorm2d(512),
        #     torch.nn.ELU(),
        #     torch.nn.MaxPool2d(kernel_size=3)
        # )
        # self.layer4 = torch.nn.Sequential(
        #     torch.nn.Conv2d(512, 256, kernel_size=1),
        #     torch.nn.BatchNorm2d(256),
        #     torch.nn.ELU()
        # )

        # =========================================

        # self.enetB7Full = torch.hub.load("pytorch/vision", "efficientnet_b7", weights="IMAGENET1K_V1")
        # modelList = list(self.enetB7Full.features.children())
        # self.enetB7 = torch.nn.Sequential(*modelList[:-5])

        # self.layer00 = torch.nn.Sequential(
        #     torch.nn.Conv2d(80, 128, kernel_size=1),
        #     torch.nn.BatchNorm2d(128),
        #     torch.nn.ELU(),
        # )

        # self.layer01 = torch.nn.Sequential(
        #     torch.nn.Conv2d(128, 256, kernel_size=1),
        #     torch.nn.BatchNorm2d(256),
        #     torch.nn.ELU(),
        # )

        # self.layer0 = torch.nn.Sequential(
        #     torch.nn.Conv2d(256, 512, kernel_size=1),
        #     torch.nn.BatchNorm2d(512),
        #     torch.nn.ELU(),
        # )

        # self.layer1 = torch.nn.Sequential(
        #     torch.nn.Conv2d(512, 512, kernel_size=3),
        #     torch.nn.BatchNorm2d(512),
        #     torch.nn.ELU(),
        # )
        # self.layer2 = torch.nn.Sequential(
        #     torch.nn.Conv2d(512, 512, kernel_size=3),
        #     torch.nn.BatchNorm2d(512),
        #     torch.nn.ELU(),
        #     torch.nn.MaxPool2d(kernel_size=3)
        # )
        # self.layer3 = torch.nn.Sequential(
        #     torch.nn.Conv2d(512, 256, kernel_size=1),
        #     torch.nn.BatchNorm2d(256),
        #     torch.nn.ELU()
        # )

        # Don't update params in EfficientNet
        for param in self.enetB7.parameters():
            param.requires_grad = False

    def forward(self, rendering_images):
        # print(rendering_images.size())  # torch.Size([batch_size, n_views, img_c, img_h, img_w])
        rendering_images = rendering_images.permute(1, 0, 2, 3, 4).contiguous()
        rendering_images = torch.split(rendering_images, 1, dim=0)
        image_features = []

        for img in rendering_images:

            features = self.enetB7.extract_features(img.squeeze(dim=0))
            # print('resnet')
            # print(features.shape)
            # features = self.layer00(features)
            # print('l00')
            # print(features.shape)
            # features = self.layer01(features)
            # print('l01')
            # print(features.shape)
            features = self.layer0(features)
            # print('l0')
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
            features = self.layer4(features)
            # print('l4')
            # print(features.shape)
            image_features.append(features)

        image_features = torch.stack(image_features).permute(1, 0, 2, 3, 4).contiguous()
        return image_features

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConUnet(nn.Module):
    def __init__(self, num_classes, in_channels=1, initial_filter_size=64,
                 kernel_size=3, do_instancenorm=True, mode="cls"):
        super(SupConUnet, self).__init__()

        self.encoder = UNet(num_classes, in_channels, initial_filter_size, kernel_size, do_instancenorm, stage="train")
        # self.encoder =MultiResUnet(64,64,in_channels)
        if mode == 'mlp':
            self.head = nn.Sequential(nn.Conv2d(initial_filter_size, 256, kernel_size=1),
                                      nn.Conv2d(256, num_classes, kernel_size=1))
        elif mode == "cls":
            self.head = nn.Conv2d(initial_filter_size, num_classes, kernel_size=1)
        else:
            raise NotImplemented("This mode is not supported yet")

    def forward(self, x, y=None, stage="train"):
        # y = self.encoder(x)
        # output = self.head(y)
        # return output
        if stage == "train":
            y = self.encoder(x, stage="train")
            output = self.head(y)
            return output
        if stage == "test":
            y = self.encoder(x, stage="test")
            output = self.head(y)
            return output
        # output = F.normalize(self.head(y), dim=1)
        if stage == "mix_train":
            output1, output2, expand_x, expand_y = self.encoder(x, y, stage="mix_train")
            result1, result2 = self.head(expand_x), self.head(expand_y)
            return output1, output2, result1, result2
        return y


class SupConUnetInfer(nn.Module):
    def __init__(self, num_classes, in_channels=1, initial_filter_size=64,
                 kernel_size=3, do_instancenorm=True):
        super(SupConUnetInfer, self).__init__()

        self.encoder = UNet(num_classes, in_channels, initial_filter_size, kernel_size, do_instancenorm)
        self.head = nn.Conv2d(initial_filter_size, num_classes, kernel_size=1)

    def forward(self, x):
        y = self.encoder(x)
        output = self.head(y)
        # output = F.normalize(self.head(y), dim=1)

        return y, output


class UNet(nn.Module):

    def __init__(self, num_classes, in_channels=1, initial_filter_size=64, kernel_size=3, do_instancenorm=True,
                 stage="train"):
        super().__init__()

        self.contr_1_1 = self.contract(in_channels, initial_filter_size, kernel_size, instancenorm=do_instancenorm)
        self.contr_1_2 = self.contract(initial_filter_size, initial_filter_size, kernel_size,
                                       instancenorm=do_instancenorm)
        self.pool = nn.MaxPool2d(2, stride=2)

        self.contr_2_1 = self.contract(initial_filter_size, initial_filter_size * 2, kernel_size,
                                       instancenorm=do_instancenorm)
        self.contr_2_2 = self.contract(initial_filter_size * 2, initial_filter_size * 2, kernel_size,
                                       instancenorm=do_instancenorm)

        self.contr_3_1 = self.contract(initial_filter_size * 2, initial_filter_size * 2 ** 2, kernel_size,
                                       instancenorm=do_instancenorm)
        self.contr_3_2 = self.contract(initial_filter_size * 2 ** 2, initial_filter_size * 2 ** 2, kernel_size,
                                       instancenorm=do_instancenorm)

        self.contr_4_1 = self.contract(initial_filter_size * 2 ** 2, initial_filter_size * 2 ** 3, kernel_size,
                                       instancenorm=do_instancenorm)
        self.contr_4_2 = self.contract(initial_filter_size * 2 ** 3, initial_filter_size * 2 ** 3, kernel_size,
                                       instancenorm=do_instancenorm)

        self.center = nn.Sequential(
            nn.Conv2d(initial_filter_size * 2 ** 3, initial_filter_size * 2 ** 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(initial_filter_size * 2 ** 4, initial_filter_size * 2 ** 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(initial_filter_size * 2 ** 4, initial_filter_size * 2 ** 3, 2, stride=2),
            nn.ReLU(inplace=True),
        )

        self.expand_4_1 = self.expand(initial_filter_size * 2 ** 4, initial_filter_size * 2 ** 3)
        self.expand_4_2 = self.expand(initial_filter_size * 2 ** 3, initial_filter_size * 2 ** 3)
        self.upscale4 = nn.ConvTranspose2d(initial_filter_size * 2 ** 3, initial_filter_size * 2 ** 2, kernel_size=2,
                                           stride=2)

        self.expand_3_1 = self.expand(initial_filter_size * 2 ** 3, initial_filter_size * 2 ** 2)
        self.expand_3_2 = self.expand(initial_filter_size * 2 ** 2, initial_filter_size * 2 ** 2)
        self.upscale3 = nn.ConvTranspose2d(initial_filter_size * 2 ** 2, initial_filter_size * 2, 2, stride=2)

        self.expand_2_1 = self.expand(initial_filter_size * 2 ** 2, initial_filter_size * 2)
        self.expand_2_2 = self.expand(initial_filter_size * 2, initial_filter_size * 2)
        self.upscale2 = nn.ConvTranspose2d(initial_filter_size * 2, initial_filter_size, 2, stride=2)

        self.expand_1_1 = self.expand(initial_filter_size * 2, initial_filter_size)
        self.expand_1_2 = self.expand(initial_filter_size, initial_filter_size)
        # Output layer for segmentation
        # self.final = nn.Conv2d(initial_filter_size, num_classes, kernel_size=1)  # kernel size for final layer = 1, see paper

        self.softmax = torch.nn.Softmax2d()

        # Output layer for "autoencoder-mode"
        self.output_reconstruction_map = nn.Conv2d(initial_filter_size, out_channels=1, kernel_size=1)

    @staticmethod
    def contract(in_channels, out_channels, kernel_size=3, instancenorm=True):
        if instancenorm:
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.LeakyReLU(inplace=True))
        else:
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
                nn.LeakyReLU(inplace=True))
        return layer

    @staticmethod
    def expand(in_channels, out_channels, kernel_size=3):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
            nn.LeakyReLU(inplace=True),
        )
        return layer

    @staticmethod
    def center_crop(layer, target_width, target_height):
        batch_size, n_channels, layer_width, layer_height = layer.size()
        xy1 = (layer_width - target_width) // 2
        xy2 = (layer_height - target_height) // 2
        return layer[:, :, xy1:(xy1 + target_width), xy2:(xy2 + target_height)]

    def Mix(self, feature1, feature2):

        # bxcxhxw  size
        change_size = feature1.size()[2] // 2
        feature1_up = feature1[:, :, :change_size, :]
        feature1_dwon = feature1[:, :, change_size:, :]
        feature2_up = feature2[:, :, :change_size, :]
        feature2_dwon = feature2[:, :, change_size:, :]
        # feature1_up = self.Confidence_deep_supervise(feature1_up,feature2_up)
        # feature2_dwon = self.Confidence_deep_supervise(feature2_dwon,feature1_dwon)
        feature1 = torch.cat([feature1_up, feature2_dwon], dim=2)
        feature2 = torch.cat([feature2_up, feature1_dwon], dim=2)
        return feature1, feature2

    def Mix_seg(self, feature1, feature2):
        # bxcxhxw  size
        change_size = feature1.size()[2] // 2
        feature1_up = feature1[:, :, :change_size, :]
        feature1_dwon = feature1[:, :, change_size:, :]
        feature2_up = feature2[:, :, :change_size, :]
        feature2_dwon = feature2[:, :, change_size:, :]
        # feature1_up = self.Confidence_deep_supervise(feature1_up,feature2_up)
        # feature2_dwon = self.Confidence_deep_supervise(feature2_dwon,feature1_dwon)
        feature1 = torch.cat([feature1_up, feature2_dwon], dim=2)
        feature2 = torch.cat([feature2_up, feature1_dwon], dim=2)
        return feature1, feature2

    def Confidence_deep_supervise(self, feature1, feature2):
        feature1 = torch.where(feature1>feature2,feature1,feature2)
        return feature1

    # TODO:
    # x1, x2 = encoder(input)
    # output1s = [x1]
    # output2s = [x2]
    # for de1_layer, de2_layer in zip(self.de1_layers, self.de2_layers):
    #     y1 = de1_layer(x1)
    #     y2 = de2_layer(x2)
    #     if i != 0:
    #         output1s.append(y1)
    #         output2s.append(y2)
    #     if i != len(de1_layer):
    #         x1, x2 = op(y1, y2)
    #
    # output1s = [x1, xd1_2, xd1_3, xd1_4]
    # output2s = [x1, xd2_2, xd2_3, xd2_4]
    # return output1s, output2s
    def forward_full_supervised(self, x, enable_concat=True):
        concat_weight = 1
        if not enable_concat:
            concat_weight = 0

        contr_1 = self.contr_1_2(self.contr_1_1(x))
        pool = self.pool(contr_1)

        contr_2 = self.contr_2_2(self.contr_2_1(pool))
        pool = self.pool(contr_2)

        contr_3 = self.contr_3_2(self.contr_3_1(pool))
        pool = self.pool(contr_3)

        contr_4 = self.contr_4_2(self.contr_4_1(pool))
        pool = self.pool(contr_4)

        center = self.center(pool)

        crop = self.center_crop(contr_4, center.size()[2], center.size()[3])
        concat1 = torch.cat([center, crop * concat_weight], 1)

        expand = self.expand_4_2(self.expand_4_1(concat1))
        upscale = self.upscale4(expand)

        crop = self.center_crop(contr_3, upscale.size()[2], upscale.size()[3])
        concat2 = torch.cat([upscale, crop * concat_weight], 1)

        expand = self.expand_3_2(self.expand_3_1(concat2))
        upscale = self.upscale3(expand)

        crop = self.center_crop(contr_2, upscale.size()[2], upscale.size()[3])
        concat3 = torch.cat([upscale, crop * concat_weight], 1)

        expand = self.expand_2_2(self.expand_2_1(concat3))
        upscale = self.upscale2(expand)

        crop = self.center_crop(contr_1, upscale.size()[2], upscale.size()[3])
        concat4 = torch.cat([upscale, crop * concat_weight], 1)

        expand = self.expand_1_2(self.expand_1_1(concat4))

        return expand

    def forward_mix_feature(self, x, y=None, enable_concat=True):
        concat_weight = 1
        if not enable_concat:
            concat_weight = 0
        # encoder layer x - 1
        contr_1_x = self.contr_1_2(self.contr_1_1(x))
        pool_x = self.pool(contr_1_x)
        # encoder layer y -1
        contr_1_y = self.contr_1_2(self.contr_1_1(y))
        pool_y = self.pool(contr_1_y)

        # encoder layer x -2
        contr_2_x = self.contr_2_2(self.contr_2_1(pool_x))
        pool_x = self.pool(contr_2_x)
        # encoder layer y -2
        contr_2_y = self.contr_2_2(self.contr_2_1(pool_y))
        pool_y = self.pool(contr_2_y)

        # encoder layer x -3
        contr_3_x = self.contr_3_2(self.contr_3_1(pool_x))
        pool_x = self.pool(contr_3_x)
        # encoder layer y -3
        contr_3_y = self.contr_3_2(self.contr_3_1(pool_y))
        pool_y = self.pool(contr_3_y)

        # encoder layer x -4
        contr_4_x = self.contr_4_2(self.contr_4_1(pool_x))
        pool_x = self.pool(contr_4_x)
        # encoder layer y -4
        contr_4_y = self.contr_4_2(self.contr_4_1(pool_y))
        pool_y = self.pool(contr_4_y)

        center_x = self.center(pool_x)
        center_y = self.center(pool_y)
        # TODO : define the feature of two encoders
        output1 = [center_x]
        output2 = [center_y]
        # decoder layer x -1
        crop_x = self.center_crop(contr_4_x, center_x.size()[2], center_x.size()[3])
        concat1_x = torch.cat([center_x, crop_x * concat_weight], 1)
        # decoder layer y -1
        crop_y = self.center_crop(contr_4_y, center_y.size()[2], center_y.size()[3])
        concat1_y = torch.cat([center_y, crop_y * concat_weight], 1)

        concat1_x, concat1_y = self.Mix(concat1_x, concat1_y)
        output1.append(concat1_x)
        output2.append(concat1_y)
        # crop_x, crop_y = self.Mix(crop_x, crop_y)
        # output1.append(crop_x)
        # output2.append(crop_y)
        # decoder layer x - 2
        expand_x = self.expand_4_2(self.expand_4_1(concat1_x))
        upscale_x = self.upscale4(expand_x)
        # decoder layer y -2
        expand_y = self.expand_4_2(self.expand_4_1(concat1_y))
        upscale_y = self.upscale4(expand_y)

        # concat operation -x-1
        crop_x = self.center_crop(contr_3_x, upscale_x.size()[2], upscale_x.size()[3])
        concat2_x = torch.cat([upscale_x, crop_x * concat_weight], 1)

        # concat operation -y-1
        crop_y = self.center_crop(contr_3_y, upscale_y.size()[2], upscale_y.size()[3])
        concat2_y = torch.cat([upscale_y, crop_y * concat_weight], 1)
        #TODO :
        # crop_x, crop_y = self.Mix(crop_x, crop_y)
        # output1.append(crop_x)
        # output2.append(crop_y)
        concat2_x, concat2_y = self.Mix(concat2_x, concat2_y)
        output1.append(concat2_x)
        output2.append(concat2_y)
        # decoder layer x -3
        expand_x = self.expand_3_2(self.expand_3_1(concat2_x))
        upscale_x = self.upscale3(expand_x)
        # decoder layer y -3
        expand_y = self.expand_3_2(self.expand_3_1(concat2_y))
        upscale_y = self.upscale3(expand_y)

        crop_x = self.center_crop(contr_2_x, upscale_x.size()[2], upscale_x.size()[3])
        concat3_x = torch.cat([upscale_x, crop_x * concat_weight], 1)

        crop_y = self.center_crop(contr_2_y, upscale_y.size()[2], upscale_y.size()[3])
        concat3_y = torch.cat([upscale_y, crop_y * concat_weight], 1)

        concat3_x, concat3_y = self.Mix(concat3_x, concat3_y)
        output1.append(concat3_x)
        output2.append(concat3_y)

        expand_x = self.expand_2_2(self.expand_2_1(concat3_x))
        upscale_x = self.upscale2(expand_x)

        expand_y = self.expand_2_2(self.expand_2_1(concat3_y))
        upscale_y = self.upscale2(expand_y)

        crop_x = self.center_crop(contr_1_x, upscale_x.size()[2], upscale_x.size()[3])
        concat4_x = torch.cat([upscale_x, crop_x * concat_weight], 1)

        crop_y = self.center_crop(contr_1_y, upscale_y.size()[2], upscale_y.size()[3])
        concat4_y = torch.cat([upscale_y, crop_y * concat_weight], 1)

        concat4_x, concat4_y = self.Mix(concat4_x, concat4_y)
        output1.append(concat4_x)  # [center_x,concat1_x,concat2_x,concat3_x,concat4_x]
        output2.append(concat4_y)  # [center_y,concat1_y,concat2_y,concat3_y,concat4_y]

        expand_x = self.expand_1_2(self.expand_1_1(concat4_x))
        expand_y = self.expand_1_2(self.expand_1_1(concat4_y))

        return output1, output2, expand_x, expand_y

    def forward(self, x, y=None, enable_concat=True, stage="mix_train"):
        if stage == "train":
            expand = self.forward_full_supervised(x, enable_concat=True)
            return expand
        if stage == "mix_train":
            output1, output2, expand_x, expand_y = self.forward_mix_feature(x, y, enable_concat=True)
            return output1, output2, expand_x, expand_y
        if stage == "test":
            expand = self.forward_full_supervised(x, enable_concat=True)
            return expand


class DownsampleUnet(nn.Module):
    def __init__(self, in_channels=1, initial_filter_size=64, kernel_size=3, do_instancenorm=True):
        super().__init__()

        self.contr_1_1 = self.contract(in_channels, initial_filter_size, kernel_size, instancenorm=do_instancenorm)
        self.contr_1_2 = self.contract(initial_filter_size, initial_filter_size, kernel_size,
                                       instancenorm=do_instancenorm)
        self.pool = nn.MaxPool2d(2, stride=2)

        self.contr_2_1 = self.contract(initial_filter_size, initial_filter_size * 2, kernel_size,
                                       instancenorm=do_instancenorm)
        self.contr_2_2 = self.contract(initial_filter_size * 2, initial_filter_size * 2, kernel_size,
                                       instancenorm=do_instancenorm)
        # self.pool2 = nn.MaxPool2d(2, stride=2)

        self.contr_3_1 = self.contract(initial_filter_size * 2, initial_filter_size * 2 ** 2, kernel_size,
                                       instancenorm=do_instancenorm)
        self.contr_3_2 = self.contract(initial_filter_size * 2 ** 2, initial_filter_size * 2 ** 2, kernel_size,
                                       instancenorm=do_instancenorm)
        # self.pool3 = nn.MaxPool2d(2, stride=2)

        self.contr_4_1 = self.contract(initial_filter_size * 2 ** 2, initial_filter_size * 2 ** 3, kernel_size,
                                       instancenorm=do_instancenorm)
        self.contr_4_2 = self.contract(initial_filter_size * 2 ** 3, initial_filter_size * 2 ** 3, kernel_size,
                                       instancenorm=do_instancenorm)
        # self.pool4 = nn.MaxPool2d(2, stride=2)

        self.center = nn.Sequential(
            nn.Conv2d(initial_filter_size * 2 ** 3, initial_filter_size * 2 ** 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(initial_filter_size * 2 ** 4, initial_filter_size * 2 ** 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(initial_filter_size * 2 ** 4, initial_filter_size * 2 ** 3, 2, stride=2),
            nn.ReLU(inplace=True),
        )

        self.softmax = torch.nn.Softmax2d()

    @staticmethod
    def contract(in_channels, out_channels, kernel_size=3, instancenorm=True):
        if instancenorm:
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.LeakyReLU(inplace=True))
        else:
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
                nn.LeakyReLU(inplace=True))
        return layer

    @staticmethod
    def center_crop(layer, target_width, target_height):
        batch_size, n_channels, layer_width, layer_height = layer.size()
        xy1 = (layer_width - target_width) // 2
        xy2 = (layer_height - target_height) // 2
        return layer[:, :, xy1:(xy1 + target_width), xy2:(xy2 + target_height)]

    def forward(self, x, enable_concat=True):
        concat_weight = 1
        if not enable_concat:
            concat_weight = 0

        contr_1 = self.contr_1_2(self.contr_1_1(x))
        pool = self.pool(contr_1)

        contr_2 = self.contr_2_2(self.contr_2_1(pool))
        pool = self.pool(contr_2)

        contr_3 = self.contr_3_2(self.contr_3_1(pool))
        pool = self.pool(contr_3)

        contr_4 = self.contr_4_2(self.contr_4_1(pool))
        pool = self.pool(contr_4)

        center = self.center(pool)

        return center


# TODO:golbal use network

class GlobalConUnet(nn.Module):
    def __init__(self, in_channels=1, initial_filter_size=64):
        super().__init__()
        self.encoder = DownsampleUnet(in_channels, initial_filter_size)

    def forward(self, x):
        y = self.encoder(x)

        return y


class MLP(nn.Module):
    def __init__(self, input_channels=512, num_class=128):
        super().__init__()

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.f1 = nn.Linear(input_channels, input_channels)
        self.f2 = nn.Linear(input_channels, num_class)

    def forward(self, x):
        x = self.gap(x)
        y = self.f1(x.squeeze())
        y = self.f2(y)

        return y


class UpsampleUnet2(nn.Module):
    def __init__(self, in_channels=1, initial_filter_size=64, kernel_size=3, do_instancenorm=True):
        super().__init__()

        self.contr_1_1 = self.contract(in_channels, initial_filter_size, kernel_size, instancenorm=do_instancenorm)
        self.contr_1_2 = self.contract(initial_filter_size, initial_filter_size, kernel_size,
                                       instancenorm=do_instancenorm)
        self.pool = nn.MaxPool2d(2, stride=2)

        self.contr_2_1 = self.contract(initial_filter_size, initial_filter_size * 2, kernel_size,
                                       instancenorm=do_instancenorm)
        self.contr_2_2 = self.contract(initial_filter_size * 2, initial_filter_size * 2, kernel_size,
                                       instancenorm=do_instancenorm)
        # self.pool2 = nn.MaxPool2d(2, stride=2)

        self.contr_3_1 = self.contract(initial_filter_size * 2, initial_filter_size * 2 ** 2, kernel_size,
                                       instancenorm=do_instancenorm)
        self.contr_3_2 = self.contract(initial_filter_size * 2 ** 2, initial_filter_size * 2 ** 2, kernel_size,
                                       instancenorm=do_instancenorm)
        # self.pool3 = nn.MaxPool2d(2, stride=2)

        self.contr_4_1 = self.contract(initial_filter_size * 2 ** 2, initial_filter_size * 2 ** 3, kernel_size,
                                       instancenorm=do_instancenorm)
        self.contr_4_2 = self.contract(initial_filter_size * 2 ** 3, initial_filter_size * 2 ** 3, kernel_size,
                                       instancenorm=do_instancenorm)
        # self.pool4 = nn.MaxPool2d(2, stride=2)

        self.center = nn.Sequential(
            nn.Conv2d(initial_filter_size * 2 ** 3, initial_filter_size * 2 ** 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(initial_filter_size * 2 ** 4, initial_filter_size * 2 ** 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(initial_filter_size * 2 ** 4, initial_filter_size * 2 ** 3, 2, stride=2),
            nn.ReLU(inplace=True),
        )

        self.expand_4_1 = self.expand(initial_filter_size * 2 ** 4, initial_filter_size * 2 ** 3)
        self.expand_4_2 = self.expand(initial_filter_size * 2 ** 3, initial_filter_size * 2 ** 3)
        self.upscale4 = nn.ConvTranspose2d(initial_filter_size * 2 ** 3, initial_filter_size * 2 ** 2, kernel_size=2,
                                           stride=2)

        self.expand_3_1 = self.expand(initial_filter_size * 2 ** 3, initial_filter_size * 2 ** 2)
        self.expand_3_2 = self.expand(initial_filter_size * 2 ** 2, initial_filter_size * 2 ** 2)
        self.upscale3 = nn.ConvTranspose2d(initial_filter_size * 2 ** 2, initial_filter_size * 2, 2, stride=2)

        self.expand_2_1 = self.expand(initial_filter_size * 2 ** 2, initial_filter_size * 2)
        self.expand_2_2 = self.expand(initial_filter_size * 2, initial_filter_size * 2)
        self.softmax = torch.nn.Softmax2d()

    @staticmethod
    def contract(in_channels, out_channels, kernel_size=3, instancenorm=True):
        if instancenorm:
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.LeakyReLU(inplace=True))
        else:
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
                nn.LeakyReLU(inplace=True))
        return layer

    @staticmethod
    def expand(in_channels, out_channels, kernel_size=3):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
            nn.LeakyReLU(inplace=True),
        )
        return layer

    @staticmethod
    def center_crop(layer, target_width, target_height):
        batch_size, n_channels, layer_width, layer_height = layer.size()
        xy1 = (layer_width - target_width) // 2
        xy2 = (layer_height - target_height) // 2
        return layer[:, :, xy1:(xy1 + target_width), xy2:(xy2 + target_height)]

    def forward(self, x, enable_concat=True):
        concat_weight = 1
        if not enable_concat:
            concat_weight = 0

        contr_1 = self.contr_1_2(self.contr_1_1(x))
        pool = self.pool(contr_1)

        contr_2 = self.contr_2_2(self.contr_2_1(pool))
        pool = self.pool(contr_2)

        contr_3 = self.contr_3_2(self.contr_3_1(pool))
        pool = self.pool(contr_3)

        contr_4 = self.contr_4_2(self.contr_4_1(pool))
        pool = self.pool(contr_4)

        center = self.center(pool)

        crop = self.center_crop(contr_4, center.size()[2], center.size()[3])
        concat = torch.cat([center, crop * concat_weight], 1)

        expand = self.expand_4_2(self.expand_4_1(concat))
        upscale = self.upscale4(expand)

        crop = self.center_crop(contr_3, upscale.size()[2], upscale.size()[3])
        concat = torch.cat([upscale, crop * concat_weight], 1)

        expand = self.expand_3_2(self.expand_3_1(concat))
        upscale = self.upscale3(expand)

        crop = self.center_crop(contr_2, upscale.size()[2], upscale.size()[3])
        concat = torch.cat([upscale, crop * concat_weight], 1)

        expand = self.expand_2_2(self.expand_2_1(concat))

        return expand


class LocalConUnet2(nn.Module):
    def __init__(self, num_classes, in_channels=1, initial_filter_size=64):
        super().__init__()
        self.encoder = UpsampleUnet2(in_channels, initial_filter_size)
        self.head = MLP(input_channels=initial_filter_size * 2, num_class=num_classes)

    def forward(self, x):
        y = self.encoder(x)

        return y


class UpsampleUnet3(nn.Module):
    def __init__(self, in_channels=1, initial_filter_size=64, kernel_size=3, do_instancenorm=True):
        super().__init__()

        self.contr_1_1 = self.contract(in_channels, initial_filter_size, kernel_size, instancenorm=do_instancenorm)
        self.contr_1_2 = self.contract(initial_filter_size, initial_filter_size, kernel_size,
                                       instancenorm=do_instancenorm)
        self.pool = nn.MaxPool2d(2, stride=2)

        self.contr_2_1 = self.contract(initial_filter_size, initial_filter_size * 2, kernel_size,
                                       instancenorm=do_instancenorm)
        self.contr_2_2 = self.contract(initial_filter_size * 2, initial_filter_size * 2, kernel_size,
                                       instancenorm=do_instancenorm)
        # self.pool2 = nn.MaxPool2d(2, stride=2)

        self.contr_3_1 = self.contract(initial_filter_size * 2, initial_filter_size * 2 ** 2, kernel_size,
                                       instancenorm=do_instancenorm)
        self.contr_3_2 = self.contract(initial_filter_size * 2 ** 2, initial_filter_size * 2 ** 2, kernel_size,
                                       instancenorm=do_instancenorm)
        # self.pool3 = nn.MaxPool2d(2, stride=2)

        self.contr_4_1 = self.contract(initial_filter_size * 2 ** 2, initial_filter_size * 2 ** 3, kernel_size,
                                       instancenorm=do_instancenorm)
        self.contr_4_2 = self.contract(initial_filter_size * 2 ** 3, initial_filter_size * 2 ** 3, kernel_size,
                                       instancenorm=do_instancenorm)
        # self.pool4 = nn.MaxPool2d(2, stride=2)

        self.center = nn.Sequential(
            nn.Conv2d(initial_filter_size * 2 ** 3, initial_filter_size * 2 ** 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(initial_filter_size * 2 ** 4, initial_filter_size * 2 ** 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(initial_filter_size * 2 ** 4, initial_filter_size * 2 ** 3, 2, stride=2),
            nn.ReLU(inplace=True),
        )

        self.expand_4_1 = self.expand(initial_filter_size * 2 ** 4, initial_filter_size * 2 ** 3)
        self.expand_4_2 = self.expand(initial_filter_size * 2 ** 3, initial_filter_size * 2 ** 3)
        self.upscale4 = nn.ConvTranspose2d(initial_filter_size * 2 ** 3, initial_filter_size * 2 ** 2, kernel_size=2,
                                           stride=2)

        self.expand_3_1 = self.expand(initial_filter_size * 2 ** 3, initial_filter_size * 2 ** 2)
        self.expand_3_2 = self.expand(initial_filter_size * 2 ** 2, initial_filter_size * 2 ** 2)

        self.softmax = torch.nn.Softmax2d()

    @staticmethod
    def contract(in_channels, out_channels, kernel_size=3, instancenorm=True):
        if instancenorm:
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.LeakyReLU(inplace=True))
        else:
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
                nn.LeakyReLU(inplace=True))
        return layer

    @staticmethod
    def expand(in_channels, out_channels, kernel_size=3):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
            nn.LeakyReLU(inplace=True),
        )
        return layer

    @staticmethod
    def center_crop(layer, target_width, target_height):
        batch_size, n_channels, layer_width, layer_height = layer.size()
        xy1 = (layer_width - target_width) // 2
        xy2 = (layer_height - target_height) // 2
        return layer[:, :, xy1:(xy1 + target_width), xy2:(xy2 + target_height)]

    def forward(self, x, enable_concat=True):
        concat_weight = 1
        if not enable_concat:
            concat_weight = 0

        contr_1 = self.contr_1_2(self.contr_1_1(x))
        pool = self.pool(contr_1)

        contr_2 = self.contr_2_2(self.contr_2_1(pool))
        pool = self.pool(contr_2)

        contr_3 = self.contr_3_2(self.contr_3_1(pool))
        pool = self.pool(contr_3)

        contr_4 = self.contr_4_2(self.contr_4_1(pool))
        pool = self.pool(contr_4)

        center = self.center(pool)

        crop = self.center_crop(contr_4, center.size()[2], center.size()[3])
        concat = torch.cat([center, crop * concat_weight], 1)

        expand = self.expand_4_2(self.expand_4_1(concat))
        upscale = self.upscale4(expand)

        crop = self.center_crop(contr_3, upscale.size()[2], upscale.size()[3])
        concat = torch.cat([upscale, crop * concat_weight], 1)

        expand = self.expand_3_2(self.expand_3_1(concat))

        return expand


class LocalConUnet3(nn.Module):
    def __init__(self, num_classes, in_channels=1, initial_filter_size=64):
        super().__init__()
        self.encoder = UpsampleUnet3(in_channels, initial_filter_size)
        self.head = MLP(input_channels=initial_filter_size * 2 * 2, num_class=num_classes)

    def forward(self, x):
        y = self.encoder(x)

        return y

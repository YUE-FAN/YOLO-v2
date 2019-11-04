"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import torch.nn as nn
import torch
import numpy as np


class Yolo(nn.Module):
    def __init__(self, num_classes,
                 anchors=[(1.3221, 1.73145), (3.19275, 4.00944), (5.05587, 8.09892), (9.47112, 4.84053),
                          (11.2364, 10.0071)], layer=99):
        super(Yolo, self).__init__()
        self.num_classes = num_classes
        self.anchors = anchors
        assert layer in [11, 21, 22, 23, 31, 32, 33, 41, 42, 43, 44, 45, 46, 51, 52, 53, 54, 55, 56, 57, 61, 62, 99]
        self.layer = layer

        self.stage1_conv1 = nn.Sequential(nn.Conv2d(3, 32, 3, 1, 1, bias=False), nn.BatchNorm2d(32),
                                          nn.LeakyReLU(0.1, inplace=True), nn.MaxPool2d(2, 2))
        self.stage1_conv2 = nn.Sequential(nn.Conv2d(32, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64),
                                          nn.LeakyReLU(0.1, inplace=True), nn.MaxPool2d(2, 2))
        self.stage1_conv3 = nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1, bias=False), nn.BatchNorm2d(128),
                                          nn.LeakyReLU(0.1, inplace=True))
        self.stage1_conv4 = nn.Sequential(nn.Conv2d(128, 64, 1, 1, 0, bias=False), nn.BatchNorm2d(64),
                                          nn.LeakyReLU(0.1, inplace=True))
        self.stage1_conv5 = nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1, bias=False), nn.BatchNorm2d(128),
                                          nn.LeakyReLU(0.1, inplace=True), nn.MaxPool2d(2, 2))
        self.stage1_conv6 = nn.Sequential(nn.Conv2d(128, 256, 3, 1, 1, bias=False), nn.BatchNorm2d(256),
                                          nn.LeakyReLU(0.1, inplace=True))
        self.stage1_conv7 = nn.Sequential(nn.Conv2d(256, 128, 1, 1, 0, bias=False), nn.BatchNorm2d(128),
                                          nn.LeakyReLU(0.1, inplace=True))
        self.stage1_conv8 = nn.Sequential(nn.Conv2d(128, 256, 3, 1, 1, bias=False), nn.BatchNorm2d(256),
                                          nn.LeakyReLU(0.1, inplace=True), nn.MaxPool2d(2, 2))
        self.stage1_conv9 = nn.Sequential(nn.Conv2d(256, 512, 3, 1, 1, bias=False), nn.BatchNorm2d(512),
                                          nn.LeakyReLU(0.1, inplace=True))
        self.stage1_conv10 = nn.Sequential(nn.Conv2d(512, 256, 1, 1, 0, bias=False), nn.BatchNorm2d(256),
                                           nn.LeakyReLU(0.1, inplace=True))
        self.stage1_conv11 = nn.Sequential(nn.Conv2d(256, 512, 3, 1, 1, bias=False), nn.BatchNorm2d(512),
                                           nn.LeakyReLU(0.1, inplace=True))
        self.stage1_conv12 = nn.Sequential(nn.Conv2d(512, 256, 1, 1, 0, bias=False), nn.BatchNorm2d(256),
                                           nn.LeakyReLU(0.1, inplace=True))
        self.stage1_conv13 = nn.Sequential(nn.Conv2d(256, 512, 3, 1, 1, bias=False), nn.BatchNorm2d(512),
                                           nn.LeakyReLU(0.1, inplace=True))

        self.stage2_a_maxpl = nn.MaxPool2d(2, 2)
        self.stage2_a_conv1 = nn.Sequential(nn.Conv2d(512, 1024, 3, 1, 1, bias=False),
                                            nn.BatchNorm2d(1024), nn.LeakyReLU(0.1, inplace=True))
        self.stage2_a_conv2 = nn.Sequential(nn.Conv2d(1024, 512, 1, 1, 0, bias=False), nn.BatchNorm2d(512),
                                            nn.LeakyReLU(0.1, inplace=True))
        self.stage2_a_conv3 = nn.Sequential(nn.Conv2d(512, 1024, 3, 1, 1, bias=False), nn.BatchNorm2d(1024),
                                            nn.LeakyReLU(0.1, inplace=True))
        self.stage2_a_conv4 = nn.Sequential(nn.Conv2d(1024, 512, 1, 1, 0, bias=False), nn.BatchNorm2d(512),
                                            nn.LeakyReLU(0.1, inplace=True))
        self.stage2_a_conv5 = nn.Sequential(nn.Conv2d(512, 1024, 3, 1, 1, bias=False), nn.BatchNorm2d(1024),
                                            nn.LeakyReLU(0.1, inplace=True))
        self.stage2_a_conv6 = nn.Sequential(nn.Conv2d(1024, 1024, 3, 1, 1, bias=False), nn.BatchNorm2d(1024),
                                            nn.LeakyReLU(0.1, inplace=True))
        self.stage2_a_conv7 = nn.Sequential(nn.Conv2d(1024, 1024, 3, 1, 1, bias=False), nn.BatchNorm2d(1024),
                                            nn.LeakyReLU(0.1, inplace=True))

        self.stage2_b_conv = nn.Sequential(nn.Conv2d(512, 64, 1, 1, 0, bias=False), nn.BatchNorm2d(64),
                                           nn.LeakyReLU(0.1, inplace=True))

        self.stage3_conv1 = nn.Sequential(nn.Conv2d(256 + 1024, 1024, 3, 1, 1, bias=False), nn.BatchNorm2d(1024),
                                          nn.LeakyReLU(0.1, inplace=True))
        self.stage3_conv2 = nn.Conv2d(1024, len(self.anchors) * (5 + num_classes), 1, 1, 0, bias=False)

    def forward(self, output):
        if self.layer == 11:
            output = self.shuffling(output)
        output = self.stage1_conv1(output)
        if self.layer == 21:
            output = self.shuffling(output)
        output = self.stage1_conv2(output)
        if self.layer == 22:
            output = self.shuffling(output)
        output = self.stage1_conv3(output)
        if self.layer == 23:
            output = self.shuffling(output)
        output = self.stage1_conv4(output)
        if self.layer == 31:
            output = self.shuffling(output)
        output = self.stage1_conv5(output)
        if self.layer == 32:
            output = self.shuffling(output)
        output = self.stage1_conv6(output)
        if self.layer == 33:
            output = self.shuffling(output)
        output = self.stage1_conv7(output)
        if self.layer == 41:
            output = self.shuffling(output)
        output = self.stage1_conv8(output)
        if self.layer == 42:
            output = self.shuffling(output)
        output = self.stage1_conv9(output)
        if self.layer == 43:
            output = self.shuffling(output)
        output = self.stage1_conv10(output)
        if self.layer == 44:
            output = self.shuffling(output)
        output = self.stage1_conv11(output)
        if self.layer == 45:
            output = self.shuffling(output)
        output = self.stage1_conv12(output)
        if self.layer == 46:
            output = self.shuffling(output)
        output = self.stage1_conv13(output)

        residual = output

        output_1 = self.stage2_a_maxpl(output)
        if self.layer == 51:
            output_1 = self.shuffling(output_1)
        output_1 = self.stage2_a_conv1(output_1)
        if self.layer == 52:
            output_1 = self.shuffling(output_1)
        output_1 = self.stage2_a_conv2(output_1)
        if self.layer == 53:
            output_1 = self.shuffling(output_1)
        output_1 = self.stage2_a_conv3(output_1)
        if self.layer == 54:
            output_1 = self.shuffling(output_1)
        output_1 = self.stage2_a_conv4(output_1)
        if self.layer == 55:
            output_1 = self.shuffling(output_1)
        output_1 = self.stage2_a_conv5(output_1)
        if self.layer == 56:
            output_1 = self.shuffling(output_1)
        output_1 = self.stage2_a_conv6(output_1)
        if self.layer == 57:
            output_1 = self.shuffling(output_1)
        output_1 = self.stage2_a_conv7(output_1)

        output_2 = self.stage2_b_conv(residual)
        batch_size, num_channel, height, width = output_2.data.size()
        output_2 = output_2.view(batch_size, num_channel // 4, height, 2, width, 2).contiguous()
        output_2 = output_2.permute(0, 3, 5, 1, 2, 4).contiguous()
        output_2 = output_2.view(batch_size, -1, height // 2, width // 2)

        output = torch.cat((output_1, output_2), 1)
        if self.layer == 61:
            output = self.shuffling(output)
        output = self.stage3_conv1(output)

        if self.layer == 62:
            output = self.shuffling(output)

        output = self.stage3_conv2(output)

        return output

    def _setup(self, inplane, spatial_size):
        indices = np.empty((inplane, spatial_size), dtype=np.int64)
        for i in range(inplane):
            indices[i, :] = np.arange(indices.shape[1]) + i * indices.shape[1]
        return indices

    def shuffling(self, x):
        x_shape = x.size()  # [128, 3, 32, 32]
        x = x.view(x_shape[0], -1)
        indices = self._setup(x_shape[1], x_shape[2] * x_shape[3])
        for i in range(x_shape[1]):
            np.random.shuffle(indices[i])
        x = x[:, torch.from_numpy(indices)].view(x_shape)
        return x


if __name__ == "__main__":
    net = Yolo(20)
    print(net.stage1_conv1[0])

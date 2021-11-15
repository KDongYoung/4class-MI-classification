import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DCGAN_Discriminator(nn.Module):

    def __init__(self, args, featmap_dim=512, n_channel=1):
        super(DCGAN_Discriminator, self).__init__()
        self.featmap_dim = featmap_dim
        self.args=args

        self.conv1 = nn.Conv2d(n_channel, featmap_dim // 4, 3, stride=2, padding=2)

        self.conv2 = nn.Conv2d(featmap_dim // 4, featmap_dim // 2, 3, stride=2, padding=2)
        self.BN2 = nn.BatchNorm2d(featmap_dim // 2)

        self.conv3 = nn.Conv2d(featmap_dim // 2, featmap_dim, 3, stride=2, padding=2)
        self.BN3 = nn.BatchNorm2d(featmap_dim)

        self.fc = nn.Linear(featmap_dim * (self.args.img_size//2-7) * (self.args.time_window//2-7), 1)

    def forward(self, x):
        """
        Strided convulation layers,
        Batch Normalization after convulation but not at input layer,
        LeakyReLU activation function with slope 0.2.
        """
        x = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        print(x.shape)
        x = F.leaky_relu(self.BN2(self.conv2(x)), negative_slope=0.2)
        print(x.shape)
        x = F.leaky_relu(self.BN3(self.conv3(x)), negative_slope=0.2)
        print(x.shape)
        x = x.view(-1, self.featmap_dim * (self.args.img_size//2-7) * (self.args.time_window//2-6))
        x = F.sigmoid(self.fc(x))
        return x


class DCGAN_Generator(nn.Module):

    def __init__(self, args, featmap_dim=1024, n_channel=1, noise_dim=100):
        super(DCGAN_Generator, self).__init__()
        self.featmap_dim = featmap_dim
        self.args=args
        self.data_shape=(args.channels, args.img_size, args.time_window)  

        self.fc1 = nn.Linear(noise_dim, (self.args.img_size//2-7) * (self.args.time_window//2-7) * featmap_dim)
        self.conv1 = nn.ConvTranspose2d(featmap_dim, (featmap_dim//2), 3, stride=1, padding=0)

        self.BN1 = nn.BatchNorm2d(featmap_dim//2)
        self.conv2 = nn.ConvTranspose2d(featmap_dim//2, (featmap_dim//4), 6, stride=1, padding=0)

        self.BN2 = nn.BatchNorm2d(featmap_dim//4)
        self.conv3 = nn.ConvTranspose2d(featmap_dim//4, n_channel, 6, stride=2, padding=2)

    def forward(self, x):
        """
        Project noise to featureMap * width * height,
        Batch Normalization after convulation but not at output layer,
        ReLU activation function.
        """
        x = self.fc1(x)
        x = x.view(-1, self.featmap_dim, (self.args.img_size//2)-7, (self.args.time_window//2)-7)
        print(x.shape)
        x = F.relu(self.BN1(self.conv1(x)))
        print(x.shape)
        x = F.relu(self.BN2(self.conv2(x)))
        print(x.shape)
        x = F.tanh(self.conv3(x))
        print(x.shape)

        return x
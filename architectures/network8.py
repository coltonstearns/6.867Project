import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from architectures.network_base import NetworkBase
'''
Encoder based on VGG16 architecture (without final fully connected layers)
'''
class Network_8(NetworkBase):  # inherit from base class torch.nn.Module
    def __init__(self, save_dir, num_classes):
        super(Network_8, self).__init__(save_dir, num_classes)  # initialize Module characteristics
        
        self.section_outputs = [None, None, None]
        self.sections = []
        self.section_pools = []
        
        # 1: goes from (1280 X 720 X 3) -> (640 X 320 X 32) VERY LARGE
        self.conv3_1_64 = nn.Conv2d(3, 8, kernel_size = 5, stride = 1, padding = 4, dilation = 2)
        self.conv3_2_64 = nn.Conv2d(8, 8, kernel_size = 3, stride = 1, padding = 1, dilation = 1)
        self.max_pool_1 = nn.MaxPool2d(2, stride = 2)
        self.sections.append([self.conv3_1_64, self.conv3_2_64])
        self.section_pools.append(self.max_pool_1)
        self.reduction_layers = []
        # max pooling between layers 

        # 2: goes from (640 x 360 x 32) to (320 x 180 x 64)
        self.conv3_1_128 = nn.Conv2d(8, 16, kernel_size=3, stride = 1, padding = 5, dilation = 5)
        self.conv3_2_128 = nn.Conv2d(16, 16, kernel_size=3, stride = 1, padding = 1, dilation = 1)
        self.max_pool_2 = nn.MaxPool2d(2, stride = 2)
        self.sections.append([self.conv3_1_128, self.conv3_2_128])
        self.section_pools.append(self.max_pool_2)

        # 3: goes from (320 x 180 x 64) to (160, 90, 128)
        self.conv3_1_256 = nn.Conv2d(16, 16, kernel_size=3, stride = 1, padding = 1, dilation = 1)
        self.conv3_2_256 = nn.Conv2d(16, 16, kernel_size=3, stride = 1, padding = 3, dilation = 3)
        self.conv3_3_256 = nn.Conv2d(16, 16, kernel_size=3, stride = 1, padding = 3, dilation = 3)
        self.max_pool_3 = nn.MaxPool2d(2, stride = 2)
        self.sections.append([self.conv3_1_256, self.conv3_2_256, self.conv3_3_256])
        self.section_pools.append(self.max_pool_3)

        # 4: goes from (160 x 90 x 128) to (80, 45, 256)
        self.conv3_1_512 = nn.Conv2d(16, 32, kernel_size=3, stride = 1, padding = 2, dilation = 2)
        self.conv3_2_512 = nn.Conv2d(32, 32, kernel_size=3, stride = 1, padding = 1, dilation  = 1)
        self.conv3_3_512 = nn.Conv2d(32, 32, kernel_size=3, stride = 1, padding = 1, dilation = 1)
        self.max_pool_4 = nn.MaxPool2d(2, stride = 2)
        self.sections.append([self.conv3_1_512, self.conv3_2_512, self.conv3_3_512])
        self.section_pools.append(self.max_pool_4)

        # 5: goes from (80, 45, 256) to (80, 45, 256)
        self.conv3_4_512 = nn.Conv2d(32, 16, kernel_size=3, stride = 1, padding = 1)
        self.conv3_5_512 = nn.Conv2d(16, 8, kernel_size=3, stride = 1, padding = 1)
        self.conv3_6_512 = nn.Conv2d(8, 4, kernel_size=3, stride = 1, padding = 1)
        self.sections.append([self.conv3_4_512, self.conv3_5_512, self.conv3_6_512])
        
        self.full_transition = nn.Linear(80 * 4 * 45, 80* 4 * 45)
        # 6: bottom transition layer
        self.bottom_transition = nn.Conv2d(4, 32, kernel_size = 3, padding = 1, stride = 1)

        # 7: (80, 45, 256) to (160 x 90 x 128) 
        self.deconv_1 = nn.ConvTranspose2d(32, 16, kernel_size = 3, stride = 2, padding = 1, output_padding = 1)
        self.skip_connect_1 = nn.Conv2d(32, 16, kernel_size = 1, stride = 1)
        
        # 8: (160 x 90 x 128) to (320 x 180 x 64) 
        self.deconv_2 = nn.ConvTranspose2d(16, 16, kernel_size = 3, stride = 2, padding = 1, output_padding = 1)
        self.skip_connect_2 = nn.Conv2d(32, 16, kernel_size = 1, stride = 1)

        # 9: (320 x 180 x 64) to (640 x 320 x 32)
        self.deconv_3 = nn.ConvTranspose2d(16, 8, kernel_size = 3, stride = 2, padding = 1, output_padding = 1)
        self.skip_connect_3 = nn.Conv2d(16, 8, kernel_size = 1, stride = 1)

        self.deconvolutions = [self.deconv_1, self.deconv_2, self.deconv_3]
        self.skip_connections = [self.skip_connect_1, self.skip_connect_2, self.skip_connect_3]

        # 10: (640 x 320 x 32) to (1280 x 720 x 16)
        self.final_deconv = nn.ConvTranspose2d(8, 4, kernel_size = 3, stride = 2, padding = 1, output_padding = 1)      
        self.classify_layer = nn.Conv2d(4, num_classes, kernel_size = 1, stride = 1)
        

    """
    Defines how we perform a forward pass of the VGG16 neural network
    """
    def forward(self, x):
        # downsample
        for index, section in enumerate(self.sections):
            for layer in section:
                x = F.relu(layer(x))
            if(index < 4):
                x = self.section_pools[index](x)
            if index < 3:
                self.section_outputs[index] = x
        
        x_old_shape = x.size()
        x = F.relu(self.full_transition(x.view(-1))) 
        x = x.view((1, 4, 80, 45)) 
        # perform final convolution at bottom layer
        x = F.relu(self.bottom_transition(x))
        
        # upsample
        for index, deconv_layer in enumerate(self.deconvolutions):
            x = F.relu(deconv_layer(x))  # upsample through deconvolution
            x = torch.cat((x, self.section_outputs[-index-1]), dim = 1)  # concatenate skiplayer to channels
            x = F.relu(self.skip_connections[index](x))  # compute upsample given skip-connection info
        
        x = F.relu(self.final_deconv(x))
        x = self.classify_layer(x)

        return nn.LogSoftmax(dim = 1)(x)


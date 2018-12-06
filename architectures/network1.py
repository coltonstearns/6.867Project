import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Every image is 1280 x 720 pixels.
'''
class Network_1(nn.Module):  # inherit from base class torch.nn.Module
    def __init__(self, save_dir, num_classes):
        super(Network_1, self).__init__()  # initialize Module characteristics
        
        self.save_dir = save_dir
        # goes from (720 x 720 x 3) to (353 x 353 x 8)
        self.conv1 = nn.Conv2d(3, 8, kernel_size=6, stride = 2, padding = 0, dilation = 3)

        # goes from (353 x 353 x 8) to (175 x 175 x 16)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5, stride = 2, padding = 0, dilation = 1)

        # goes from (175 x 175 x 16) to (36 x 36 x 8)
        self.conv3 = nn.Conv2d(16, 8, kernel_size=4, stride = 5, padding = 3, dilation = 1)

        # reduce tensor to simple 3D value; this will be our result
        self.classify_layer = nn.Conv2d(8, num_classes, kernel_size = 1, stride = 1, padding = 0)

    """
    Defines how we perform a forward pass.
    """
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(nn.functional.interpolate(x, scale_factor = 20, mode = 'bilinear', align_corners=True))  # NOT SURE IF WE SHOULD USE RELU HERE!
        x = self.classify_layer(x)  # finish with 2d classification

        return nn.LogSoftmax(dim = 1)(x)  # return softmax for probability of 2d tensor

    def save(self):
        torch.save(self.state_dict(), self.save_dir)

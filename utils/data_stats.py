from torchvision import transforms
import torch.utils.data as data
import torch
from utils.data_loading import DeepDriveDataset, load_datasets
import dill
from tqdm import tqdm
import sys
import pickle


class DataStats:
    IMAGE_WIDTH  = 1280
    IMAGE_HEIGHT = 720

    def __init__(self, dataset, num_classes = 3):
        self.dataset = dataset
        self.class_distribution = torch.zeros((num_classes, self.IMAGE_WIDTH, self.IMAGE_HEIGHT), dtype = torch.float32)
        self.mean_rgb = torch.zeros((3), dtype = torch.float32)
        self.num_classes = num_classes
    
    def get_pixel_distribution(self, pixel):
        r"""
        input:  
            pixel: (pixel_x, pixel_y) the location of the pixel in the image
        output:
            returns a list of probabilities for [class0, class1, class2] 
        """

        return self.class_distribution[:, pixel[0], pixel[1]]
    
    def get_distribution(self):
        r"""
        returns the per-pixel distribution over classes for the dataset
        """
        return self.class_distribution

    def one_hot(self, image):
        r"""
        Converts a 2d tensor of dimension (W, H) with values [0, 1, 2] to a 3d tensor with
        dimension (3, W, H) where the first layer is one at positions where the image has 0, the
        second layer has ones where the image has a 1, and the third layer has ones where the original image 
        had 2's. All other value are 0. 
        """
        new_image = torch.zeros((self.num_classes, image.shape[0], image.shape[1]), dtype = torch.float32)
        for i in range(self.num_classes):
            new_image[i] = image.eq(i)

        if self.num_classes == 2:
            new_image[1] += image.eq(2)

        return new_image

    def collect_all_stats(self, outfile):
        r"""
        Collects information about the prior class distribution, mean RGB values
        """
        print("Collecting Dataset Statistics...")
        #collect dataset statistics
        for idx, (image, target) in tqdm(enumerate(self.dataset)):
            #calculate prior class distribution
            # MIGHT BE FASTER NOT TO DIVIDE DURING CALCULATION --> SAVES US 70000 * (1280*720) divisions?
            self.class_distribution += self.one_hot(target)

            assert(image.shape[0] == 3), "Unexpected image shape: {}".format(image.shape)

            #calculate rgb
            avg_image_r = torch.sum(image[0])
            avg_image_g = torch.sum(image[1])
            avg_image_b = torch.sum(image[2])
            image_rgb = torch.Tensor([avg_image_r, avg_image_g, avg_image_b])
            self.mean_rgb += image_rgb # rolling mean of rgb values

        # normalize to mean
        self.class_distribution /= len(self.dataset)
        self.mean_rgb /= len(self.dataset)

        try:
            with open(outfile, "wb") as ofile:
                #save statistics
                dill.dump([self.class_distribution, self.mean_rgb], ofile)
        except:
            pass

        try:
            list_version = [deep_list(self.class_distribution), deep_list(self.mean_rgb)]
            with open("pickle_" + outfile, "wb") as ofile:
                #save statistics
                pickle.dump(list_version, ofile)
        except:
            pass

    def load_stats(self, infile):
        r"""
        Loads statistics stored in infile
        """
        with open(infile, "rb") as ifile:
            self.class_distribution, self.mean_rgb = dill.load(ifile)


def deep_list(input):
    if len(input.shape) == 0:
        return float(input)

    current = list(input)
    for i in range(len(current)):
        current[i] = deep_list(current[i])

    return current
        




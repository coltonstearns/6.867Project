from torchvision import transforms
import torch.utils.data as data
import torch
from data_loading import DeepDriveDataset, load_datasets
import dill
from tqdm import tqdm


class DataStats:
    def __init__(self, dataset):
        self.dataset = dataset
        self.class_distribution = torch.zeros(dataset[0][1].shape, dtype = torch.float32)
        self.mean_rgb = torch.zeros((3), dtype = torch.float32)
    
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
        new_image = torch.zeros((3, image.shape[0], image.shape[1]), dtype = torch.float32)
        new_image[0] = image.eq(0)
        new_image[1] = image.eq(1)
        new_image[2] = image.eq(2)
        return new_image

    def collect_all_stats(self, outfile):
        r"""
        Collects information about the prior class distribution, mean RGB values
        """
        print("Collecting Dataset Statistics...")
        #collect dataset statistics 
        for idx, (image, target) in tqdm(enumerate(self.dataset)):
            #calculate prior class distribution
            self.class_distribution = (self.class_distribution*idx + self.one_hot(target))/(idx+1)

            assert(image.shape[0] == 3), "Unexpected image shape: {}".format(image.shape)

            #calculate rgb
            avg_image_r = torch.sum(image[0])
            avg_image_g = torch.sum(image[1])
            avg_image_b = torch.sum(image[2])
            image_rgb = torch.Tensor([avg_image_r, avg_image_g, avg_image_b])
            self.mean_rgb = (self.mean_rgb * idx + image_rgb) / (idx + 1)

        with open(outfile, "w") as ofile:
            #save statistics
            dill.dump([self.class_distribution, self.mean_rgb], ofile)
    
    def load_stats(self, infile):
        r"""
        Loads statistics stored in infile
        """
        with open(infile, 'r') as ifile:
            self.class_distribution, self.mean_rgb = dill.load(ifile)

        
        




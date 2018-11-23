from torchvision import transforms
import torch.utils.data as data
import torch

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import os
import os.path
import sys

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

# ======================================================================================#
# =================== Obtain paths to all data images and targets ======================#
# ======================================================================================#

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
        extensions (iterable of strings): extensions to consider (lowercase)
    Returns:
        bool: True if the filename ends with one of given extensions
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def is_image_file(filename):
    """Checks if a file is an allowed image extension.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def make_dataset(image_dir, semantic_image_labels_dir):
    """
    Goes through all images in image_dir and creates a list of (path_to_image, path_to_semantic_label_image) tuples.
    Requires the directory to the semantic-image labels in order to reference them correctly. 
    Args:
        image_dir (string): Root directory path of images.
        semantic_image_labels_dir (string): Root directory path of image-labels
    Returns:
        list: (path_to_image, path_to_label_image), where each is a string to an image on the device.
    """
    # get full root path
    image_dir = os.path.expanduser(image_dir)
    semantic_image_labels_dir = os.path.expanduser(semantic_image_labels_dir)

    # get all image names from the image_dir
    if sys.version_info >= (3, 5):
        # Faster and available in Python 3.5 and above
        files = [d.name for d in os.scandir(image_dir)]
    else:
        files = [d for d in os.listdir(image_dir)] #if os.path.isdir(os.path.join(image_dir, d))

    # go through all image names and create tuple of (image path, label path)
    images_and_lables = []
    print("files length: " + str(len(files)))
    for file in files:
        # check if file is actually an image
        if not is_image_file(file):
            print("file: " + str(file) + " is not an image")
            continue
        # otherwise, add this and it's label counterpart to our list
        label_image_name = file[:-4] + "_drivable_id.png"  # get rid of ".jpg" and add "_drivable_id.png"
        item = (image_dir + "/" + file, semantic_image_labels_dir + "/" + label_image_name)
        images_and_lables.append(item)

    return images_and_lables

# ======================================================================================#
# ======================================================================================#


# =====================================================================================#
# =========================== Loaders used to load images =============================#
# =====================================================================================#
def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def pil_black_and_white_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        x = img.crop((0, 0, 1, 1))
        #x.show()
        return img


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

# ======================================================================================#
# ======================================================================================#


# =====================================================================================#
# =============================== Create Dataset Class ================================#
# =====================================================================================#
class DeepDriveDataset(data.Dataset):
    # loaders for Deep Drive Images and Drivable Maps Labels
    IMAGE_LOADER = default_loader
    TARGET_LOADER = pil_black_and_white_loader

    # extensions for viable images
    EXTENSIONS = IMG_EXTENSIONS

    """A loader for the deep drive dataset for semantic segmentation of images. Loads bdd100k images as well as bdd100k
    drivable maps as labels.

    Args:
        image_dir (string): Root directory path of images.
        semantic_image_labels_dir (string): Root directory path of image-labels
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``

     Attributes:
        samples (list): List of (image path, class_index) tuples
    """
    def __init__(self, image_dir, semantic_image_labels_dir, transform = None):
        # get all of our data
        samples = make_dataset(image_dir, semantic_image_labels_dir)
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in folder of: " + image_dir + "\n"
                               "Supported extensions are: " + ",".join(self.EXTENSIONS)))

        self.root = image_dir
        self.samples = samples
        self.transform = transform
        self.target_transform = None

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) Sample is the sample image, transformed by the transform 
            specification of the Dataset. Target is a (m x n x 3) one-hot encoded torch.tensor containing
            each pixel labeled as 0 or 1 for the 3 classes: not drivable area, drivable other lanes, and drivable
            current lane.
        """
        # load images
        sample_path, target_path = self.samples[index]
        sample = default_loader(sample_path)
        target = pil_black_and_white_loader(target_path)

        # perform equivalent transform on BOTH image and target 
        if self.transform is not None:
            sample, target = self.transform(np.array(sample, dtype = np.float64), np.array(target))

        target = torch.LongTensor(target.T)
        sample = torch.FloatTensor(sample.T)
        return sample, target


    '''
    Overrides object definition of length to be the number of samples in the dataset.
    '''
    def __len__(self):
        return len(self.samples)

    '''
    Defines the string representation of our dataset.
    '''
    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

# ======================================================================================#
# ======================================================================================#


# ======================================================================================#
# ==================== Create the Dataset with desired Transforms ======================#
# ======================================================================================#

'''
Takes in the width and height of all images, as well as the width and height for each
image to be cropped to. Returns a function that, given a tuple of images, returns a tuple
of each image cropped to the same random section. The random section has (crop_width by crop_height).
'''
def random_crop_images(width, height, crop_width, crop_height):
    if crop_width > width or crop_height > height:
        raise ValueError("The crop size must be smaller than the image size.")

    i = np.random.randint(0, width - crop_width) if width != crop_width else 0
    j = np.random.randint(0, height - crop_height) if height != crop_height else 0

    def crop_images(image, target):
        cropped_image = transforms.functional.crop(image, i, j, crop_height, crop_width)
        cropped_target = target.crop((i, j, i + crop_height, j + crop_width))
        return (cropped_image, cropped_target)

    return crop_images






def load_datasets(image_dir = "C:/Users/cstea/Documents/6.867 Final Project/bdd100k_images/bdd100k/images/100k",
                 label_dir = "C:/Users/cstea/Documents/6.867 Final Project/bdd100k_drivable_maps/bdd100k/drivable_maps/labels"):
    '''
    Loads the Berkeley Deep Drive Datasets into a pytorch data.Dataset class. Currently has structure of Berkeley Data Folders
    hard coded into loading scheme, and therefore, this function will fail if one modifies the folder structure of the data.

    Args:
        image_dir (string): the local machine's directory containing the "100k" images
        label_dir (string): the local machine's directory containing the "100k" drivable map --> labels (Note that these png images
            have pixel values of 0 if that pixel is not drivable road area, and 1 if it is)
    '''

    # load train and test datasets given my PC's folder paths

    train_dataset = DeepDriveDataset(image_dir + "/train", label_dir + "/train", transform = preprocess) 
                                        #transform = random_crop_images(1280, 720, 720, 720))
    test_dataset = DeepDriveDataset(image_dir + "/val", label_dir + "/val", transform = preprocess)
                                         #$transform = random_crop_images(1280, 720, 720, 720))

    return train_dataset, test_dataset


# Main method goes through and tests loaded database pictures and tensor labels
if __name__ == "__main__":
    train, test = load_datasets()
    for i in range(10):
        images, targets = train.__getitem__(i)
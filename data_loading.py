from torchvision import transforms
import torch.utils.data as data

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import os
import os.path
import sys


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
        files = [d for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))]

    # go through all image names and create tuple of (image path, label path)
    images_and_lables = []
    for file in files:
        # check if file is actually an image
        if not is_image_file(file):
            continue
        # otherwise, add this and it's label counterpart to our list
        label_image_name = file[:-4] + "_drivable_id.png"  # get rid of ".jpg" and add "_drivable_id.png"
        item = (image_dir + "/" + file, semantic_image_labels_dir + "/" + label_image_name)
        images_and_lables.append(item)

    return images_and_lables


class DatasetFolder(data.Dataset):
    """
    Args:
        image_dir (string): Root directory path of images.
        semantic_image_labels_dir (string): Root directory path of image-labels
        loader (callable): A function to load a sample given its path.
        extensions (list[string]): A list of allowed extensions.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.

     Attributes:
        samples (list): List of (sample path, label path) tuples
        root (string): The path of the folder containing the images
    """

    def __init__(self, image_dir, semantic_image_labels_dir, loader, extensions, transform=None, target_transform=None):
        # get all of our data
        samples = make_dataset(image_dir, semantic_image_labels_dir)
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in folder of: " + image_dir + "\n"
                               "Supported extensions are: " + ",".join(extensions)))

        # pytorch attributes
        self.root = image_dir
        self.loader = loader
        self.extensions = extensions

        # attributes of our dataset
        self.samples = samples
        self.transform = transform
        self.target_transform = None  # target transform is not applicable to our uses

    def __getitem__(self, index):  # this is what I'll actually change!!
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is the path of the target segmented-image.
        """
        sample_path, target_path = self.samples[index]
        sample = self.loader(sample_path)
        target = self.loader(target_path)

        if self.transform is not None:
            sample = self.transform(sample)
            target = self.transform(target)

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


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


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


class DeepDriveDataset(DatasetFolder):
    """A loader for the deep drive dataset for semantic segmentation of images. Loads bdd100k images as well as bdd100k
    drivable maps as labels.
    Args:
        image_dir (string): Root directory path of images.
        semantic_image_labels_dir (string): Root directory path of image-labels
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        loader (callable, optional): A function to load an image given its path.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """
    def __init__(self, image_dir, semantic_image_labels_dir, transform=None, loader=default_loader):
        super(DeepDriveDataset, self).__init__(image_dir, semantic_image_labels_dir, loader = loader,
                                         extensions = IMG_EXTENSIONS, transform=transform)
        self.imgs = self.samples

        
# ===================================================================================================
# ================================ Call the Loader on this Device ===================================
# ===================================================================================================

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
    train_dataset = DeepDriveDataset(image_dir + "/train", label_dir + "/train", transform = transforms.CenterCrop(720))
    test_dataset = DeepDriveDataset(image_dir + "/test", label_dir + "/test", transform = transforms.CenterCrop(720))

    return train_dataset, test_dataset

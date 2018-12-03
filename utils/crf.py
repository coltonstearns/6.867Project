import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax
from PIL import Image

from utils.data_loading import load_datasets
import torch

def crf_batch_postprocessing(image_batch, output_batch, num_classes):
    """
    fcn_output (torch.tensor): a 3D pytorch log-softmax encoded tensor. The dimensions must be (k, num_classes = 3, width, height).
    original_image (torch.tensor): an RBG image represented as a torch tensor with dimensions (k, 3, width, height)
    """

    original_type = type(output_batch)
    use_cuda = output_batch.is_cuda

    images = image_batch.cpu().data.numpy().astype(np.uint8)  # (k, 3, width, height)

    # get rid of softmax log output
    output_batch = np.e**(output_batch.cpu().data.numpy())   # (k, num_classes = 3, width, height)
    
    # run CRF on each image and output from the batch
    processed_batch = np.empty(output_batch.shape)
    for i in range(len(images)):
        processed_batch[i, :, :, :] = crf_postprocessing(images[i, :, :, :], output_batch[i, :, :, :], num_classes)

    return original_type(processed_batch).cuda() if use_cuda else original_type(processed_batch)

"""
Takes in a pytorch tensor, runs a dense CRF postprocessing on it, and returns
a processed, equivalent pytorch tensor.

Args:
    fcn_output (np.array): a 3D numpy array that's one output of our FCN. The dimensions must be (num_classes = 3, width, height).
    original_image (np.array): the corresponding RBG image to our output; has with dimensions (3, width, height)

Return:
    np.array: the new probability of each class for every pixel of the input. Output is formatted as
        a 3D log-softmax numpy array, with dimensions (num_classes = 3, width, height)
"""
def crf_postprocessing(original_image, fcn_output, num_classes):
    # create our CRF model
    dense_crf = dcrf.DenseCRF2D(1280, 720, num_classes)  # width, height, nlabels

    # convert our softmax output into the unary PDF of our model
    unary_potentials = unary_from_softmax(fcn_output)
    dense_crf.setUnaryEnergy(unary_potentials.astype(np.float32))

    # ===================================== Hyperparameters ========================================
    # define a compatability matrix for misclassifying objects
    # this matrix says it is 5x worse to classify drivable as not drivable
    # vs current lane as other lane
    if num_classes == 3:
        compatability_matrix = np.array([[0., 3., 3.],
                                        [3., 0., 3.],
                                        [3., 3., 0.]]).astype(np.float32)
        location_xy_stdev = 3
        color_xy_stdev = 80
        color_rgb_stdev = 15
        num_smoothing_iters = 1

    elif num_classes == 2:
        compatability_matrix = np.array([[0., 1.],
                                     [1., 0.]]).astype(np.float32)
        location_xy_stdev = 3
        color_xy_stdev = 30
        color_rgb_stdev = 5
        num_smoothing_iters = 1

    else:
        assert(False), "CRF postprocessing only supports 2 and 3 classes"
    
    # ==============================================================================================

    # add pairwise connections for smoothing pixel location in CRF
    dense_crf.addPairwiseGaussian(sxy = location_xy_stdev, compat = compatability_matrix)

    # add pairwise connections for Color similarity in CRF
    dense_crf.addPairwiseBilateral(sxy = color_xy_stdev, srgb = color_rgb_stdev, rgbim = original_image.T.copy(order = 'C'), compat = compatability_matrix*num_classes)

    # run 5 iterations of the dense CRF filtering
    Q = dense_crf.inference(num_smoothing_iters)
    log_probabilities = np.log(np.array(Q).reshape((num_classes, 1280, 720)))

    # convert our output to be the same type as the original pytorch tensor
    return log_probabilities

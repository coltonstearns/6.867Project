import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax
from PIL import Image


from data_loading import load_datasets
import torch

"""
Takes in a pytorch tensor, runs a dense CRF postprocessing on it, and returns
a processed, equivalent pytorch tensor.

Args:
    fcn_output (torch.tensor): a 3D pytorch log-softmax encoded tensor. The dimensions must be (num_classes = 3, width, height).
    original_image (np.array): an RBG image-array, e.g. im.dtype == np.uint8 and im.shape == (height, width, 3)

Return:
    torch.tensor: the new probability of each class for every pixel of the input. Output is formatted as
        a 3D pytorch log-softmax tensor, with dimensions (num_classes = 3, width, height)
    np.array: a 2d array the size of the original image, with a[i, j] = most_probably_class
"""
def crf_postprocessing(original_image, fcn_output):
    # create our CRF model
    dense_crf = dcrf.DenseCRF2D(1280, 720, 3)  # width, height, nlabels

    # convert our softmax output into the unary PDF of our model
    softmax_output = np.e**(fcn_output.numpy())  # raise to power of e to undo log of softmax
    unary_potentials = unary_from_softmax(softmax_output)
    dense_crf.setUnaryEnergy(unary_potentials.astype(np.float32))

    # ===================================== Hyperparameters ========================================
    # define a compatability matrix for misclassifying objects
    # this matrix says it is 5x worse to classify drivable as not drivable
    # vs current lane as other lane
    compatability_matrix = np.array([[0., 1., 1.],
                                     [1., 0., 5.],
                                     [1., 5., 0.]]).astype(np.float32)
    location_xy_stdev = 3
    color_xy_stdev = 80
    color_rgb_stdev = 13
    # ==============================================================================================

    # add pairwise connections for smoothing pixel location in CRF
    dense_crf.addPairwiseGaussian(sxy = location_xy_stdev, compat = compatability_matrix)

    # add pairwise connections for Color similarity in CRF
    dense_crf.addPairwiseBilateral(sxy = color_xy_stdev, srgb = color_rgb_stdev, rgbim = original_image, compat = compatability_matrix*3)

    # run 5 iterations of the dense CRF filtering
    Q = dense_crf.inference(5)
    log_probabilities = np.log(np.array(Q).reshape((3, 1280, 720)))
    MAP_prediction = np.argmax(Q, axis=0).reshape((1280,720))

    # convert our output to be the same type as the original pytorch tensor
    original_type = type(fcn_output)
    return original_type(log_probabilities), MAP_prediction



# ================================================================================================ #
# ======================== Test to make sure CRF works (code below is crap) ====================== #
# ================================================================================================ #
if __name__ == "__main__":
    train_dataset, _ = load_datasets()
    random_index = np.random.randint(0, len(train_dataset))
    sample, target = train_dataset.__getitem__(random_index)

    # turn tensor into arraylike rgb image that can be processed by C code
    rbg_image = sample.numpy().T.astype(np.uint8)
    image = Image.fromarray(rbg_image)
    image.show()
    image.save("real_image.jpg")
    rbg_image = rbg_image.copy(order = 'C')


    # turn perfect target into noisy target
    target = target.numpy()
    target = np.array([x for x in (np.where(target == i, 1, 0) for i in range(3))])

    # show actual target
    image = Image.fromarray((target.T * 200).astype(np.uint8))
    image.show()
    image.save("real_target.jpg")

    # perturb target with noise
    target = target.tolist()
    for i in range(len(target[0])):
        if i % 100 == 0:
            print(str(i) + "out of " + str(len(target[0])))

        for j in range(len(target[0][0])):
            random_noise = np.random.rand() / 2
            category = np.argmax([target[0][i][j], target[1][i][j], target[2][i][j]])
            for k in range(3):
                if category == k:
                    target[k][i][j] -= random_noise
                else:
                    target[k][i][j] += random_noise/2
    target = np.array(target)

    # show noisy target
    image = Image.fromarray((np.floor(target.T * 200)).astype(np.uint8))
    image.show()
    image.save("noisy_target.jpg")


    # convert into expected format
    target = np.log(target)
    target = torch.tensor(target)
    image.save("crf_target.jpg")


    processed, map = crf_postprocessing(rbg_image, target)
    assert np.argmax(processed.numpy(), axis = 0).all() == map.all()

    processed = np.e**(processed.numpy()).T  # reformat into 3D array
    image = Image.fromarray((np.floor(processed * 200)).astype(np.uint8))
    image.show()

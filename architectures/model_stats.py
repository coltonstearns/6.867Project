import numpy as np
import matplotlib
from matplotlib import pyplot as plt

"""
Maintains information about the model
"""
class ModelStats:
    def __init__(self, num_classes = 2):
        self.loss = []
        self.accuracy = []
        self.confusion = np.zeros((num_classes, num_classes))
        self.per_class_loss = [] #not used
        self.per_class_accuracy = []
        self.num_classes = num_classes

    #def graph_accuracy_with_time(self, save_to):
    #def graph_per_class_accuracy_with_time(self, save_to):

    #def graph_loss_with_time(self, save_to):

    def print_summary(self):
        print('\n--------------------------------------------------------------')
        try:
            loss = self.loss[-1]
        except: 
            print("Loss information Not Available")
            loss = 0
        try:
            accuracy = self.accuracy[-1]
        except:
            print("Accuracy information Not Available")
            accuracy = 0

        print('\nAverage loss: {:.4f}, Accuracy: {:.0f}%\n'.format(
             loss, accuracy))

        acc_dict = self.confusion
        
        if acc_dict.shape[0] == 3:
            print('\n Class |  Samples  | % Class 0 | % Class 1 | %Class 2 |')
            for class_type in range(len(acc_dict)):
                total = acc_dict[class_type][0] + acc_dict[class_type][1] + acc_dict[class_type][2]
                if total == 0: 
                    print(' {}     |     0     |    n/a    |    n/a    |    n/a    |'.format(class_type))
                else: 
                    print(' {}     | {} |   {:.2f}   |   {:.2f}   |   {:.2f}   |'.format(class_type, total, 100*acc_dict[class_type][0]/total, 
                    100*acc_dict[class_type][1]/total, 100*acc_dict[class_type][2]/total))

        else:
            print('\n Class |    Samples    | % Class 0 | % Class 1 |')
            for class_type in range(len(acc_dict)):
                total = acc_dict[class_type][0] + acc_dict[class_type][1]
                if total == 0:
                    print(' {}     |       0       |    n/a    |    n/a    |'.format(class_type))
                else:
                    print(' {}     | {} |   {:.2f}   |   {:.2f}   |'.format(class_type, total, 100*acc_dict[class_type][0]/total,
                    100*acc_dict[class_type][1]/total))
                
        print('--------------------------------------------------------------')

def get_per_class_loss(loss, target, loss_vec):
    for i in range(len(loss_vec)):
        mask = target.eq(i)
        total_num = torch.sum(mask)
        if(total_num.item() > 0):
            loss_vec[i] = torch.sum(torch.masked_select(loss, mask))/total_num.item()
        else:
            loss_vec[i] = torch.sum(torch.masked_select(loss, mask))


def get_per_class_accuracy(pred, target, acc_dict):
    """
    Takes in a batch of predictions and targets as pytorch tensors, as well as an accuracy matrix. Mutates the 
    accuracy matrix by summing the prediction-target pixel-wise accuracies in each entry.

    Args:
        pred (torch.tensor): 3D tensor. Axis 0 has each image output, axes 1 and 2 define the predicted output; each entry
        will be 0, 1, or 2 depending on the class
        target (torch.tensor): same as pred, but the correct target
        acc_dict (2d list): a 3 by 3 matrix containing accuracies of pixel ratings. acc_dict[0][1] indicates pixels that the 
            prediction labeled class 0, but the target labeled class 1
    """
    prediction_numpy, target_numpy = pred.cpu().numpy(), target.cpu().numpy()

    def prediction_error(predicted_label, target_label):
        """
        To get the number of times our output label is 0, but the target is 2, we would call
        prediction_error(predicted_label = 0, target_label = 2)
        """
        return len(np.where(np.logical_not(np.logical_or(prediction_numpy - predicted_label, target_numpy - target_label)))[0])

    for i in range(len(acc_dict)):
        for j in range(len(acc_dict[i])):
            acc_dict[j][i] += prediction_error(i, j)

  


def visualize_output(pred, target, image):
    """
    Args:
        pred (torch.tensor): 3D tensor. Axis 0 has each image output, axes 1 and 2 define the predicted output; each entry
            will be 0, 1, or 2 depending on the class
        target (torch.tensor): same as pred, but the correct target
    """

    prediction_numpy, target_numpy, raw_image = pred.cpu().data.numpy()[0,:,:], target.cpu().data.numpy()[0,:,:], image.cpu().data.numpy()[0,:,:,:]
    total_image = (np.hstack((prediction_numpy, target_numpy))*100)
    total_image = np.array(total_image, dtype = np.uint8).T
    real_image = np.array(raw_image, dtype = np.uint8).T

    # show actual target
    image = Image.fromarray(total_image, "L")
    image2 = Image.fromarray(real_image)
    image.show()
    image2.show()



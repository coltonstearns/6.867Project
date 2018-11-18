import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# import our pytorch formatted datasets from data_loading.py script
from data_loading import load_datasets


'''
Every image is 1280 x 720 pixels, but we will input training chunks of 720 x 720 pixels.
'''
class FCN(nn.Module):  # inherit from base class torch.nn.Module
    def __init__(self, save_dir):
        super(FCN, self).__init__()  # initialize Module characteristics
        
        self.save_dir = save_dir
        # goes from (720 x 720 x 3) to (353 x 353 x 8)
        self.conv1 = nn.Conv2d(3, 8, kernel_size=6, stride = 2, padding = 0, dilation = 3)

        # goes from (353 x 353 x 8) to (175 x 175 x 16)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5, stride = 2, padding = 0, dilation = 1)

        # goes from (175 x 175 x 16) to (36 x 36 x 8)
        self.conv3 = nn.Conv2d(16, 8, kernel_size=4, stride = 5, padding = 3, dilation = 1)

        # reduce tensor to simple 3D value; this will be our result
        self.classify_layer = nn.Conv2d(8, 3, kernel_size = 1, stride = 1, padding = 0)

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

    

def train(model, device, train_loader, optimizer, epoch, log_spacing = 7200, save_spacing = 100, per_class = False):
    """
    Args:
        model (nn.Module): the FCN pytorch model
        device (torch.device): represents if we are running this on GPU or CPU
        optimizer (torch.optim): the optimization object that trains the network. Ex: torch.optim.Adam(modle.parameters())
        train_loader (torch.utils.data.DataLoader): the pytorch object that contains all training data and targets
        epoch (int): the epoch number we are on
        log_spacing (int): prints training statistics to display every <log_spacing> batches
        save_spacing (int): saves most recent version of model every <save_spacing> batches
        per_class (boolean): true if want class-level statistics printed. false otherwise
    """
    model.train()  # puts it in training mode
    sum_num_correct = 0
    sum_loss = 0
    num_batches_since_log = 0
    loss_func = nn.CrossEntropyLoss(reduction = "elementwise_mean")
    acc_dict = [[0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0]]

    # run through data in batches, train network on each batch
    for batch_idx, (data, target) in tqdm(enumerate(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()  # reset gradient to 0 (so doesn't accumulate)
        output = model(data)  # runs batch through the model
        loss = loss_func(output, target)  # compute loss of output

        # convert into 1 channel image with predicted class values 
        pred = torch.argmax(output, dim = 1, keepdim=False)
        assert(pred.shape == (train_loader.batch_size, 1280, 720)), "got incorrect shape of: " + str(pred.shape)

        correct_pixels = pred.eq(target.view_as(pred)).sum().item()
        sum_num_correct += correct_pixels

        sum_loss += loss.item()
        loss.backward()  # take loss object and calculate gradient; updates optimizer
        optimizer.step()  # update model parameters with loss gradient

        #update per-class accuracies
        if(per_class):
            get_per_class_accuracy(pred, target, acc_dict)

        if batch_idx % log_spacing == 0:
            print_log(correct_pixels, sum_loss, batch_idx + 1, train_loader.batch_size, "Training Set", per_class, acc_dict)

        if batch_idx % save_spacing == 0:
            print('Saving Model to: ' + str(model.save_dir))
            model.save()


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

    prediction_numpy, target_numpy = pred.data.numpy(), target.data.numpy()

    def prediction_error(predicted_label, target_label):
        """
        To get the number of times our output label is 0, but the target is 2, we would call
        prediction_error(predicted_label = 0, target_label = 2)
        """
        return len(np.where(np.logical_not(np.logical_or(prediction_numpy - predicted_label, target_numpy - target_label)))[0])

    for i in range(len(acc_dict)):
        for j in range(len(acc_dict[i])):
            acc_dict[i][j] += prediction_error(i, j)


def test(model, device, test_loader, dataset_name="Test set", iters_per_log = 7000):
    model.eval()
    test_loss = 0
    correct = 0
    loss_func = nn.CrossEntropyLoss()
    batches_done = 0
    #initialize per class accuracy
    # Target 1:  
    # Target 2:
    # Target 3: 
    acc_dict = [[0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0]]

    with torch.no_grad():
        for batch_idx, (data, target) in tqdm(enumerate(test_loader)):  # runs through trainer
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_func(output, target).item()

            ##convert into 1 channel image with values 
            pred = torch.argmax(output, dim = 1, keepdim=False)
            assert(pred.shape == (test_loader.batch_size, 1280, 720)), "got incorrect shape of: " + str(pred.shape)

            correct_pixels = pred.eq(target.view_as(pred)).sum().item()
            correct += correct_pixels
            
            get_per_class_accuracy(pred, target, acc_dict)
            #verify_pixels = get_per_class_accuracy(pred, target, acc_dict)
            #assert(verify_pixels == correct_pixels)
            batches_done += 1

            if(batches_done % iters_per_log == 0):
                print_log(correct, test_loss, batches_done, test_loader.batch_size, dataset_name, True, acc_dict)

        print_log(correct, test_loss, len(test_loader.dataset), 1, dataset_name, True, acc_dict)       

def print_log(correct_pixels, loss, num_samples, batch_size, name, use_acc_dict = False, acc_dict = None):

    loss = loss/(num_samples*batch_size)
    total_samples = num_samples*batch_size*1280*720
    print('\n--------------------------------------------------------------')
    print('\n{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        name, loss, correct_pixels, total_samples,
        100. * correct_pixels / total_samples))

    if use_acc_dict:
        print('\n Class |  Samples  | % Class 0 | % Class 1 | %Class 2 |')
        for class_type in range(len(acc_dict)):
            total = acc_dict[class_type][0] + acc_dict[class_type][1] + acc_dict[class_type][2]
            if total == 0: 
                print(' {}     |     0     |    n/a    |    n/a    |    n/a    |'.format(class_type))
            else: 
                print(' {}     | {} |   {}   |   {}   |   {}   |'.format(class_type, total, 100*acc_dict[class_type][0]/total, 
                    100*acc_dict[class_type][1]/total, 100*acc_dict[class_type][2]/total))
    print('--------------------------------------------------------------')


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from PIL import Image

# our own code imports
from utils.crf import crf_batch_postprocessing

class SegmentationTrainer:
    """
    Class to train segmentation model
    """
    def __init__(self, model, device, train_loader, test_loader, optimizer, data_stats,
                 num_classes = 3, log_spacing = 100, save_spacing = 100, per_class = False):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.num_classes = num_classes
        self.optimizer = optimizer
        self.log_spacing = log_spacing
        self.save_spacing = save_spacing
        self.per_class = per_class
        self.training_confusion = np.zeros((num_classes, num_classes))
        self.test_confusion = np.zeros((num_classes, num_classes))
        self.data_statistics = data_stats

    def train(self, epoch, start_index = 0):
        """
        Args:
            model (nn.Module): the FCN pytorch model
            device (torch.device): represents if we are running this on GPU or CPU
            optimizer (torch.optim): the optimization object that trains the network. Ex: torch.optim.Adam(modle.parameters())
            train_loader (torch.utils.data.DataLoader): the pytorch object that contains all training data and targets
            epoch (int): the epoch number we are on
            log_spacing (int): prints training statistics to display every <lo 
            
           _spacing> batches
            save_spacing (int): saves most recent version of model every <save_spacing> batches
            per_class (boolean): true if want class-level statistics printed. false otherwise
        """
        self.model.train()  # puts it in training mode
        sum_num_correct = 0
        sum_loss = 0
        num_batches_since_log = 0
        loss_func = nn.CrossEntropyLoss(reduction = "none")
        # run through data in batches, train network on each batch
        for batch_idx, (data, target) in tqdm(enumerate(self.train_loader)):
            if(batch_idx < start_index): continue
            loss_vec = torch.zeros((self.num_classes), dtype = torch.float32)
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()  # reset gradient to 0 (so doesn't accumulate)
            output = self.model(data)  # runs batch through the model
            loss = loss_func(output, target)  # compute loss of output

            # convert into 1 channel image with predicted class values 
            pred = torch.argmax(output, dim = 1, keepdim = False)
            assert(pred.shape == (self.train_loader.batch_size, 1280, 720)), "got incorrect shape of: " + str(pred.shape)

            correct_pixels = pred.eq(target.view_as(pred)).sum().item()
            sum_num_correct += correct_pixels

            get_per_class_loss(loss, target, loss_vec)
            loss = torch.sum(loss_vec)

            sum_loss += loss.item()
            loss.backward()  # take loss object and calculate gradient; updates optimizer
            self.optimizer.step()  # update model parameters with loss gradient

            #update per-class accuracies
            if(self.per_class):
                get_per_class_accuracy(pred, target, self.training_confusion)

            if batch_idx % self.log_spacing == 0:
                print("Loss Vec: {}".format(loss_vec))
                print_log(sum_num_correct, sum_loss, batch_idx + 1, self.train_loader.batch_size, 
                          "Training Set", self.per_class, self.training_confusion)

            if batch_idx % self.save_spacing == 0:
                print('Saving Model to: ' + str(self.model.save_dir))
                self.model.save()

    def test(self, dataset_name= "Test set", use_crf = True, iters_per_log = 100, visualize = False, use_prior = True):
        self.model.eval()
        test_loss = 0
        correct = 0
        loss_func = nn.CrossEntropyLoss()
        batches_done = 0

        with torch.no_grad():
            # prior = torch.ones(self.data_statistics.get_distribution().shape) - self.data_statistics.get_distribution()
            # calculate an UNBIASED prior
            prior = self.data_statistics.get_distribution().to(self.device)
            for i in range(self.num_classes):
                prior[i] = prior[i] / (torch.mean(prior[i]))  #  scales relative probs to have mean of 1
            normalization = torch.sum(prior, dim = 0)  # sum along classes
            prior /= normalization

            for batch_idx, (data, target) in tqdm(enumerate(self.test_loader)):  # runs through trainer
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                if use_prior:
                    alpha = .75
                    for i in range(len(output)):  # could be multiple images in output batch
                        output[i] = alpha * output[i] + (1-alpha) * prior

                if use_crf:
                    output = crf_batch_postprocessing(data, output, self.num_classes)

                test_loss += loss_func(output, target).item()

                #convert into 1 channel image with values 
                pred = torch.argmax(output, dim = 1, keepdim = False)
                assert(pred.shape == (self.test_loader.batch_size, 1280, 720)), "got incorrect shape of: " + str(pred.shape)

                correct_pixels = pred.eq(target.view_as(pred)).sum().item()
                correct += correct_pixels
                
                get_per_class_accuracy(pred, target, self.test_confusion)
                batches_done += 1

                if(batches_done % self.log_spacing == 0):
                    print_log(correct, test_loss, batches_done, self.test_loader.batch_size, dataset_name, True, self.test_confusion)
                    if visualize:
                        visualize_output(pred, target)

            print_log(correct, test_loss, len(self.test_loader.dataset), 1, dataset_name, True, self.training_confusion)    
            
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

  
def print_log(correct_pixels, loss, num_samples, batch_size, name, use_acc_dict = False, acc_dict = None):
    loss = loss/(num_samples*batch_size)
    total_samples = num_samples*batch_size*1280*720
    print('\n--------------------------------------------------------------')
    print('\n{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        name, loss, correct_pixels, total_samples,
        100. * correct_pixels / total_samples))

    if use_acc_dict:
        if acc_dict.shape[0] == 3:
            print('\n Class |  Samples  | % Class 0 | % Class 1 | %Class 2 |')
            for class_type in range(len(acc_dict)):
                total = acc_dict[class_type][0] + acc_dict[class_type][1] + acc_dict[class_type][2]
                if total == 0: 
                    print(' {}     |     0     |    n/a    |    n/a    |    n/a    |'.format(class_type))
                else: 
                    print(' {}     | {} |   {}   |   {}   |   {}   |'.format(class_type, total, 100*acc_dict[class_type][0]/total, 
                        100*acc_dict[class_type][1]/total, 100*acc_dict[class_type][2]/total))

        else:
            print('\n Class |  Samples  | % Class 0 | % Class 1 |')
            for class_type in range(len(acc_dict)):
                total = acc_dict[class_type][0] + acc_dict[class_type][1]
                if total == 0:
                    print(' {}     |     0     |    n/a    |    n/a    |'.format(class_type))
                else:
                    print(' {}     | {} |   {}   |   {}   |'.format(class_type, total, 100*acc_dict[class_type][0]/total,
                        100*acc_dict[class_type][1]/total))
        print('--------------------------------------------------------------')


def visualize_output(pred, target):
    """
    Args:
        pred (torch.tensor): 3D tensor. Axis 0 has each image output, axes 1 and 2 define the predicted output; each entry
            will be 0, 1, or 2 depending on the class
        target (torch.tensor): same as pred, but the correct target
    """

    prediction_numpy, target_numpy = pred.cpu().data.numpy()[0,:,:], target.cpu().data.numpy()[0,:,:]
    total_image = (np.hstack((prediction_numpy, target_numpy))*100)
    total_image = np.array(total_image, dtype = np.uint8).T

    # show actual target
    image = Image.fromarray(total_image, "L")
    image.show()



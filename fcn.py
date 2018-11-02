import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# import our pytorch formatted datasets from data_loading.py script
from data_loading import load_datasets


'''
Every image is 1280 x 720 pixels, but we will input training chunks of 720 x 720 pixels.
'''
class FCN(nn.Module):  # inherit from base class torch.nn.Module
    def __init__(self):
        super(FCN, self).__init__()  # initialize Module characteristics

        # goes from (720 x 720 x 3) to (353 x 353 x 30)
        self.conv1 = nn.Conv2d(3, 30, kernel_size=6, stride = 2, padding = 0, dilation = 3)

        # goes from (353 x 353 x 30) to (175 x 175 x 90)
        self.conv2 = nn.Conv2d(30, 90, kernel_size=5, stride = 2, padding = 0, dilation = 1)

        # goes from (353 x 353 x 90) to (72 x 72 x 90)
        self.conv3 = nn.Conv2d(90, 90, kernel_size=4, stride = 5, padding = 3, dilation = 1)

        # up sample to (720 x 720 x 90)
        self.upsample1 = nn.Upsample(scale_factor = 10, mode = 'bilinear')

        # reduce tensor to simple 3D value; this will be our result
        self.classify_layer = nn.Conv3d(90, 2, kernel_size = (90, 1, 1), stride = 1, padding = 0, dilation = 0)

    """
    Defines how we perform a forward pass.
    """
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.upsample1(x))  # NOT SURE IF WE SHOULD USE RELU HERE!
        x = self.classify_layer(x)  # finish with 2d classification

        return nn.LogSoftmax()(x)  # return softmax for probability of 2d tensor

    """
    Saves the model to the given path
    """
    def save(self, path = "./model.bak"):
        torch.save(self.state_dict(), path)
    

        


# TODO: Have not modified the train function yet; this will not work!
def train(model, device, train_loader, optimizer, epoch):
    """
    Args:
        model (nn.Module): our neural network
        device (torch.device("cuda" if use_cuda else "cpu")): represents if we are running this on GPU or CPU
        train_loader (torch.utils.data.DataLoader): the pytorch object that contains all training data and targets
        epoch (int): the epoch number we are on
    """
    model.train()  # puts it in training mode
    sum_num_correct = 0
    sum_loss = 0
    num_batches_since_log = 0
    loss_func = nn.CrossEntropyLoss()
    # train_loader is torch.utils.data.DataLoader
    for batch_idx, (data, target) in enumerate(train_loader):  # runs through trainer
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_func(output, target)

        #convert into 1 channel image with values 
        pred = output.max(1, keepdim=True, axis = 2)
        assert(pred.shape() == (720, 720))


        correct_pixels = pred.eq(target.view_as(pred)).sum().item()
        sum_num_correct += correct_pixels

        sum_loss += loss.item()
        num_batches_since_log += 1
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{:05d}/{} ({:02.0f}%)]\tLoss: {:.6f}\tAccuracy: {:02.0f}%'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), sum_loss / num_batches_since_log,
                100. * sum_num_correct / (num_batches_since_log * train_loader.batch_size * 720 * 720))
            )
            sum_num_correct = 0
            sum_loss = 0
            num_batches_since_log = 0


def test(model, device, test_loader, dataset_name="Test set"):
    model.eval()
    test_loss = 0
    correct = 0
    loss_func = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_func(output, target).item() # sum up batch loss
            pred = output.max(1, keepdim=True, axis = 2) # get the index of the max log-probability
            assert(pred.shape() == (720, 720))
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\n{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        dataset_name,
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def training_procedure(train_dataset, test_dataset, daniels_photos):
    # ------------------ Training Parameters ----------------------#
    args = dict()
    args["seed"] = 73912
    args["no_cuda"] = False
    args["batch_size"] = 32
    args["test-batch-size"] = 1000
    #--------------------------------------------------------------#


    # ---------------------- Hyperparameters ----------------------#
    params = dict()
    params["epochs"] = 10
    params["lr"] = 0.1
    #--------------------------------------------------------------#


    # ------------- Sets up pseudo-random number generator and GPU ----------#
    torch.manual_seed(args["seed"])
    use_cuda = not args["no_cuda"] and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    # -----------------------------------------------------------------------#


    # Wrap training and testing data into pytorch DataLoader for easy manipulation
    train_loader = torch.utils.data.DataLoader(train_dataset,
                        batch_size=args["batch_size"], shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                        batch_size=args["test-batch-size"], shuffle=True, **kwargs)


    # Set up our FCN
    model = FCN().to(device)
    optimizer = optim.Adam(lr = params["lr"])  # IS ADAM ACTUALLY A GOOD OPTIMZER FOR US?

    # Train the model
    for epoch in range(1, params["epochs"] + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)


if __name__ == '__main__':
    # call data_loading.py script to load datasets
    train_dataset, test_dataset = load_datasets()

    # run our training procedure
    training_procedure(train_dataset, test_dataset)

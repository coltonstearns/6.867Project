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
        self.figure_number = 0
        self.colors = ['r', 'g', 'b']

    def start_new_graph(self):
        plt.figure(self.figure_number)
        self.figure_number+=1

    def graph_accuracy_with_time(self):
        plt.plot(self.accuracy, 'orange', label = "Total Accuracy")
    
    def graph_per_class_accuracy_with_time(self):
        accuracies = [[] for _ in range(num_classes)]
        for accuracy_list in self.per_class_accuracy:
            for i in range(self.num_classes):
                accuracies[i].append(accuracy_list[i])
        
        for i in range(self.num_classes):
            plt.plot(accuracies[i], self.colors[i], label = "Class {} Accuracy".format(i))
        

    def graph_loss_with_time(self):
        plt.plot(self.loss, 'purple', label = "Total Loss")

    def save_plot(self, title):
        plt.savefig(title + ".png")

    def show(self):
        plt.show()

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




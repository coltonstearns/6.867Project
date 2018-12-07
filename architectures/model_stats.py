import numpy as np

"""
Maintains information about the model
"""
class ModelStats:
    def __init__(self, num_classes = 2):
        self.loss = []
        self.accuracy = []
        self.confusion = np.zeros((num_classes, num_classes))
        self.per_class_loss = []
        self.per_class_accuracy = []
        self.num_classes = num_classes




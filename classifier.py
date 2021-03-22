from functions import *
import numpy as np

class OneLayerClassifier:

    def __init__(self, dims, num_labels=10):
        self.W = np.random.normal(0, 0.1, (num_labels, dims))
        self.b = np.random.normal(0, 0.1, (num_labels, 1))

    def evaluate_classifier(self, data):
        s = np.dot(self.W, data) + self.b
        probabilities = softmax(s)
        return probabilities


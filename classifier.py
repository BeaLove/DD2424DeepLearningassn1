from functions import *
import numpy as np

class OneLayerClassifier:

    def __init__(self, dims, lr = 0.005, num_labels=10):

        np.random.seed(0)
        self.W = np.random.normal(0, 0.1, (num_labels, dims))
        self.b = np.random.normal(0, 0.1, (num_labels, 1))
        self.targets = 0
        self.num_labels = num_labels
        self.lr = lr

    def evaluateClassifier(self, data):
        s = np.dot(self.W, data) + self.b
        probabilities = softmax(s)
        return probabilities

    def update_weights(self, loss, data):
        #lr*data*error
        self.W -= self.lr * np.dot(data, loss)

    def computeAccuracy(self, data, labels):
        probabilities = self.evaluateClassifier(data)
        predictions = np.where(probabilities==max(probabilities), 1, 0)
        accuracy = abs(predictions - labels)/data.shape[0]
        return accuracy



one = OneLayerClassifier(data.shape[0], lr=0.005, num_labels=10)

#probs = one.evaluateClassifier(data[:, :100])
lamda = 10
cost = ComputeCost(train_data, train_targets, one.W, one.b, lamda)

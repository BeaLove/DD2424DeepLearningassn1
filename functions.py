import numpy as np


class OneLayerClassifier:

    def __init__(self, dims, lr=0.001, lamda=0., num_labels=10):

        np.random.seed(0)
        self.W = np.random.normal(0, 0.1, (num_labels, dims))
        self.b = np.random.normal(0, 0.1, (num_labels, 1))
        self.targets = 0
        self.num_labels = num_labels
        self.lr = lr
        self.lamda = lamda

    def evaluateClassifier(self, data):
        s = np.dot(self.W, data) + self.b
        probabilities = softmax(s)
        return probabilities

    def update_weights(self, loss, data):
        # lr*data*error
        self.W -= self.lr * np.dot(data, loss)

    def computeAccuracy(self, data, labels):
        probabilities = self.evaluateClassifier(data)
        predictions = np.where(probabilities == max(probabilities), 1, 0)
        accuracy = np.sum(abs(predictions - labels)) / data.shape[1]
        return accuracy*100

    def miniBatchGD(self, data, labels, val_data, val_labels, epochs=20, batch_size=100):
        mini_batches = []
        target_batches = []
        for i in range(0, batch_size, len(data)):
            mini_batches.append(data[:, i:i + batch_size])
            target_batches.append(labels[:, i:i + batch_size])
        cost_batch = []
        cost_train = []
        cost_val = []
        for epoch in range(epochs):
            print("epoch ", epoch)
            for batch in range(len(mini_batches)):
                y_hat = self.evaluateClassifier(mini_batches[batch])
                grad, grad_b = self.ComputeGradients(mini_batches[batch], target_batches[batch], y_hat)
                cost = self.ComputeCost(mini_batches[batch], target_batches[batch])
                cost_batch.append(cost)
                self.W += self.lr * grad
                self.b += self.lr * grad_b
            print("accuracy on training data" , self.computeAccuracy(data, labels))
            print("training data cost epoch as mean of minibatches", np.mean(cost_batch))
            training_cost = self.ComputeCost(data, labels)
            cost_train.append(training_cost)
            val_cost = self.ComputeCost(val_data, val_labels)
            cost_val.append(val_cost)
            print("cost on training data after epoch ", training_cost)
            print("cost on validation data ", val_cost)
        return cost_train, cost_val

    def ComputeGradients(self, X, Y, P):
        batch_size = X.shape[1]
        #print("batch size ", batch_size)
        # p = one.evaluateClassifier(X)
        G = -(Y - P)
        dot = np.dot(G, np.transpose(X))
        d_w1 = 1 / batch_size * dot
        grad_w = d_w1 + 2 * self.lamda * self.W
        grad_b = 1 / batch_size * np.sum(G, axis=1)
        grad_b = grad_b.reshape(-1,1)
        return grad_w, grad_b

    def ComputeCost(self, X, Y):
        batch_size = X.shape[1]
        p = self.evaluateClassifier(X)
        # log_in = np.dot(np.transpose(Y), p)
        sum = 0
        y = Y.T
        for i in range(Y.shape[1]):
            sum += -np.log(np.dot(y[i, :], p[:, i]))
            # log_sum = np.sum(-np.log(log_in))
        loss_part1 = sum / batch_size
            # loss_part1 = (1/batch_size) * log_sum
        loss_part2 = self.lamda * np.sum(self.W ** 2)
        cost = loss_part1 + loss_part2
        return cost


def softmax(x):
    """ Standard definition of the softmax function """
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def LoadBatch(filename):
    # """ Copied from the dataset website """
    import pickle
    with open('Dataset/' + filename, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict




'''
def ComputeGradsNum(X, Y, P, W, b, lamda, h, classifier):
    # """ Converted from matlab code """
    no = W.shape[0]
    d = X.shape[0]

    grad_W = np.zeros(W.shape);
    grad_b = np.zeros((no, 1));

    c = ComputeCost(X, Y, W, b, lamda);

    for i in range(len(b)):
        b_try = np.array(b)
        b_try[i] += h
        c2 = ComputeCost(X, Y, W, b_try, lamda)
        grad_b[i] = (c2 - c) / h

    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W_try = np.array(W)
            W_try[i, j] += h
            c2 = ComputeCost(X, Y, W_try, b, lamda)
            grad_W[i, j] = (c2 - c) / h

    return [grad_W, grad_b]


def ComputeGradsNumSlow(X, Y, P, W, b, lamda, h):
    """ Converted from matlab code """
    no = W.shape[0]
    d = X.shape[0]

    grad_W = np.zeros(W.shape);
    grad_b = np.zeros((no, 1));

    for i in range(len(b)):
        b_try = np.array(b)
        b_try[i] -= h
        c1 = ComputeCost(X, Y, W, b_try, lamda)

        b_try = np.array(b)
        b_try[i] += h
        c2 = ComputeCost(X, Y, W, b_try, lamda)

        grad_b[i] = (c2 - c1) / (2 * h)

    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W_try = np.array(W)
            W_try[i, j] -= h
            c1 = ComputeCost(X, Y, W_try, b, lamda)

            W_try = np.array(W)
            W_try[i, j] += h
            c2 = ComputeCost(X, Y, W_try, b, lamda)

            grad_W[i, j] = (c2 - c1) / (2 * h)

    return [grad_W, grad_b]'''


def montage(W):
    """ Display the image for each label in W """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 5)
    for i in range(2):
        for j in range(5):
            im = W[i * 5 + j, :].reshape(32, 32, 3, order='F')
            sim = (im - np.min(im[:])) / (np.max(im[:]) - np.min(im[:]))
            sim = sim.transpose(1, 0, 2)
            ax[i][j].imshow(sim, interpolation='nearest')
            ax[i][j].set_title("y=" + str(5 * i + j))
            ax[i][j].axis('off')
    plt.show()


def save_as_mat(data, name="model"):
    """ Used to transfer a python model to matlab """
    import scipy.io as sio

    # sio.savemat(name'.mat',{name:b})
    sio.savemat(name + ".mat", {name: data})


def get_mean_std(train_data):
    mean = np.mean(train_data, axis=0)
    std = np.std(train_data, axis=0)
    return mean, std


def PreProcessing(data, labels, mean, std):
    print("in preprocessing")
    scaled = data - mean
    normalized = scaled / std
    targets = np.transpose(np.eye(10)[labels])

    return normalized, targets

'''
def check_gradients(data, targets, lamda, classifier):
    P = classifier.evaluateClassifier(data)
    gradient, b_gradient = ComputeGradients(data, targets, P, classifier.W, lamda)
    gradient_num, gradient_b_num = ComputeGradsNum(data, targets, P, classifier.W, classifier.b, lamda, h=1e-6)
    gradient_num_slow, gradient_b_num_slow = ComputeGradsNumSlow(data, targets, P, classifier.W, classifier.b, lamda,
                                                                 h=1e-6)
    print("compare backprop function to numerical calculation: ")
    diff = gradient - gradient_num
    diff_b = b_gradient - gradient_b_num
    if np.all(abs(diff) <= 1e-6):
        print("grad equivalent")
    else:
        print("grad incorrect")
        print(np.mean(diff))
    if np.all(abs(diff_b) <= 1e-6):
        print("grad b good")
    else:
        print("grad b bad")
    print("compare backprop function to slow numerical calculation: ")
    diff = gradient - gradient_num_slow
    if np.all(abs(diff) <= 1e-6):
        print("equivalent")
    else:
        print("incorrect")
    return gradient, b_gradient
    '''


dict = LoadBatch("data_batch_1")
data = dict[b'data'].transpose()  # we need data in columns not rows
labels = dict[b'labels']
mean, std = get_mean_std(data)
train_data, train_targets = PreProcessing(data, labels, mean, std)
dict_val = LoadBatch("data_batch_2")
val_data = dict_val[b'data'].transpose()
val_labels = dict_val[b'labels']
val_data, val_targets = PreProcessing(val_data, val_labels, mean, std)
dict_test = LoadBatch("test_batch")
test_data = dict_test[b'data'].transpose()
test_labels = dict_test[b'labels']
test_data, test_targets = PreProcessing(test_data, test_labels, mean, std)

one = OneLayerClassifier(data.shape[0], lr=0.001, lamda=0.1, num_labels=10)

# probs = one.evaluateClassifier(data[:, :100])
# cost = ComputeCost(train_data, train_targets, one.W, one.b, lamda)
# in_data = train_data[:, :100]
# train_targets =  train_targets[:,:100]

#check_gradients(train_data[:, :100], train_targets[:, :100], 1, one)
one.miniBatchGD(train_data, train_targets, val_data, val_targets, epochs=20, batch_size=100)

print("out")

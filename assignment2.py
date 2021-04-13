import numpy as np
import matplotlib.pyplot as plt


class TwoLayerClassifier:

    def __init__(self, input_dims, hidden_dims=50, lr=0.001, lamda=0., num_labels=10):

        np.random.seed(245)
        std1 = 1/np.sqrt(input_dims)
        std2 = 1/np.sqrt(hidden_dims)
        #w1 size m x d, w2 size k x m
        self.W1 = np.random.normal(0, std1, (hidden_dims, input_dims))
        self.W2 = np.random.normal(0, std2, (num_labels, hidden_dims))
        self.b1 = np.zeros((hidden_dims, 1))
        self.b2 = np.zeros((num_labels, 1))
        self.targets = 0
        self.num_labels = num_labels
        self.lr = lr
        self.lamda = lamda

    def evaluateClassifier(self, data):
        s1 = np.dot(self.W1, data) + self.b1
        h = np.where(s1 > 0, s1, 0)
        s = np.dot(self.W2, h) + self.b2
        probabilities = softmax(s)
        return h, probabilities

    def update_weights(self, loss, data):
        # lr*data*error
        self.W -= self.lr * np.dot(data, loss)

    def computeAccuracy(self, data, labels):
        h, probabilities = self.evaluateClassifier(data)
        pred_labels = [np.argmax(probabilities[:, i]) for i in range(probabilities.shape[1])]
        #pred_labels = np.array([np.argmax(probabilities[:,i]) for i in range(probabilities.shape[1])])
        labels = np.asarray(labels)
        pred_labels = np.asarray(pred_labels)
        correct = np.where(pred_labels == labels, 1, 0)
        #sum = np.sum(correct)
        accuracy = np.sum(correct) / data.shape[1]
        return accuracy*100

    def miniBatchGD(self, data, targets, val_data, val_targets, train_labels, val_labels, loops=3, epochs_per_loop=16, batch_size=100, n_s=500):
        #mini_batches = []
        target_batches = []
        #cost_batch = []
        cost_train = []
        cost_val = []
        accuracy_train = []
        accuracy_val = []
        t=1
        etas = []
        mini_batches = []
        for i in range(0, data.shape[1], batch_size):
            mini_batches.append(data[:, i:i + batch_size])
            target_batches.append(targets[:, i:i + batch_size])
        idx = np.arange(len(mini_batches))
        for l in range(loops):
            for epoch in range(epochs_per_loop):
                print("loop, epoch", l, epoch)
                np.random.shuffle(idx)
                for i in idx:
                    #y_hat = self.evaluateClassifier(mini_batches[batch])
                    grad_w1, grad_w2, grad_b1, grad_b2 = self.ComputeGradients(mini_batches[i], target_batches[i])
                    lr = self.computeLR(t, l=l, n_s=n_s)
                    etas.append(lr)
                    self.W2 -= lr * grad_w2
                    self.b2 -= lr * grad_b2
                    self.W1 -= lr * grad_w1
                    self.b1 -= lr * grad_b1
                    #print(2*n_s)
                    t += 1
                    #print("batch done")
                training_cost = self.ComputeCost(data, targets, self.W1, self.W2, self.b1, self.b2, self.lamda)
                cost_train.append(training_cost)
                val_cost = self.ComputeCost(val_data, val_targets, self.W1, self.W2, self.b1, self.b2, self.lamda)
                cost_val.append(val_cost)
                train_accuracy = self.computeAccuracy(train_data, train_labels)
                val_accuracy = self.computeAccuracy(val_data, val_labels)
                accuracy_train.append(train_accuracy)
                accuracy_val.append(val_accuracy)
                print("cost on training data after epoch ", training_cost)
                print("cost on validation data ", val_cost)
                print("train accuracy ", train_accuracy)
                print(" validation accuracy ", val_accuracy)

        return cost_train, cost_val, accuracy_train, accuracy_val, etas

    #def minibatchGDSVM(self, data, labels, val_data, val_labels, epochs = 40, batch_size=100):

    def ComputeGradients(self, X, Y):
        #print("compute gradients")
        batch_size = X.shape[1]
        h, p = self.evaluateClassifier(X)
        G = -(Y - p)
        grad_w2 = 1/batch_size * np.dot(G, np.transpose(h)) + 2*self.lamda*self.W2
        grad_b2 = 1/batch_size * np.sum(G, axis=1)
        grad_b2 = grad_b2.reshape(-1,1)
        G_2nd = np.dot(self.W2.T, G)
        ind = np.where(h > 0, 1, 0)
        G_2nd_ind = G_2nd*ind
        grad_w1 = 1/batch_size * np.dot(G_2nd_ind, X.T) + 2*self.lamda*self.W1
        grad_b1 = 1 / batch_size * np.sum(G_2nd_ind, axis=1)
        grad_b1 = grad_b1.reshape(-1,1)
        #print("out gradients")
        return grad_w1, grad_w2, grad_b1, grad_b2

    def ComputeGradientsSVM(self, X, Y):
        batch_size = X.shape[1]
        s = np.dot(self.W, X) + self.b
        G = -(s - Y + 1)
        indicator = np.where(G > 0, 1, 0)
        G_ind = indicator * G
        grad_w = 1/batch_size * np.dot(G_ind, np.transpose(X))
        grad_b = 1/batch_size * np.sum(G_ind,axis=1).reshape(-1,1)
        return grad_w, grad_b

    def ComputeCost(self, X, Y, W1, W2, b1, b2, lam):
        batch_size = X.shape[1]
        s1 = np.dot(W1, X) + b1
        h = np.where(s1 > 0, s1, 0)
        s = np.dot(W2, h) + b2
        p = softmax(s)
        sum = 0
        y = Y.T
        for i in range(Y.shape[1]):
            sum += -np.log(np.dot(y[i, :], p[:, i]))
            # log_sum = np.sum(-np.log(log_in))
        cross_entropy = sum / batch_size
            # loss_part1 = (1/batch_size) * log_sum
        reg_1 = np.sum(W1**2)
        reg_2 = np.sum(W2**2)
        cost = cross_entropy + lam * (reg_1 + reg_2)
        return cost

    def computeLR(self, t, l, batch_size=100, eta_min=1e-5, eta_max=1e-1, n_s=500):
        #l = 1 #number of cycles
        #print("2l +1*n_s: ", (2*l + 1)*n_s)
        #print("l", l)
        if (2*l*n_s) <= t <= ((2*l)+1)*n_s:
            #print("in if", t)
            num = t - 2*l*n_s
            eta_t = eta_min + num/n_s *(eta_max-eta_min)
            return eta_t
        elif ((2*l)+1)*n_s <= t <= 2*(l+1)*n_s:
            #print("in elif", t)
            num = t - (2*l+1)*n_s
            #print("num", num)
            eta_t = eta_max - num/n_s * (eta_max-eta_min)
            #print("eta_t", eta_t)
            return eta_t
        else:
            raise ValueError("bad eta, t, thresholds:", t, (2*l*n_s), ((2*l)+1), ((2*l)+1)*n_s, 2*(l+1)*n_s)



def softmax(x):
    """ Standard definition of the softmax function """
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def LoadBatch(filename):
    # """ Copied from the dataset website """
    import pickle
    with open('Dataset/' + filename, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def ComputeGradsNum(X, Y, P, W1, W2, b1, b2, lam, h, classifier):
    print("compute grads num")
    grad_W1 = np.zeros(W1.shape)
    grad_b1 = np.zeros(b1.shape)
    grad_W2 = np.zeros(W2.shape)
    grad_b2 = np.zeros(b2.shape)

    c = classifier.ComputeCost(X, Y, W1, W2, b1, b2, lam)

    for i in range(len(b1)):
        b1_try = np.array(b1)
        b1_try[i] += h
        c2 = classifier.ComputeCost(X, Y, W1, W2, b1_try, b2, lam)
        grad_b1[i] = (c2 - c) / h

    print("done b1")

    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            W1_try = np.array(W1)
            W1_try[i, j] += h
            c2 = classifier.ComputeCost(X, Y, W1_try, W2, b1, b2, lam)
            grad_W1[i, j] = (c2 - c) / h

    print("done w1")

    for i in range(len(b2)):
        b2_try = np.array(b2)
        b2_try[i] += h
        c2 = classifier.ComputeCost(X, Y, W1, W2, b1, b2_try, lam)
        grad_b2[i] = (c2 - c) / h
    print("done b2")
    for i in range(W2.shape[0]):
        for j in range(W2.shape[1]):
            W2_try = np.array(W2)
            W2_try[i, j] += h
            c2 = classifier.ComputeCost(X, Y, W1, W2_try, b1, b2, lam)
            grad_W2[i, j] = (c2 - c) / h
    print("done w2")

    return [grad_W1, grad_W2, grad_b1, grad_b2]


def ComputeGradsNumSlow(X, Y, P, W1, W2, b1, b2, lam, h, classifier):
    print("start slow")
    grad_W1 = np.zeros(W1.shape)
    grad_b1 = np.zeros(b1.shape)
    grad_W2 = np.zeros(W2.shape)
    grad_b2 = np.zeros(b2.shape)

    for i in range(len(b1)):
        b1_try = np.array(b1)
        b1_try[i] -= h
        c1 = classifier.ComputeCost(X, Y, W1, W2, b1_try, b2, lam)

        b1_try = np.array(b1)
        b1_try[i] += h
        c2 = classifier.ComputeCost(X, Y, W1, W2, b1_try, b2, lam)

        grad_b1[i] = (c2 - c1) / (2 * h)

    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            W1_try = np.array(W1)
            W1_try[i, j] -= h
            c1 = classifier.ComputeCost(X, Y, W1_try, W2, b1, b2, lam)

            W1_try = np.array(W1)
            W1_try[i, j] += h
            c2 = classifier.ComputeCost(X, Y, W1_try, W2, b1, b2, lam)

            grad_W1[i, j] = (c2 - c1) / (2 * h)

    for i in range(len(b2)):
        b2_try = np.array(b2)
        b2_try[i] -= h
        c1 = classifier.ComputeCost(X, Y, W1, W2, b1, b2_try, lam)

        b2_try = np.array(b2)
        b2_try[i] += h
        c2 = classifier.ComputeCost(X, Y, W1, W2, b1, b2_try, lam)

        grad_b2[i] = (c2 - c1) / (2 * h)

    for i in range(W2.shape[0]):
        for j in range(W2.shape[1]):
            W2_try = np.array(W2)
            W2_try[i, j] -= h
            c1 = classifier.ComputeCost(X, Y, W1, W2_try, b1, b2, lam)

            W2_try = np.array(W2)
            W2_try[i, j] += h
            c2 = classifier.ComputeCost(X, Y, W1, W2_try, b1, b2, lam)

            grad_W2[i, j] = (c2 - c1) / (2 * h)

    return [grad_W1, grad_W2, grad_b1, grad_b2]

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
    #plt.savefig("plots/ "+fname +".png")
    plt.show()


def save_as_mat(data, name="model"):
    """ Used to transfer a python model to matlab """
    import scipy.io as sio

    # sio.savemat(name'.mat',{name:b})
    sio.savemat(name + ".mat", {name: data})


def get_mean_std(train_data):
    mean = np.mean(train_data, axis=1)
    mean = mean.reshape(-1,1)
    std = np.std(train_data, axis=1)
    std = std.reshape(-1,1)
    return mean, std


def PreProcessing(data, labels, mean, std):
    print("in preprocessing")
    scaled = data - mean
    normalized = scaled / std
    targets = np.transpose(np.eye(10)[labels])

    return normalized, targets

#def plot(train_cost, val_cost, W):


def check_gradients(data, targets, lamda, classifier):
    P = classifier.evaluateClassifier(data)
    grad_w1, grad_w2, grad_b1, grad_b2 = classifier.ComputeGradients(data, targets)
    gradient1_num, gradient2_num, gradientb1_num, gradientb2_num = ComputeGradsNum(data, targets, P, classifier.W1, classifier.W2, classifier.b1, classifier.b2, lamda, 1e-6, classifier)
    #gradient1_num_slow, gradient2_num_slow, gradientb1_num_slow, gradientb2_num_slow = ComputeGradsNumSlow(data, targets, P, classifier.W1, classifier.W2, classifier.b1, classifier.b2, lamda,
                                                                 #1e-6, classifier)
    print("compare backprop function to numerical calculation for parameters lambda, data size", lamda, data.shape[1])
    diff1 = grad_w1 - gradient1_num
    diff2 = grad_w2 - gradient2_num
    diff_b1 = grad_b1 - gradientb1_num
    diff_b2 = grad_b2 - gradientb2_num
    print("hello")
    np.testing.assert_almost_equal(grad_w1, gradient1_num, 7)
    np.testing.assert_almost_equal(grad_w2, gradient2_num, 7)
    np.testing.assert_almost_equal(grad_b1, gradientb1_num, 7)
    np.testing.assert_almost_equal(grad_b2, gradientb2_num, 7)
    #np.testing.assert_almost_equal(grad_w1, gradient1_num_slow, 7)
    #np.testing.assert_almost_equal(grad_w2, gradient2_num_slow, 7)

    if np.all(abs(diff1) <= 1e-6):
        print("grad equivalent")
    else:
        print("grad incorrect")
        print(np.max(abs(diff1)))
        print("epsilon comparison: ")
        #comp = abs(diff)/max(1e-6, (abs(gradient) + abs(gradient_num)))
        #print(max(comp))
    if np.all(abs(diff2) <= 1e-6):
        print("grad w2 good")
    else:
        print("grad w2 bad")
        print(np.max(abs(diff2)))
    print("compare backprop function to slow numerical calculation: ")
    #diff = gradient - gradient_num_slow
    if np.all(abs(diff_b1) <= 1e-6):
        print("b1 equivalent")
    else:
        print("incorrect")
        print(np.max(abs(diff_b1)))
    if np.all(abs(diff_b2) <= 1e-6):
        print("b2 good")
    else:
        print("b2 bad")
        print(np.max(diff_b2))




dict = LoadBatch("data_batch_1")
data = np.asarray(dict[b'data'].transpose(), dtype=float)/255.  # we need data in columns not rows
train_labels = dict[b'labels']
mean, std = get_mean_std(data)
train_data, train_targets = PreProcessing(data, train_labels, mean, std)
dict_val = LoadBatch("data_batch_2")
val_data = np.asarray(dict_val[b'data'].transpose(), dtype=float)/255
val_labels = dict_val[b'labels']
val_data, val_targets = PreProcessing(val_data, val_labels, mean, std)
dict_test = LoadBatch("test_batch")
test_data = np.asarray(dict_test[b'data'].transpose(), dtype=float)/255
test_labels = dict_test[b'labels']
test_data, test_targets = PreProcessing(test_data, test_labels, mean, std)
n_dims = data.shape[0]

two = TwoLayerClassifier(3072, hidden_dims=50, lr=0.001, lamda=0.01, num_labels=10)
#h, probs = two.evaluateClassifier(train_data)
#grad_w2, grad_b2, grad_w1, grad_b1 = two.ComputeGradients(train_data, train_targets)

#cost = two.ComputeCost(train_data, train_targets, two.W1, two.W2, two.b1, two.b2, 0)
#cost_train, cost_val = two.miniBatchGD(train_data[:,:100], train_targets[:,:100], val_data, val_targets, epochs=200, batch_size=100)
#plt.plot(cost_train)
#plt.plot(cost_val)
#plt.show()

#test learning rate:
'''etas = []
n_s = 800
loops = 3
t = 1
for loop in range(loops):
    for i in range(1,(2*n_s +1)):
        etas.append(two.computeLR(t, l=loop, n_s=n_s))
        t += 1
print(etas)
plt.plot(etas)
plt.show()'''

check_gradients(train_data[:, :50], train_targets[:, :50], 0, two)


cost_train, cost_val, acc_train, acc_val, etas = two.miniBatchGD(train_data, train_targets, val_data, val_targets, train_labels, val_labels, loops=3, epochs_per_loop=16, n_s=800)

plt.subplot(1,3,1)
plt.plot(cost_train, label="training cost")
plt.plot(cost_val, label='validation cost')
plt.legend()
plt.subplot(1,3,2)
plt.plot(acc_train, label='training accuracy')
plt.plot(acc_val, label='validation accuracy')
plt.legend()
plt.subplot(1,3,3)
plt.plot(etas)
plt.show()

# probs = one.evaluateClassifier(data[:, :100])
# cost = ComputeCost(train_data, train_targets, one.W, one.b, lamda)
# in_data = train_data[:, :100]
# train_targets =  train_targets[:,:100]'''



params = [[0, 0.1], [0, 0.001], [0.1, 0.001], [1, 0.001]]


#montage(dict[b'data'])

'''
for i in range(len(params)):
    classifier = OneLayerClassifier(n_dims, lr=params[i][1], lamda=params[i][0], num_labels=10)
    train_cost, val_cost = classifier.miniBatchGD(train_data, train_targets, val_data, val_targets, epochs=40, batch_size=100)
    #train_ = [i for i in train_cost if not np.isnan(i) and not i == float('inf')]
    #val_ = [i for i in train_cost if not np.isnan(i) and not i == float('inf')]
    #plt.subplot(1, 2, 1)
    #plt.ylim([0, ])
    plt.plot(train_cost, label="train cost")
    plt.title("parameters: lamda" + str(params[i][0]) + "eta" + str(params[i][1]))
    #plt.subplot(1,2,2)
    plt.plot(val_cost, label="validation cost")
    plt.legend()
    #plt.subplot(1, 2, 2)
    #weight = np.where(np.isnan(classifier.W) or classifier.W == float('inf'), 1500)
    #plt.imshow(classifier.W)
    #axes[i, 0].set_title(label=str("lamda" + str(params[i][0]) + "eta " + str(params[i][1])))
    accuracy, pred = classifier.computeAccuracy(test_data, test_labels)
    print("accuracy", accuracy)
    #print(classifier.W)
    filename = "noshuffle lr" + str(params[i][1]) + "lamda " + str(params[i][0]) + ".png"
    plt.savefig(fname="plots/" + filename)
    #plt.show()
    #weight2 = classifier.W
    montage(classifier.W)
print("out")'''

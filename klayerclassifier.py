import numpy as np
import matplotlib.pyplot as plt
import sys


class kLayerClassifier:

    def __init__(self, layers, lamda=0., batch_normalize=True):
        '''Args: layers: list of length number of layers + 1 with layer dimensions
                        contains: input dimensions, hidden layer(s) dimensions, output dimensions (num classes)
                lr: learning rate
                lamda: regularization parameter
                batch_normalize: boolean indicating whether or not batch normalization should be used
            Out: initialized k Layer Classifier'''
        np.random.seed(245)
        self.weights = []
        self.biases = []
        self.gammas = []
        self.betas = []
        self.avg_means = []
        self.avg_vars = []
        self.t = 1
        self.alpha = 0.9
        #initialize the hidden layers with He initialization:
        '''experiment with sensitivity to initialization'''
    sigmas = [1e-1, 1e-3, 1e-4]
        for l in range(1, len(layers)):
            self.weights.append(np.random.normal(0, sigmas[l-1], (layers[l], layers[l-1])))
            self.biases.append(np.zeros((layers[l], 1)))
            self.gammas.append(np.ones((layers[l], 1)))
            self.betas.append(np.zeros((layers[l], 1)))
        self.gammas.pop(-1)
        self.betas.pop(-1)
        #output layer
        self.k = len(layers)-2
        #outputs from each layer
        self.layer_activations = []
        #self.lr = lr
        self.lamda = lamda
        self.batch_normalize = batch_normalize
        self.layer_mean = []
        self.layer_var = []


    def evaluateClassifier(self, data, layer):
        '''Args: data: input to the layer in question
                layer: layer number
            Out: s: activations of that layer'''
        s = np.dot(self.weights[layer], data) + self.biases[layer]
        #probabilities = softmax(s)
        return s

    def computeClassifierOutput(self, x):
        return softmax(x)

    def ReLU(self, s):
        return np.where(s > 0, s, 0)

    def batchNormalize(self, s, mean, var):

        remove_mean = s-mean
        normalized = remove_mean*np.power(var + np.finfo(float).eps, -0.5)
        test = np.mean(normalized, axis=1)
        test2 = np.std(normalized, axis=1)
        return normalized

    def forward(self, data, means, vars_):
        self.layer_activations = []
        s = self.evaluateClassifier(data, 0)
        in_layer = {'s': s}
        #in_layer['input'] = data
        if self.batch_normalize:
            mean = np.mean(s, axis=1, keepdims=True)
            var = np.var(s, axis=1, keepdims=True)
            #print("mean layer 0", mean, self.t)
            #print('var layer 0', var, self.t)
            self.layer_mean.append(mean)
            self.layer_var.append(var)
            if self.t == 1:
                self.avg_means.append(mean)
                self.avg_vars.append(var)
            else:
                self.avg_means[0] = self.avg_means[0]* self.alpha + (1-self.alpha)*mean
                self.avg_vars[0] = self.avg_vars[0]*self.alpha + (1-self.alpha)*var
                #print('layer 0 t, mean', self.t, means[0])
                #print('layer 0, t, var', self.t, vars_[0])
            in_layer['norm'] = self.batchNormalize(in_layer['s'], means[0], vars_[0])
            s_tilde = self.gammas[0] * in_layer['norm'] + self.betas[0]
            in_layer['act'] = self.ReLU(s_tilde)
        else:
            in_layer['act'] = self.ReLU(in_layer['s'])
        self.layer_activations.append(in_layer)
        for l in range(1, self.k):
            s = self.evaluateClassifier(self.layer_activations[l - 1]['act'], l)
            #print('layer {}, s {}'.format(l, s))
            layer = {'s': s}
            #layer['input'] = self.layer_activations[l - 1]['act']
            if self.batch_normalize:
                mean = np.mean(s, axis=1, keepdims=True)
                var = np.var(s, axis=1, keepdims=True)
                #print("mean layer", l, mean, self.t)
                #print("var layer ", l, var, self.t)
                self.layer_mean.append(mean)
                self.layer_var.append(var)
                if self.t == 1:
                    self.avg_means.append(mean)
                    self.avg_vars.append(var)
                else:
                    self.avg_means[l] = self.avg_means[l] * self.alpha + (1 - self.alpha) * mean
                    self.avg_vars[l] = self.avg_vars[l] * self.alpha + (1 - self.alpha) * var
                    #print('layer t, mean', l, self.t, means[l])
                    #print('layer, t, var', l, self.t, vars_[l])
                layer['norm'] = self.batchNormalize(layer['s'], means[l], vars_[l])
                #print('normalized layer', layer['norm'])
                #print('gammas', self.gammas)
                #print('betas', self.betas)
                s_tilde = layer['norm'] * self.gammas[l] + self.betas[l]
                layer['act'] = self.ReLU(s_tilde)
            else:
                layer['act'] = self.ReLU(layer['s'])
            self.layer_activations.append(layer)
        #final layer:
        out_layer = {}
        last_input = self.layer_activations[-1]['act']
        out_layer['s'] = self.evaluateClassifier(self.layer_activations[-1]['act'], self.k)
        #self.layer_activations.append(out_layer)
        p = softmax(out_layer['s'])
        return p

    def batchNormBackPass(self, G, S, mean, var):
        #print(self.t)

        batch_size = G.shape[1]
        ones = np.ones((batch_size, 1))
        sigma_1 = np.power(var + np.finfo(float).eps, -0.5)
        #print('sigma1', sigma_1)
        sigma_2 = np.power(var + np.finfo(float).eps, -1.5)
        #print("sigma2", sigma_2)
        sigma_1_ones = np.dot(sigma_1, ones.T)
        G1_1 = G*sigma_1_ones
        G1 = G * sigma_1
        G2 = G * sigma_2
        D = S - mean
        c = np.sum(G2*D, axis=1, keepdims=True)

        G = G1 - np.sum(G1, axis=1, keepdims=True)/batch_size - (D * c)/batch_size
        return G

    def ComputeGradients(self, X, P, Y):
        self.W_gradients = []
        self.b_gradients = []
        self.gamma_gradients = []
        self.beta_gradients = []

        batch_size = X.shape[1]
        G = -(Y - P)

        gradW_k = np.dot(G, self.layer_activations[self.k-1]['act'].T)/batch_size + 2*self.lamda*self.weights[self.k]
        grad_b_k = np.sum(G, axis=1)/batch_size
        grad_b_k = grad_b_k.reshape(-1,1)
        self.W_gradients.insert(0, gradW_k)
        self.b_gradients.insert(0, grad_b_k)
        G = np.dot(self.weights[self.k].T, G)
        ind = np.where(self.layer_activations[self.k-1]['act'] > 0, 1, 0)
        G = G * ind
        for l in range(self.k-1, -1, -1):
            ones = np.ones((batch_size, 1))
            if self.batch_normalize:
                grad_gamma1 = np.dot(G*self.layer_activations[l]['norm'], ones)/batch_size
                grad_gamma = np.sum(G*self.layer_activations[l]['norm'], axis=1, keepdims=True)/batch_size
                grad_beta = np.sum(G, axis=1, keepdims=True)/batch_size
                grad_beta1 = np.dot(G, ones)/batch_size
                self.gamma_gradients.insert(0, grad_gamma)
                self.beta_gradients.insert(0, grad_beta)
                G = G * self.gammas[l]
                G = self.batchNormBackPass(G, self.layer_activations[l]['s'], self.layer_mean[l], self.layer_var[l])
                #print('G from batchNormBackPass, t=', self.t, G)
            if l == 0:
                grad_W = np.dot(G, X.T)/batch_size + 2*self.lamda*self.weights[0]
                grad_b = np.sum(G, axis=1)/batch_size
                grad_b = grad_b.reshape(-1,1)
            else:
                grad_W = np.dot(G, self.layer_activations[l-1]['act'].T)/batch_size + 2*self.lamda*self.weights[l]
                grad_b = np.sum(G, axis=1)/batch_size
                grad_b = grad_b.reshape(-1,1)
            self.W_gradients.insert(0, grad_W)
            self.b_gradients.insert(0, grad_b)
            if l > 0:
                G = np.dot(self.weights[l].T, G)
                ind = np.where(self.layer_activations[l-1]['act'] > 0, 1, 0)
                G = G * ind
        self.layer_mean = []
        self.layer_var = []
        return self.W_gradients, self.b_gradients


    def update_weights(self, eta):
        for l in range(len(self.weights)):
            self.weights[l] -= eta*self.W_gradients[l]
            self.biases[l] -= eta*self.b_gradients[l]
        if self.batch_normalize:
            for l in range(len(self.gamma_gradients)):
                self.gammas[l] -= eta*self.gamma_gradients[l]
                self.betas[l] -= eta*self.beta_gradients[l]
        #self.weights[0] -= eta*self.W_gradients[0]
        #self.betas[0] -= eta*self.b_gradients[0]

    def computeAccuracy(self, data, labels):
        probabilities = self.forward(data, self.avg_means, self.avg_vars)
        pred_labels = [np.argmax(probabilities[:, i]) for i in range(probabilities.shape[1])]
        # pred_labels = np.array([np.argmax(probabilities[:,i]) for i in range(probabilities.shape[1])])
        labels = np.asarray(labels)
        pred_labels = np.asarray(pred_labels)
        correct = np.where(pred_labels == labels, 1, 0)
        # sum = np.sum(correct)
        accuracy = np.sum(correct) / data.shape[1]
        return accuracy * 100

    def miniBatchGD(self, data, targets, val_data, val_targets, train_labels, val_labels, loops, epochs_per_loop,
                    batch_size=100, n_s=500):
        target_batches = []
        accuracy_train = []
        accuracy_val = []
        train_cost = []
        val_cost = []
        self.t = 1
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
                    p = self.forward(mini_batches[i], self.layer_mean, self.layer_var)

                    W_gradients, b_gradients = self.ComputeGradients(mini_batches[i], p, target_batches[i])
                    lr = self.computeLR(self.t, l=l, n_s=n_s)
                    etas.append(lr)
                    self.update_weights(lr)


                    self.t += 1


                train_accuracy = self.computeAccuracy(data, train_labels)
                val_accuracy = self.computeAccuracy(val_data, val_labels)
                accuracy_train.append(train_accuracy)
                accuracy_val.append(val_accuracy)
                train_cost.append(self.ComputeCost(data, targets, self.avg_means, self.avg_vars))
                val_cost.append(self.ComputeCost(val_data, val_targets, self.avg_means, self.avg_vars))
                print("train accuracy ", train_accuracy)
                print(" validation accuracy ", val_accuracy)

        return accuracy_train, accuracy_val, train_cost, val_cost, etas


    def ComputeCost(self, X, Y, means, vars_):
        batch_size = X.shape[1]
        p = self.forward(X, means, vars_)
        sum = 0
        y = Y.T
        for i in range(Y.shape[1]):
            sum += -np.log(np.dot(y[i, :], p[:, i]))
        cost = sum / batch_size
        # loss_part1 = (1/batch_size) * log_sum
        for l in range(self.k):
            cost += self.lamda * np.sum(self.weights[l]**2)
        return cost

    def computeLR(self, t, l, eta_min=1e-5, eta_max=1e-1, n_s=500):
        # l = 1 #number of cycles

        if (2 * l * n_s) <= t <= ((2 * l) + 1) * n_s:

            num = t - 2 * l * n_s
            eta_t = eta_min + num / n_s * (eta_max - eta_min)
            return eta_t
        elif ((2 * l) + 1) * n_s <= t <= 2 * (l + 1) * n_s:
            num = t - (2 * l + 1) * n_s
            eta_t = eta_max - num / n_s * (eta_max - eta_min)
            return eta_t
        else:
            raise ValueError("bad eta, t, thresholds:", t, (2 * l * n_s), ((2 * l) + 1), ((2 * l) + 1) * n_s,
                             2 * (l + 1) * n_s)

    def ComputeGradsNum(self, X, Y, lam, h):
        print("compute grads num")
        grad_W1 = np.zeros(W1.shape)
        grad_b1 = np.zeros(b1.shape)
        grad_W2 = np.zeros(W2.shape)
        grad_b2 = np.zeros(b2.shape)
        W_gradients = []
        b_gradients = []
        for l in range(len(self.weights)):
            W_gradients[l] = np.zeros(self.weights[l].shape)
            b_gradients[l] = np.zeros(self.biases.shape)

        c = self.ComputeCost(X, Y)
        for l in range(len(b_gradients)):
            for i in range(len(b_gradients[l])):
                b_try = np.array(b_gradients[l])
                b_try[i] += h
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


def softmax(x):
    """ Standard definition of the softmax function """
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def LoadBatch(filename):
    # """ Copied from the dataset website """
    import pickle
    with open('Dataset/' + filename, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

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
    # plt.savefig("plots/ "+fname +".png")
    plt.show()


def save_as_mat(data, name="model"):
    """ Used to transfer a python model to matlab """
    import scipy.io as sio

    # sio.savemat(name'.mat',{name:b})
    sio.savemat(name + ".mat", {name: data})


def get_mean_std(train_data):
    mean = np.mean(train_data, axis=1)
    mean = mean.reshape(-1, 1)
    std = np.std(train_data, axis=1)
    std = std.reshape(-1, 1)
    return mean, std


def PreProcessing(data, labels, mean, std):
    print("in preprocessing")
    scaled = data - mean
    normalized = scaled / std
    targets = np.transpose(np.eye(10)[labels])

    return normalized, targets


# def plot(train_cost, val_cost, W):


def check_gradients(data, targets, lamda, classifier):
    P = classifier.evaluateClassifier(data)
    grad_w1, grad_w2, grad_b1, grad_b2 = classifier.ComputeGradients(data, targets)
    gradient1_num, gradient2_num, gradientb1_num, gradientb2_num = ComputeGradsNum(data, targets, P, classifier.W1,
                                                                                   classifier.W2, classifier.b1,
                                                                                   classifier.b2, lamda, 1e-6,
                                                                                   classifier)
    # gradient1_num_slow, gradient2_num_slow, gradientb1_num_slow, gradientb2_num_slow = ComputeGradsNumSlow(data, targets, P, classifier.W1, classifier.W2, classifier.b1, classifier.b2, lamda,
    # 1e-6, classifier)
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
    # np.testing.assert_almost_equal(grad_w1, gradient1_num_slow, 7)
    # np.testing.assert_almost_equal(grad_w2, gradient2_num_slow, 7)

    if np.all(abs(diff1) <= 1e-6):
        print("grad equivalent")
    else:
        print("grad incorrect")
        print(np.max(abs(diff1)))
        print("epsilon comparison: ")
        # comp = abs(diff)/max(1e-6, (abs(gradient) + abs(gradient_num)))
        # print(max(comp))
    if np.all(abs(diff2) <= 1e-6):
        print("grad w2 good")
    else:
        print("grad w2 bad")
        print(np.max(abs(diff2)))
    print("compare backprop function to slow numerical calculation: ")
    # diff = gradient - gradient_num_slow
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

def coarse_random_search(l_min, l_max, train_data, train_targets, train_labels, val_data,
                         val_targets, val_labels, test_data, test_labels):
    print('start random search')
    batch_size = 100
    num_batches = train_data.shape[1] / batch_size
    n_s = int(2 * num_batches)
    epochs = int((2 * n_s) / num_batches)
    loops = 2
    rand_ = np.random.uniform(0, 1, (8, 1))
    lamdas = l_min + (l_max - l_min) * rand_
    lamdas = 10 ** lamdas
    accuracy = 0
    best_performance = {'lambda': 0, 'accuracy': 0}
    for idx, lamda in enumerate(lamdas):
        classifier = kLayerClassifier(layers=[3072, 50, 50, 10], lamda=lamda, batch_normalize=True)
        acc_train, acc_val, train_cost, val_cost, etas = classifier.miniBatchGD(train_data, train_targets, val_data,
                                                                                val_targets, train_labels, val_labels,
                                                                                loops=loops, epochs_per_loop=epochs,
                                                                                batch_size=100, n_s=n_s)
        test_accuracy = classifier.computeAccuracy(test_data, test_labels)
        with open('performance_finesearch.txt', 'a') as performance:
            performance.write("lambda: {}, validation accuracy: {}".format(lamda, acc_val[-1]))
            performance.write('\n')
            print("lambda: ", lamda, "end accuracy: ", acc_val[-1])
            if test_accuracy > accuracy:
                best_performance['accuracy'], best_performance['lambda'] = test_accuracy, lamda
            performance.write("\n best performance: lambda {}, accuracy {}".format(best_performance['lambda'],
                                                                                   best_performance['accuracy']))
        plt.subplot(1, 2, 1)
        plt.plot(acc_train, label="train accuracy")
        plt.plot(acc_val, label="validation accuracy")
        plt.legend()
        plt.title("lambda: {} end val accuracy {}".format(lamda, acc_val[-1]))
        plt.subplot(1, 2, 2)
        plt.plot(etas)
        plt.savefig("assgn 2 plots/assgn3coarse_search{}".format(idx))
        plt.clf()

    print('end random search')

data_dicts = []
data_dicts.append(LoadBatch("data_batch_1"))
data_dicts.append(LoadBatch("data_batch_2"))
data_dicts.append(LoadBatch("data_batch_3"))
data_dicts.append(LoadBatch("data_batch_4"))
data_dicts.append(LoadBatch("data_batch_5"))
train_data = np.asarray(data_dicts[0][b'data'].transpose(), dtype=float)
train_labels = data_dicts[0][b'labels']
for d in data_dicts[1:]:
    train_data = np.concatenate((train_data, d[b'data'].transpose()), axis=1)
    train_labels = np.concatenate((train_labels, d[b'labels']))
val_data = train_data[:, :5000]
val_labels = train_labels[:5000]
train_data = train_data[:, 5000:]
train_labels = train_labels[5000:]

mean, std = get_mean_std(train_data)
train_data, train_targets = PreProcessing(train_data, train_labels, mean, std)
val_data, val_targets = PreProcessing(val_data, val_labels, mean, std)

dict_test = LoadBatch("test_batch")
test_data = np.asarray(dict_test[b'data'].transpose(), dtype=float)
test_labels = dict_test[b'labels']
test_data, test_targets = PreProcessing(test_data, test_labels, mean, std)
n_dims = train_data.shape[0]

#coarse_random_search(-3, -2, train_data, train_targets, train_labels, val_data, val_targets, val_labels,
                    #  test_data, test_labels)

#layers: in, hidden layer, out eg two layers
lamda = 0.00744241
classifier = kLayerClassifier(layers=[3072, 50, 50, 10], lamda=lamda, batch_normalize=True)
#p = classifier.forward(train_data)
#w_grads, b_grads = classifier.ComputeGradients(train_data, p, train_targets)
#classifier.update_weights(eta=0.02)


loops = 3
batch_size = 100
num_batches = train_data.shape[1] / batch_size
print("num batches", num_batches)
n_s = int(5 * num_batches)
print("n_s", n_s)
epochs_per_loop = int(2 * n_s / num_batches)
print("epochs per loop", epochs_per_loop)
accuracy_train, accuracy_val, train_cost, val_cost, etas = classifier.miniBatchGD(train_data,
                                                            train_targets,val_data, val_targets, train_labels, val_labels,
                                                            loops=3, epochs_per_loop=epochs_per_loop, n_s=n_s, batch_size=batch_size)


plt.subplot(1, 3, 1)
plt.plot(accuracy_train, label='training accuracy')
plt.plot(accuracy_val, label='validation accuracy')
plt.legend()
plt.subplot(1, 3, 2)
plt.plot(train_cost, label="training cost")
plt.plot(val_cost, label="validation cost")
plt.legend()
plt.subplot(1,3,3)
plt.plot(etas)
plt.suptitle("test training Batch Normalization k={} layers with lambda = {}, final validation accuracy: {}".format(classifier.k+2, lamda, accuracy_val[-1]))
plt.show()

test_accuracy = classifier.computeAccuracy(test_data, test_labels)
print(test_accuracy)
# check_gradients(train_data[:, :50], train_targets[:, :50], 0, two)'''





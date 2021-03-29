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


def softmax(x):
    """ Standard definition of the softmax function """
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def LoadBatch(filename):
	#""" Copied from the dataset website """
    import pickle
    with open('Dataset/'+ filename, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def ComputeCost(X, Y, W, b, lamda):
	batch_size = X.shape[1]
	#print("cost, batch size ", batch_size)
	p = softmax(np.dot(W, X) + b)
	#y = np.todense(Y)
	log_in = np.dot(np.transpose(Y), p)
	log_sum = np.sum(-np.log(log_in))
	loss_part1 = (1/batch_size) * log_sum
	loss_part2 = lamda * np.sum(W**2)
	cost = loss_part1 + loss_part2
	#print("cost ", cost)


	return cost

def ComputeGradients(X, Y, P, W, lamda):
	batch_size = X.shape[1]
	print("batch size ", batch_size)
	#p = one.evaluateClassifier(X)
	G = -(Y - P)
	dot = np.dot(G, np.transpose(X))
	d_w1 = 1/batch_size * dot
	grad_w = d_w1 + 2 * lamda * W
	grad_b = 1/batch_size * np.sum(G, axis=0)
	return grad_w, grad_b


def ComputeGradsNum(X, Y, P, W, b, lamda, h):
	#""" Converted from matlab code """
	no 	= 	W.shape[0]
	d 	= 	X.shape[0]

	grad_W = np.zeros(W.shape);
	grad_b = np.zeros((no, 1));

	c = ComputeCost(X, Y, W, b, lamda);
	
	for i in range(len(b)):
		b_try = np.array(b)
		b_try[i] += h
		c2 = ComputeCost(X, Y, W, b_try, lamda)
		grad_b[i] = (c2-c) / h

	for i in range(W.shape[0]):
		for j in range(W.shape[1]):
			W_try = np.array(W)
			W_try[i,j] += h
			c2 = ComputeCost(X, Y, W_try, b, lamda)
			grad_W[i,j] = (c2-c) / h

	return [grad_W, grad_b]

def ComputeGradsNumSlow(X, Y, P, W, b, lamda, h):
	""" Converted from matlab code """
	no 	= 	W.shape[0]
	d 	= 	X.shape[0]

	grad_W = np.zeros(W.shape);
	grad_b = np.zeros((no, 1));
	
	for i in range(len(b)):
		b_try = np.array(b)
		b_try[i] -= h
		c1 = ComputeCost(X, Y, W, b_try, lamda)

		b_try = np.array(b)
		b_try[i] += h
		c2 = ComputeCost(X, Y, W, b_try, lamda)

		grad_b[i] = (c2-c1) / (2*h)

	for i in range(W.shape[0]):
		for j in range(W.shape[1]):
			W_try = np.array(W)
			W_try[i,j] -= h
			c1 = ComputeCost(X, Y, W_try, b, lamda)

			W_try = np.array(W)
			W_try[i,j] += h
			c2 = ComputeCost(X, Y, W_try, b, lamda)

			grad_W[i,j] = (c2-c1) / (2*h)

	return [grad_W, grad_b]

def montage(W):
	""" Display the image for each label in W """
	import matplotlib.pyplot as plt
	fig, ax = plt.subplots(2,5)
	for i in range(2):
		for j in range(5):
			im  = W[i*5+j,:].reshape(32,32,3, order='F')
			sim = (im-np.min(im[:]))/(np.max(im[:])-np.min(im[:]))
			sim = sim.transpose(1,0,2)
			ax[i][j].imshow(sim, interpolation='nearest')
			ax[i][j].set_title("y="+str(5*i+j))
			ax[i][j].axis('off')
	plt.show()

def save_as_mat(data, name="model"):
	""" Used to transfer a python model to matlab """
	import scipy.io as sio

	#sio.savemat(name'.mat',{name:b})
	sio.savemat(name+".mat", {name:data})


def get_mean_std(train_data):
	mean = np.mean(train_data, axis=0)
	std = np.std(train_data, axis=0)
	return mean, std


def PreProcessing(data, labels, mean, std):
	print("in preprocessing")
	scaled = data - mean
	normalized = scaled/std
	targets = np.transpose(np.eye(10)[labels])

	return normalized, targets

'''def check_gradients(data, targets, lamda, classifier, batch_size=1):
	P = classifier.evaluateClassifier(data)
	gradients = []
	gradients_num = []
	b_gradients = []
	b_gradients_num = []
	for i in range(0, batch_size, data.shape[1]):
		x = data[:, i:batch_size]
		y = targets[i:batch_size]
		gradient, b_gradient = ComputeGradients(x, y, P, classifier.W, lamda)
		gradients.append(gradient)
		b_gradients.append(gradient)
		gradient_num, gradient_b_num = ComputeGradsNum(x, y, P, classifier.W, classifier.b, lamda, h=1e-6)
		gradients_num.append(gradient_num)
		b_gradients_num.append(gradient_b_num)'''

def check_gradients(data, targets, lamda, classifier):
	P = classifier.evaluateClassifier(data)
	gradient, b_gradient = ComputeGradients(data, targets, P, classifier.W, lamda)
	gradient_num, gradient_b_num = ComputeGradsNum(data, targets, P, classifier.W, classifier.b, lamda, h=1e-6)
	gradient_num_slow, gradient_b_num_slow = ComputeGradsNumSlow(data, targets, P, classifier.W, classifier.b, lamda, h=1e-6)
	print("compare backprop function to numerical calculation: ")
	diff = gradient - gradient_num
	if np.all(abs(diff)) <= 1e-3:
		print("equivalent")
	else:
		print("incorrect")
	print("compare backprop function to slow numerical calculation: ")
	diff = gradient - gradient_num_slow
	if np.all(abs(diff)) <= 1e-3:
		print("equivalent")
	else:
		print("incorrect")



dict = LoadBatch("data_batch_1")
data = dict[b'data'].transpose() #we need data in columns not rows
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

one = OneLayerClassifier(data.shape[0], lr=0.005, num_labels=10)

#probs = one.evaluateClassifier(data[:, :100])
lamda = 0
#cost = ComputeCost(train_data, train_targets, one.W, one.b, lamda)
#in_data = train_data[:, :100]
#train_targets =  train_targets[:,:100]
check_gradients(train_data[:, :100], train_targets[:,:100], 0, one)

print("out")

import numpy as np
import h5py

# Training rate
alpha = 0.002

def sigmoid(z):
    """Return the logistic function sigma(z) = 1/(1+exp(-z))."""
    return 1 / (1+np.exp(-z))

def cost(Y, Yhat):
    """Return the cost function for predictions Yhat of classifications Y."""
    return (- Y @ np.log(Yhat.T) - (1 - Y) @ np.log(1 - Yhat.T)) / m

def accuracy(Y, Yhat):
    """Return measure of the accuracy with which Yhat predicts Y."""
    return 1 - np.mean(np.abs(Y - Yhat.round()))

def model(X, w, b):
    """Apply the logistic model parameterized by w, b to features X."""
    z = w.T @ X + b
    Yhat = sigmoid(z)
    return z, Yhat


def train(X, Y, max_it=1000):
    """Train the logistic regression algorithm on the data X classified as Y."""

    # Parameter vector, w, and constant term (bias), b.
    # For random initialization, use the following:
    #w, b = np.random.random((nx,1)) * 0.01, 0.01
    # To initialize with zeros, use this line instead:
    w, b = np.zeros((nx,1)), 0

    def propagate(w, b):
        """Propagate the training by advancing w, b to reduce the cost, J."""
        z, Yhat = model(X, w, b)
        w -= alpha / m * (X @ (Yhat - Y).T)
        b -= alpha / m * np.sum(Yhat - Y)
        J = np.squeeze(cost(Y, Yhat))
        if not it % 100:
            # Provide an update on the progress we have made so far.
            print('{}: J = {}'.format(it, J))
            print('train accuracy = {:g}%'.format(accuracy(Y, Yhat) * 100))
        return w, b

    # Train the model by iteratively improving w, b.
    for it in range(max_it):
        w, b = propagate(w, b)
    return w, b

import random

train_indices = random.sample([i for i in range(320)], 250)
test_indices = [i for i in range(320) if not i in train_indices]

ds = h5py.File('./happyds.h5', 'r')

ds_x = np.array([ds["x_train"][i] for i in train_indices]).T / 100
ds_y = np.array([ds["y_train"][i] for i in train_indices])

m = len(ds_y)

# Dimension of the feature vector for each example.
nx = ds_x.size // m
# Packed feature vector and associated classification.
X, Y = ds_x.reshape((nx, m)), ds_y.reshape((1, m))

# Train the model
w, b = train(X, Y)

i = 0
features = np.asarray(ds["x_train"][test_indices[i]], dtype='uint8')[:, :].T / 100
acc_y = ds["y_train"][test_indices[i]]

print(model(features, w, b))
# from PIL import Image

# def categorize_image(filename):
#     """Categorize the image provided in filename.

#     Return 1 if the image is categorized in the y=1 class and otherwise 0.

#     """

#     im = Image.open(filename)
#     ds_x = np.asarray(im, dtype='uint8')[:, :, :3].T / 100
#     ds_y = np.array([1])
#     # Number of test examples.
#     m = len(ds_y)
#     # Dimension of the feature vector for each example.
#     nx = ds_x.size // m
#     # Packed feature vector and associated classification.
#     X, Y = ds_x.reshape((nx, m)), ds_y.reshape((1,m))
#     z, Yhat = model(X, w, b)
#     return np.squeeze(Yhat) > 0.5

# categorize_image("test.png")


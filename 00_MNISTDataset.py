# Besmei Taala
# Amir Hossein Karami 
# MainLink: https://github.com/yunjey/davian-tensorflow/blob/master/notebooks/week1/1.%20mnist_data_introduction.ipynb


from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np


# MNIST Data Set:
# It has 55,000 examples for training and 10,000 examples for testing.
# each sample is represented as a 2D matrix of size 28x28 with values from 0 to 1.


# ***** Section1 *****

print('\n ***** Section1 ***** ')

# 1- Downloading the data set
# 2- Loading the entire data set into numpy array

mnist = input_data.read_data_sets("./mnist/", one_hot=True)

# Load data
x_train = mnist.train.images
y_train = mnist.train.labels
x_test = mnist.test.images
y_test = mnist.test.labels


# ***** Section2 *****

print('\n ***** Section2 ***** ')

# Data shape:
# MNIST data set has 55,000 examples for training and 10,000 examples for testing. Each image has a size of 784 (28x28).

print("x_train: ", x_train.shape)
print("y_train: ", y_train.shape)
print("x_test: ", x_test.shape)
print("y_test: ", y_test.shape)


# ***** Section3 *****

print('\n ***** Section3 ***** ')

# Data visualization:


def plot_mnist(data, classes):
    for i in range(10):
        idxs = (classes == i)

        # get 10 images for class i
        images = data[idxs][0:10]

        for j in range(5):
            plt.subplot(5, 10, i + j * 10 + 1)
            plt.imshow(images[j].reshape(28, 28), cmap='gray')
            # print a title only once for each class
            if j == 0:
                plt.title(i)
            plt.axis('off')
    plt.show()


classes = np.argmax(y_train, 1)
plot_mnist(x_train, classes)


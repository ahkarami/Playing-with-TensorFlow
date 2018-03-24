# Besmei Taala
# Amir Hossein Karami 
# MainLink: https://github.com/yunjey/davian-tensorflow/blob/master/notebooks/week2/convolutional_neural_network.ipynb


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# ***** Section1 *****

print('\n ***** Section1 ***** ')

# load MNIST data set:
mnist = input_data.read_data_sets("./mnist", one_hot=True)
x_train = mnist.train.images
y_train = mnist.train.labels
x_test = mnist.test.images
y_test = mnist.test.labels

print("x_train: ", x_train.shape)
print("y_train: ", y_train.shape)
print("x_test: ", x_test.shape)
print("y_test: ", y_test.shape)


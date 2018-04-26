# Besmei Taala
# Amir Hossein Karami 
# MainLink: https://github.com/yunjey/davian-tensorflow/blob/master/notebooks/week2/save_and_restore_model.ipynb


import os
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


# ***** Section2 *****

print('\n ***** Section2 ***** ')


def fully_connected(x, dim_in, dim_out, name):
    with tf.variable_scope(name) as scope:
        # create variables
        w = tf.get_variable('w', shape=[dim_in, dim_out],
                            initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
        b = tf.get_variable('b', shape=[dim_out])

        # create operations
        out = tf.matmul(x, w) + b

        return out


# Create model
def neural_network(x, dim_in=784, dim_h=500, dim_out=10):
    # 1st hidden layer with ReLU
    h1 = fully_connected(x, dim_in, dim_h, name='h1')
    h1 = tf.nn.relu(h1)

    # 2nd hidden layer with ReLU
    h2 = fully_connected(h1, dim_h, dim_h, name='h2')
    h2 = tf.nn.relu(h2)

    # output layer with linear
    out = fully_connected(h2, dim_h, dim_out, name='out')

    return out


x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])


# Construct model with default value
out = neural_network(x)

# loss and optimizer
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=out))
optimizer = tf.train.RMSPropOptimizer(learning_rate=0.0001).minimize(loss)

# Test model
pred = tf.argmax(out, 1)
target = tf.argmax(y, 1)
correct_pred = tf.equal(pred, target)

# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


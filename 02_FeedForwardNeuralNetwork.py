# Besmei Taala
# Amir Hossein Karami 
# MainLink: https://github.com/yunjey/davian-tensorflow/blob/master/notebooks/week1/3.%20feed_forward_neural_network.ipynb


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
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

# Data sanity check:


def plot_mnist(data, classes, incorrect=None):
    for i in range(10):
        idxs = (classes == i)

        # for test: see last part of this ipython notbook file
        if incorrect is not None:
            idxs *= incorrect
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


# ***** Section3 *****

print('\n ***** Section3 ***** ')

# Fully-connected layer:
# Implement a fully-connected layer, by using `tf.matmul` function for 2D matrix multiplication.


def fully_connected(x, dim_in, dim_out, name):
    with tf.variable_scope(name) as scope:
        # create variables
        w = tf.get_variable('w', shape=[dim_in, dim_out],
                            initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
        b = tf.get_variable('b', shape=[dim_out])

        # create operations
        out = tf.matmul(x, w) + b

        return out


# ***** Section4 *****

print('\n ***** Section4 ***** ')

# Neural network:
# we will develop a neural network with 2 hidden layers using a fully_connected function.


# Create model:
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


# ***** Section5 *****

print('\n ***** Section5 ***** ')

# Place holder:
# To train the neural network with mini-batch gradient descent,
# placeholders should be defined for mini-batch input data and target data.
# Note: `None` type is used so that any batch size of data can be fed into the placeholder.
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])


# ***** Section6 *****

print('\n ***** Section6 ***** ')

# Construct graph:
# Construct model with default value
out = neural_network(x)

# Note: If one want to create 2 or more neural network must act as below:
# tf.get_variable_scope().reuse_variables()
# out2 = neural_network(x)


# ***** Section7 *****

print('\n ***** Section7 ***** ')

# Loss and optimizer:
# Acts loss function as a tensor and optimizer as an operator.

# loss and optimizer
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=out))
optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(loss)
#
# Test model
pred = tf.argmax(out, 1)
target = tf.argmax(y, 1)

correct_pred = tf.equal(pred, target)
incorrect_pred = tf.not_equal(pred, target)
# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# ***** Section8 *****

print('\n ***** Section8 ***** ')

# Session: train and test model:
# To launch our model, we will implement a session where the model is actually trained and tested on the MNIST data set.

# Train Phase:
# 1- Initialize all variables we created above.
# 2- The most important part of code in the training phase is:
# `sess.run([optimizer, loss], feed_dict={x: x_batch, y:y_batch})`.

# Test Phase:
# We use sess.run(accuracy, {x: mnist.test.images, y: mnist.test.labels}) to print out the test accuracy.
# Also, sess.run([pred, incorrect_pred], {x: mnist.test.images, y: mnist.test.labels}) is used to
# get predictions and incorrectly predicted indices of our model.

batch_size = 100
numberOfEpochs = 10

# launch the graph
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    # initialize tensor variables
    tf.global_variables_initializer().run()
    # training cycle
    for epoch in range(numberOfEpochs):
        avg_loss = 0.
        n_iters_per_epoch = int(mnist.train.num_examples / batch_size)
        # loop over all batches
        for i in range(n_iters_per_epoch):
            x_batch, y_batch = mnist.train.next_batch(batch_size)
            # run optimization op (backprop) and loss op (to get loss value)
            _, c = sess.run([optimizer, loss], feed_dict={x: x_batch, y: y_batch})
            # compute average loss
            avg_loss += c / n_iters_per_epoch
        print("Epoch %d, Loss: %.3f" % (epoch + 1, avg_loss))
    print("Finished training!")

    print("")

    print("Test accuracy:", sess.run(accuracy, {x: mnist.test.images, y: mnist.test.labels}))
    p, incorrect = sess.run([pred, incorrect_pred], {x: mnist.test.images, y: mnist.test.labels})

    print("Incorrect Case")
    plot_mnist(x_test, classes=p, incorrect=incorrect)  # I think this method has some bugs in some cases


# ***** Some Other Appropriate References about Using GPU with TensorFlow *****
# 1- https://medium.com/@lisulimowicz/tensorflow-cpus-and-gpus-configuration-9c223436d4ef
# 2- https://learningtensorflow.com/lesson10/



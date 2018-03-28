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


# ***** Section2 *****

print('\n ***** Section2 ***** ')

# Parameter setting:

# Parameters
learning_rate = 0.001
training_iters = 200000
batch_size = 100
display_step = 200

# Network parameters
n_input = 784
n_classes = 10
dropout = 0.5

# Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)  # to keep dropout probability

weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


# ***** Section3 *****

print('\n ***** Section3 ***** ')

# Network Implementation:


def CNN(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 28, 28, 1])
    print("Input x:               ", x.get_shape().as_list())

    # 1st Convolution Layer
    conv1 = tf.nn.conv2d(x, weights['wc1'], strides=[1, 1, 1, 1], padding='SAME')
    print("After 1st conv:        ", conv1.get_shape().as_list())
    conv1 = tf.nn.bias_add(conv1, biases['bc1'])
    print("After adding bias:     ", conv1.get_shape().as_list())
    conv1 = tf.nn.relu(conv1)
    print("After ReLU:            ", conv1.get_shape().as_list())
    # Max Pooling (down-sampling)
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    print("After 1st max_pooling: ", pool1.get_shape().as_list())

    # 2nd Convolution Layer
    conv2 = tf.nn.conv2d(pool1, weights['wc2'], strides=[1, 1, 1, 1], padding='SAME')
    print("After 2nd conv:        ", conv2.get_shape().as_list())
    conv2 = tf.nn.bias_add(conv2, biases['bc2'])
    print("After adding bias:     ", conv2.get_shape().as_list())
    conv2 = tf.nn.relu(conv2)
    print("After ReLU:            ", conv2.get_shape().as_list())
    # Max Pooling (down-sampling)
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    print("After 2nd max_pooling: ", pool2.get_shape().as_list())

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(pool2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    print("Fully connected:       ", fc1.get_shape().as_list())
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)
    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    print("Output:                ", out.get_shape().as_list())
    return out


# ***** Section4 *****

print('\n ***** Section4 ***** ')

# Optimization:

# Construct model:
pred = CNN(x, weights, biases, keep_prob)

# Define loss:
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))

# Define optimizer:
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
# optimizer = tf.train.AdadeltaOptimizer().minimize(cost)
# optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(cost)
# optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.5).minimize(cost)
# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)


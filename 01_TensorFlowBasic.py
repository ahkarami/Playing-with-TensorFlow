# Besmei Taala
# Amir Hossein Karami 
# MainLink: https://github.com/yunjey/davian-tensorflow/blob/master/notebooks/week1/2.%20tensorflow_basic.ipynb


import tensorflow as tf

# Version of TensorFlow:
print(tf.__version__)  # 1.4.0

# ***** Section1 *****

# Session:
# Session is a class for running TensorFlow operations.

print(' ***** Section1 *****')

a = tf.constant(100)

with tf.Session() as sess:
    print(sess.run(a))
    # syntactic sugar
    print(a.eval())

# or

sess = tf.Session()
print(sess.run(a))

# Interactive session:
sess = tf.InteractiveSession()
print(a.eval())     # simple usage


# ***** Section2 *****

# Constants:
# Note: The methods Tensor.eval() and Operation.run() will use that session to run ops.
# Remember:
# 1- Tensor.eval()
# 2- Operation.run()

# tensor constant example:

print(' ***** Section2 *****')
a = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float32, name='a')
print(a.eval())
print(type(a))
print("shape: ", a.get_shape(), ",type: ", type(a.get_shape()))
print("shape: ", a.get_shape().as_list(), ",type: ", type(a.get_shape().as_list()))    # this is more useful
print("number of rows: ", a.get_shape().as_list()[0])
print("number of columns: ", a.get_shape().as_list()[1])



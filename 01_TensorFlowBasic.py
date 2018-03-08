# Besmei Taala
# Amir Hossein Karami 
# MainLink: https://github.com/yunjey/davian-tensorflow/blob/master/notebooks/week1/2.%20tensorflow_basic.ipynb


import tensorflow as tf

# Version of TensorFlow:
print(tf.__version__)  # 1.4.0

# ***** Section1 *****

# Session:
# Session is a class for running TensorFlow operations.

print(' ***** Section1 ***** ')

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

print(' ***** Section2 ***** ')
a = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float32, name='a')
print(a.eval())
print(type(a))
print("shape: ", a.get_shape(), ",type: ", type(a.get_shape()))
print("shape: ", a.get_shape().as_list(), ",type: ", type(a.get_shape().as_list()))    # this is more useful
print("number of rows: ", a.get_shape().as_list()[0])
print("number of columns: ", a.get_shape().as_list()[1])
print(a[0][2].eval())  # Access Tensor's Values
print(type(a[0][0].eval()))  # <class 'numpy.float32'>


# ***** Section3 *****

# Basic functions:

print(' ***** Section3 ***** ')

# tf.argmax:
# returns the index with the largest value across dimensions of a tensor
# Note: I think the dimension order is strange (Be careful)

a = tf.constant([[1, 6, 5], [2, 3, 4]])
print(a.eval())
print("argmax over axis 0")  # axis 0 = columns (Interesting)
print(tf.argmax(a, 0).eval())  # [1 0 0]
print("argmax over axis 1")  # axis 1 = rows (Interesting)
print(tf.argmax(a, 1).eval())  # [1 2]
res1 = tf.argmax(a, 0).eval()
print(type(res1))  # numpy.ndarray

b = tf.constant([[4, 5, 9], [0, 12, 8], [1, 2, 3], [2, 10, 100]])
print(b.eval())
print("argmax over axis 0")  # axis 0 = columns (Interesting)
print(tf.argmax(b, 0).eval())  # [0 1 3]
print("argmax over axis 1")  # axis 1 = rows (Interesting)
print(tf.argmax(b, 1).eval())  # [2 1 2 2]


# tf.reduce_sum:
# computes the sum of elements across dimensions of a tensor.
a = tf.constant([[1, 1, 1], [2, 2, 2]])
print(a.eval())
print("reduce_sum over entire matrix")
print(tf.reduce_sum(a).eval())  # 9
print("reduce_sum over axis 0")
print(tf.reduce_sum(a, 0).eval())  # [3 3 3]
print("reduce_sum over axis 0 + keep dimensions")
print(tf.reduce_sum(a, 0, keep_dims=True).eval())  # [[3 3 3]]
print("reduce_sum over axis 1")
print(tf.reduce_sum(a, 1).eval())  # [3 6]
print("reduce_sum over axis 1 + keep dimensions")
print(tf.reduce_sum(a, 1, keep_dims=True).eval())  # [[3] [6]]

# tf.equal:
# returns the truth value of (x == y) element-wise.
a = tf.constant([[1, 0, 0], [0, 1, 1]])
print(a.eval())
print("Equal to 1?")
print(tf.equal(a, 1).eval())
print("Not equal to 1?")
print(tf.not_equal(a, 1).eval())

# tf.random_normal:
# outputs random values from a normal distribution.
normal1 = tf.random_normal([3], stddev=0.1)
print(normal1.eval())

normal2 = tf.random_normal([2, 3], stddev=0.1)
print(normal2.eval())


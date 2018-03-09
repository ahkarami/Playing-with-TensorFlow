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


# ***** Section4 *****

# Variables:

print(' ***** Section4 ***** ')

# we use variables to hold and update parameters (when training).
# Variables are in-memory buffers containing tensors.

# variable will be initialized with normal distribution
var = tf.Variable(tf.random_normal([3], stddev=0.1), name='var')
print(var.name)
# all variables must be initialized:
# tf.initialize_all_variables().run()  # old TensorFlow versions
tf.global_variables_initializer().run()  # new command
print(var.eval())

# Another Example:
var4 = tf.Variable(tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float32, name='a'), name='var4')
print(var4.name)
tf.global_variables_initializer().run()
print(var4.eval())

# tf.Tensor.name:
# Note that: you should be careful not to call tf.
# Variable giving same name more than once, because it will cause a fatal problem when you save and restore the models.
var2 = tf.Variable(tf.random_normal([2, 3], stddev=0.1), name='my_var')
var3 = tf.Variable(tf.random_normal([2, 3], stddev=0.1), name='my_var')
tf.global_variables_initializer().run()
print(var2.name)
print(var3.name)
print(var2.eval())
print(var3.eval())
# Conclusion: names & values of var2 & var3 are different.

# tf.global_variables  (new version of `tf.all_variables`):
# Using tf.global_variables(), we can get the names of all existing variables as follows:
for var in tf.global_variables():
    print(var.name)


# ***** Section5 *****

# Sharing variables:

print(' ***** Section5 ***** ')

# tf.get_variable:
# "tf.get_variable" is used to get or create a variable instead of a direct call to tf.Variable.
# It uses an initializer instead of passing the value directly, as in tf.Variable.
# some initializers available in TensorFlow:
# 1- tf.constant_initializer(value) initializes everything to the provided value
# 2- tf.random_uniform_initializer(a, b) initializes uniformly from [a, b]
# 3- tf.random_normal_initializer(mean, stddev) initializes from the normal distribution with the given mean and std.

my_initializer = tf.random_normal_initializer(mean=0, stddev=0.1)
v = tf.get_variable('v', shape=[2, 3], initializer=my_initializer)
tf.global_variables_initializer().run()
print('v:\n', v.eval())

my_initializer2 = tf.random_uniform_initializer(0, 10)
v2 = tf.get_variable('v2', shape=[2, 3], initializer=my_initializer2)
tf.global_variables_initializer().run()
print('v2:\n', v2.eval())

# tf.variable_scope:
# manages namespaces for names passed to tf.get_variable.
# In fact, we can add e.g., 'layer1' before variables' names.
with tf.variable_scope('layer1'):
    w0 = tf.get_variable('v0', shape=[2, 3], initializer=my_initializer)
    w1 = tf.get_variable('v1', shape=[2, 3], initializer=my_initializer)
    print(w0.name)
    print(w1.name)

with tf.variable_scope('layer2'):
    w = tf.get_variable('v', shape=[2, 3], initializer=my_initializer)
    print(w.name)

# reuse_variables:
# Note: tf.get_variable doesn't allow creating variables with the existing names.
# So in order to create variables with existing names and add them the name of the scope we can do as follows:
# (i.e.,) We can solve this problem by using scope.reuse_variables() to get preivously created variables instead of
# creating new ones.
with tf.variable_scope('layer1', reuse=True):
    w = tf.get_variable('v0')   # Unlike above, we don't need to specify shape and initializer
    print(w.name)

# or

with tf.variable_scope('layer2') as scope:
    scope.reuse_variables()
    w = tf.get_variable('v')
    print(w.name)


# ***** Section6 *****

# Place holder:

print(' ***** Section6 ***** ')

# TensorFlow provides a placeholder operation that must be fed with data on execution.
# placeholder acts as input data on runtime.
x = tf.placeholder(tf.int16)
y = tf.placeholder(tf.int16)

add = tf.add(x, y)
mul = tf.multiply(x, y)

# Launch default graph.
print("2 + 3 = %d" % sess.run(add, feed_dict={x: 2, y: 3}))
print("3 x 4 = %d" % sess.run(mul, feed_dict={x: 3, y: 4}))

# Another example:
xInput = int(input("x?: "))
yInput = int(input("y?: "))
print("x + y = %d" % sess.run(add, feed_dict={x: xInput, y: yInput}))
print("x x y = %d" % sess.run(mul, feed_dict={x: xInput, y: yInput}))

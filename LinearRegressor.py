'''A linear regression learning algorithm example using TensorFlow library.'''
__author__:"Ankit Rajshree"

#from __future__ import print_function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
rng = np.random

# Parameters
learning_rate = 0.01
no_of_epochs = 1000
display_step = 50

# Training Data
X_TrainData = np.asarray([3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167,
                          7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1])
Y_TrainData = np.asarray([1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221,
                          2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3])
n_samples = X_TrainData.shape[0]

# tf Graph Input
X_Input = tf.placeholder("float")
Y_Input = tf.placeholder("float")

# Set model weights
W = tf.Variable(rng.randn(), name="weight")
b = tf.Variable(rng.randn(), name="bias")

# Construct a linear model
pred = tf.add(tf.multiply(X_Input, W), b)

# Mean squared error
cost = tf.reduce_sum(tf.pow(pred - Y_Input, 2)) / (2 * n_samples)
# Gradient descent
#  Note, minimize() knows to modify W and b because Variable objects are trainable=True by default
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # Fit all training data
    for epoch in range(no_of_epochs):
        for (x, y) in zip(X_TrainData, Y_TrainData):
            sess.run(optimizer, feed_dict={X_Input: x, Y_Input: y})

        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            c = sess.run(cost, feed_dict={X_Input: X_TrainData, Y_Input:Y_TrainData})
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
                "W=", sess.run(W), "b=", sess.run(b))

    print("Optimization Finished!")
    training_cost = sess.run(cost, feed_dict={X_Input: X_TrainData, Y_Input: Y_TrainData})
    print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')

    # Graphic display
    plt.plot(X_TrainData, Y_TrainData, 'ro', label='Original data')
    plt.plot(X_TrainData, sess.run(W) * X_TrainData + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()

    # Testing example, as requested (Issue #2)
    test_X = np.asarray([6.83, 4.668, 8.9, 7.91, 5.7, 8.7, 3.1, 2.1])
    test_Y = np.asarray([1.84, 2.273, 3.2, 2.831, 2.92, 3.24, 1.35, 1.03])

    print("Testing... (Mean square loss Comparison)")
    testing_cost = sess.run(
        tf.reduce_sum(tf.pow(pred - Y_Input, 2)) / (2 * test_X.shape[0]),
        feed_dict={X_Input: test_X, Y_Input: test_Y})  # same function as cost above
    print("Testing cost=", testing_cost)
    print("Absolute mean square loss difference:", abs(
        training_cost - testing_cost))

    plt.plot(test_X, test_Y, 'bo', label='Testing data')
    plt.plot(X_TrainData, sess.run(W) * X_TrainData + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()
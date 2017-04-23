# LSTM Article Parser
# UML Machine Learning, Spring 2017
#
# Authors: Alex Gribov and Saurabh Kulshreshtha
#
# Email: Alexander_Gribov@student.uml.edu
# Email: Saurabh_Kulshreshtha@student.uml.edu

import os
import numpy as np
import tensorflow as tf
import math
import re
import preprocess
from dataset import Dataset

# Declare global variables.

# Directories with questions
testDirectory = "../cnn/questions/test"
trainingDirectory = "../cnn/questions/training"
validationDirectory = "../cnn/questions/validation"
smallDirectory = "../cnn/questions/subset"

#x, y = mnist.train.next_batch(3);
#print x[0];

dict, articleList = preprocess.getTextFromFolder(smallDirectory)
#print(len(dict))

# LSTM parameters
batchSize = 3
LSTM_num_units = 256
n_steps = 20
n_hidden = 128 # hidden layer num of features
n_classes = 1
n_input = 10000

# tf Graph input
# X will be 3D tensor, 1st is articles, then words, then vectorization of word.
x = tf.placeholder("float", [None, n_steps, n_input])
# Y will be a single word from the vocab.
y = tf.placeholder("float", [None, n_classes])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Training method
def trainParser(x, weights, biases):


    # Unpack input tensor:
    x = tf.unstack(x, n_steps, 1)
    
    # Set up LSTM Network
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(LSTM_num_units)
    #state = tf.zeros([batchSize, lstm.state_size])
    """
    state = lstm.zero_state(batchSize, tf.float32)
    probabilities = []
    loss = 0.0
    """
    outputs, state = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']    


    #for article in articleList:
    """
        articleVec = []
        for word in article:
            articleVec.append(dict[word])

        npVec = np.asarray(articleVec)
        inTensor = tf.convert_to_tensor(npVec)
        
        
        #print(articleVec)
        output, state = lstm(x, state)
        #outputs, state = tf.contrib.rnn.static_rnn(lstm, articleVec, dtype=tf.float32)
        outputs.append(output)

        #print(dict)
    
    output = tf.reshape(tf.concat(axis=1, values=outputs), [-1, size])
    softmax_w = tf.get_variable(
        "softmax_w", [size, vocab_size], dtype=data_type())
    softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
    logits = tf.matmul(output, softmax_w) + softmax_b
    loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
        [logits],
        [tf.reshape(input_.targets, [-1])],
        [tf.ones([batch_size * num_steps], dtype=data_type())])
    self._cost = cost = tf.reduce_sum(loss) / batch_size
    self._final_state = state
    """

pred = trainParser(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model CHANGE THIS LATER
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

dict, articleList = preprocess.getTextFromFolder(trainingDir)

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    data = Dataset("../cnn/questions/subset/numbered/", 3)
    
    # Keep training until reach max iterations
    #while step * batch_size < training_iters:
    while not data.is_done:
        batch_x, batch_y = d.next_batch()
        #= mnist.train.next_batch(batch_size)
        # Reshape data to get 28 seq of 28 elements
        batch_x = batch_x.reshape((batch_size, n_steps, n_input))
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    # Calculate accuracy for 128 mnist test images
    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    test_label = mnist.test.labels[:test_len]
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: test_data, y: test_label}))

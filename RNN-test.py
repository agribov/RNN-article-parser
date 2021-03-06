from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn

# Import MNIST data
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import preprocess
from dataset import Dataset

# Directories with questions
testDirectory = "../cnn/questions/test"
trainingDirectory = "../cnn/questions/training"
validationDirectory = "../cnn/questions/validation"
smallDirectory = "../cnn/questions/subset"
smallTest = "../cnn/questions/smallTest"

trainModel = True
modelName = 'model-sample'
saveModel = True
loadModel = False

dir = smallTest

maxS, maxQ, dSize = preprocess.getTextFromFolder(dir)
print("Done with pre-processing\n")

# Parameters
learning_rate = 0.001
training_iters = 100000
batch_size = 10
display_step = 10

# Network Parameters
n_input = 1 # MNIST data input (img shape: 28*28)
n_steps = maxS + maxQ  # timesteps
n_hidden = 128 # hidden layer num of features
#n_classes = 10 # MNIST total classes (0-9 digits)
n_classes = dSize #SIZE OF DICTIONARY

# tf Graph input
#x = tf.placeholder("float", [None, n_steps, n_input])
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    # DON'T THINK WE NEED THIS FOR TEXT PROCESSING
    x = tf.unstack(x, n_steps, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

    # Get lstm cell output
    # NOTE: IS X IN THE RIGHT SHAPE? It used to be 28 input tensors, now it is one long one.
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

print("Building graph...")

pred = RNN(x, weights, biases)
print("Done\n")

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()


# Launch the graph
with tf.Session() as sess:
    print("Initializing session\n")
    
    if loadModel:
        loader = tf.train.import_meta_graph(modelName + '.meta')
        loader.restore(sess, tf.train.latest_checkpoint('./'))
    else:
        sess.run(init)
        
    step = 1
    data = Dataset(dir + "/numbered/", batch_size, n_steps, dSize)
    # Keep training until reach max iterations
    #while step * batch_size < training_iters:
    if trainModel:
        while data.done != 1:
            batch_x, batch_y = data.next_batch()
            #batch_x, batch_y = mnist.train.next_batch(batch_size)
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
        
    if saveModel:
        saver = tf.train.Saver()
        saver.save(sess, modelName)


    """
    # Calculate accuracy for 128 mnist test images
    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    test_label = mnist.test.labels[:test_len]
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: test_data, y: test_label}))
    """

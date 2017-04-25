from __future__ import print_function

import os
import tensorflow as tf
from tensorflow.contrib import rnn

# Import MNIST data
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import preprocess
from dataset import Dataset

import pickle
import shutil

# Directories with questions
testDir = "../cnn/questions/test"
trainingDir = "../cnn/questions/training"
validationDir = "../cnn/questions/validation"
smallDir = "../cnn/questions/subset"
smallTest = "../cnn/questions/smallTest"

preprocess_files = True
trainModel = True
modelDir = './models/'
modelName = 'model-full-5epocs'
saveModel = True
loadModel = False
validation = True

if preprocess_files:
    print("Starting preprocessor")
    dirs = (testDir, trainingDir, validationDir)
    maxS, maxQ, dSize = preprocess.getTextFromFolder(dirs)
    
    try:
        os.stat(modelDir + modelName)
    except:
        os.mkdir(modelDir + modelName)
    shutil.copy2('params.p', modelDir + modelName + "/params.p")
    print("Done with pre-processing\n")
else:
    params = pickle.load(open(modelDir + modelName + "/params.p", "rb") )
    maxS = params[0]
    maxQ = params[1]
    dict = params[2]
    dSize = params[3]

# Parameters
learning_rate = 0.00005
batch_size = 256
display_step = 10
val_size = 1000

# Network Parameters
n_input = 1 # MNIST data input (img shape: 28*28)
n_steps = maxS + maxQ  # timesteps
n_hidden = 256 # hidden layer num of features
n_classes = dSize #SIZE OF DICTIONARY
dropout = 0.5
embedding_size = 50
num_layers = 3

# tf Graph input
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
    lstm_cell = rnn.DropoutWrapper(lstm_cell, output_keep_prob=dropout)
    
    #stacked_lstm = rnn.MultiRNNCell([rnn.DropoutWrapper(rnn.BasicLSTMCell(n_hidden, forget_bias=1.0), output_keep_prob=dropout) for _ in range(num_layers)], state_is_tuple=True)
    #initial_state = state = stacked_lstm.zero_state(batch_size, tf.float32)

    # Get lstm cell output
    # NOTE: IS X IN THE RIGHT SHAPE? It used to be 28 input tensors, now it is one long one.
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

print("Building graph...")
#embeddings = tf.Variable(tf.random_uniform([dSize, embedding_size], -1.0, 1.0))
#embed = tf.nn.embedding_lookup(embeddings, x)

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
    dir = trainingDir
    if loadModel:
        loader = tf.train.Saver()
        loader.restore(sess, modelDir + modelName + '/' + modelName)
        #loader = tf.train.import_meta_graph(modelDir + modelName + '/' + modelName + '.meta')
        #loader.restore(sess, tf.train.latest_checkpoint(modelDir + modelName))
    else:
        sess.run(init)
        
    step = 1
    data = Dataset(dir + "/numbered/", batch_size, n_steps, dSize)
    # Keep training until reach max iterations
    #while step * batch_size < training_iters:
    if trainModel:
        while data.done < 1:
            batch_x, batch_y = data.next_batch()
            #batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Redatashape data to get 28 seq of 28 elements
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
                      "{:.5f}".format(acc) + ", Epoch " + str(data.done))
            step += 1
        print("Optimization Finished!")
        
    if saveModel:
        try:
            os.stat(modelDir + modelName)
        except:
            os.mkdir(modelDir + modelName)
        
        saver = tf.train.Saver()
        saver.save(sess, modelDir + modelName + '/' + modelName)


    # VALIDATION:
    if validation:
        valData = Dataset(validationDir + "/numbered/", val_size, n_steps, dSize)
        text_x, test_y = valData.next_batch()
        test_x = text_x.reshape((val_size, n_steps, n_input))

        print("Running validation: ")
        print("Validation accuracy is: ",\
              sess.run(accuracy, feed_dict={x: test_x, y: test_y}))



    
    """
    # Calculate accuracy for 128 mnist test images
    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    test_label = mnist.test.labels[:test_len]
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: test_data, y: test_label}))
    """

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

# Declare global variables.

# Directories with questions
testDirectory = "../cnn/questions/test"
trainingDirectory = "../cnn/questions/training"
validationDirectory = "../cnn/questions/validation"
smallDirectory = "../cnn/questions/subset"

# LSTM parameters
batchSize = 1 # this is basically hidden layer number of features
LSTM_num_units = 10000

# Training method
def trainParser(trainingDir):

    dict, articleList = getTextFromFolder(trainingDir)
    print(dict)
    
    # Set up LSTM Network
    lstm = tf.contrib.rnn.BasicLSTMCell(LSTM_num_units)
    #state = tf.zeros([batchSize, lstm.state_size])
    state = lstm.zero_state(batchSize, tf.float32)
    probabilities = []
    loss = 0.0

    outputs = []
    
    for article in articleList:
        articleVec = []
        for word in article:
            articleVec.append(dict[word])

            npVec = np.asarray(articleVec)
        #print(articleVec)
        output, state = lstm(npVec, state)
        #outputs, state = tf.contrib.rnn.static_rnn(lstm, articleVec, dtype=tf.float32)
        outputs.append(output)

        #print(dict)
    """
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

        
        
# This function removes unnecessary words from the file. Current removed are:
#    - URL (Word starting with http)
#    - NULL words
#    - Entities

def getTextFromFolder(directory):
    # Declare dictionary for holding words
    # Move declaration into for loop if it
    # should reset with every new file.
    dict = {}
    
    # Declare word counter (keep with dict)
    nWords = 0;

    # Declare a list of articles. Each member (article) will be a list of words.
    articles = []


    for filename in os.listdir(directory):
        words = []
        if filename.endswith(".question"):
            print "Processing file: "
            print filename
            with open(os.path.join(directory, filename), 'r') as f:
                for line in f:
                    for word in re.split(';|,|\*|\n| ', line):
                        #print(word)
                        words.append(word)
                filterOutCrap(words)
                #print(words)

                # Add words to dictionary
                for word in words:
                    if not dict.has_key(word) and isinstance(word, basestring):
                        dict.setdefault(word, nWords);
                        nWords += 1
                #print dict   

                # Add this article to the list of articles.
                articles.append(words)
                
            # Comment out break if you want to run on more than one file
            #break
    
    return dict, articles


def filterOutCrap(text):
    for word in text:
        if not (word):
            #This should just remove NULL words -- why does uncommenting that stop entities from being removed? Because of the indexing?
            #text.remove(word)
            continue
        elif word[0] == '@':
            text.remove(word)
            continue
        elif len(word) >= 4:
            if word[0:4] == 'http':
                text.remove(word)

    
            
trainParser(smallDirectory);

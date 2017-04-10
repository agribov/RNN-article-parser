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

testDirectory = "../cnn/questions/test"
trainingDirectory = "../cnn/questions/training"
validationDirectory = "../cnn/questions/validation"

# Training method
def trainParser(trainingDir):
    # Declare dictionary for holding words
    # Move declaration into for loop if it
    # should reset with every new file.
    dict = {}
    
    # Declare word counter (keep with dict)
    nWords = 0;
    
    for filename in os.listdir(trainingDir):
        words = []
        if filename.endswith(".question"):
            print filename
            with open(os.path.join(trainingDir, filename), 'r') as f:
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
                print dict   
                
            # Comment out break if you want to run on more than one file
            break

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

    
            
trainParser(testDirectory);
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
import pickle

# This function removes unnecessary words from the file. Current removed are:
#    - URL (Word starting with http)
#    - NULL words
#    - Entities

def getTextFromFolder(directories):
    # Declare dictionary for holding words
    # Move declaration into for loop if it
    # should reset with every new file.
    dict = {}
    
    # Declare word counter (keep with dict)
    nWords = 0;
    maxS = 0
    maxQ = 0

    # Declare a list of articles. Each member (article) will be a list of words.
    articles = []

    print "Processing file: "
    for directory in directories:
        for filename in os.listdir(directory):
            words = []
            if filename.endswith(".question"):
                print '{0}\r'.format(filename),
                with open(os.path.join(directory, filename), 'r') as f:
                    lineCount = 0;
                    text = []
                    for line in f:
                        words = []
                        lineCount +=1
                        #print lineCount
                        for word in re.split(';|,|\*|\n| ', line):
                            #print(word)
                            words.append(word)

                        # Filter function removes delimiters
                        filterOutCrap(words)

                        # Line three is the story, line 5 is the question
                        if lineCount == 3 or lineCount == 5 or lineCount == 7:
                            #Text is a 3-tuple: first part is story, second is question, third is ans
                            #print len(words)
                            text.append(words)
                            # Add words to dictionary
                            for word in words:
                                if not dict.has_key(word) and isinstance(word, basestring):
                                    dict.setdefault(word, nWords);
                                    nWords += 1

                            if lineCount == 3:
                                maxS = max(maxS, len(words))
                            elif lineCount == 5:
                                maxQ = max(maxQ, len(words))
                            #print text
                    # Add this article to the list of articles.
                    # Each article is the 3-tuple described above.
                    articles.append(text)
                    #print articles[0][0][10]
        saveArticlesAsNumbers(articles, dict, directory)
    print
    pickle.dump((maxS, maxQ, dict, nWords), open("params.p", "wb"))
    return maxS, maxQ, nWords

def saveArticlesAsNumbers(articles, dict, dir):
    
    fileNumber = 0
    dir = dir + "/numbered"

    try:
        os.stat(dir)
    except:
        os.mkdir(dir)
    
    for article in articles:
        filename = str(fileNumber) + ".txt"
        #print article[1]
        f =  open(os.path.join(dir, filename), 'w')
        for line in article:
            #print len(line)
            for word in line:
                f.write(str(dict[word]))
                f.write(" ")
            f.write("\n")
        f.close()

        fileNumber += 1
        

def filterOutCrap(text):
    for word in text:
        if not (word):
            #This should just remove NULL words -- why does uncommenting that stop entities from being removed? Because of the indexing?
            #text.remove(word)
            continue
        elif word[0] == '@':
            #text.remove(word)
            continue
        elif len(word) >= 4:
            if word[0:4] == 'http':
                text.remove(word)


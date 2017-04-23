# grab a bunch of files and return tensors 

# so the way it is going to work is we have a class
# this class is init with the dir name for all the token id files

# this class has then one of its variables set to the batch size

# this class is finally going to have a function that picks batch size number of files and convert

import numpy as np
import glob
from itertools import izip_longest

class Dataset:

    def __init__(self, data_dir, batch_size, max_x, dSize):        
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.dict_size = dSize
        self.read = 0
        self.done = 0
        self.pad_constant = -1
        self.MAX_X = max_x # Change this so it just fits the largest article + story.
        self.list_files()
        self.list_batches()
        return

    def list_files(self):
        self.file_list = glob.glob(self.data_dir + "*.txt")
        return

    def grouper(self, iterable, n, fillvalue=None):
        '''
        Example:
                grouper('ABCDEFG', 3, 'x') --> 'ABC' 'DEF' 'Gxx'
        '''
        args = [iter(iterable)] * n
        return izip_longest(*args, fillvalue=fillvalue)

    def list_batches(self):
        self.batches = list(self.grouper(self.file_list, self.batch_size))
        return

    def next_batch_file_list(self):
        batch_fl = self.batches[self.read]
        self.read += 1
        print("Fetching batch " + str(self.read) + " out of " + str(len(self.batches)))

        if self.read == len(self.batches):
            self.read = 0 #CHANGE THIS LATER: Need to tell next_batch function that
            # We have hit the last batch
            print "Reached last batch"
            self.done = 1
        return batch_fl

    def is_done(self):
        return data.done

    def next_batch(self):
        batch_fl = self.next_batch_file_list()

        # Create numpy arrays for returning. Batch size and max length determined in __init__.
        self.snq = np.zeros((self.batch_size, self.MAX_X), dtype=np.int)
        self.answers = np.zeros((self.batch_size, self.dict_size), dtype=np.int)
        bNum = 0
        
        for file_path in batch_fl:
            if file_path != None:
                lines = [line.rstrip('\n') for line in open(file_path, 'r')]
                if len(lines) == 3:
                    story = np.array([int(e) for e in lines[0].split()])
                    question = np.array([int(e) for e in lines[1].split()])
                    self.answers[bNum][int(lines[2].split()[0])] = 1

                    # Pad the end of the array with constants (defined in __init__)
                    padSize = self.MAX_X - story.size - question.size
                    entry = np.concatenate((story, question), axis=0)
                    entry = np.lib.pad(entry, (0 ,padSize), 'constant', constant_values=self.pad_constant)
                    self.snq[bNum] = entry

                    bNum += 1

        #print self.snq.shape, self.answers.shape       
        return (self.snq, self.answers)

"""
    def next_batch(self):
        batch_fl = self.next_batch_file_list()
        self.stories = []
        self.questions = []
        self.answers = []
        self.qna = []

        for file_path in batch_fl:
            if file_path != None:
                lines = [line.rstrip('\n') for line in open(file_path, 'r')]
                if len(lines) == 3:
                    self.stories.append(np.array([int(e) for e in lines[0].split()]))
                    self.questions.append(np.array([int(e) for e in lines[1].split()]))
                    self.answers.append(np.array([int(e) for e in lines[2].split()]))
                    print self.questions[0].shape
            else:
                self.stories.append([])
                self.questions.append([])
                self.answers.append([])
        self.questions = np.array(self.questions)
        self.stories = np.array(self.stories)
        self.answers = np.array(self.answers)
        print self.questions.shape
        print self.stories.shape
        self.snq = np.concatenate((self.stories, self.questions), axis=1)
        self.qns = np.concatenate((self.questions, self.stories), axis=1)


        return (self.snq, self.answers)

"""


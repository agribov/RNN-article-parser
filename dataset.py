# grab a bunch of files and return tensors 

# so the way it is going to work is we have a class
# this class is init with the dir name for all the token id files

# this class has then one of its variables set to the batch size

# this class is finally going to have a function that picks batch size number of files and convert

import numpy as np
import glob
from itertools import izip_longest

class Dataset:

    def __init__(self, data_dir, batch_size):        
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.read = 0
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
        return batch_fl

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
                    self.stories.append([int(e) for e in lines[0].split()])
                    self.questions.append([int(e) for e in lines[1].split()])
                    self.answers.append([int(e) for e in lines[2].split()])
            else:
                self.stories.append([])
                self.questions.append([])
                self.answers.append([])
        self.questions = np.array(self.questions)
        self.stories = np.array(self.stories)
        self.answers = np.array(self.answers)
        self.snq = np.concatenate((self.stories, self.questions), axis=1)
        self.qns = np.concatenate((self.questions, self.stories), axis=1)

        return (self.snq, self.answers)




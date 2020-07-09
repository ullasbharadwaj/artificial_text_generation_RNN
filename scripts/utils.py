
"""
######################################################################
            Script to gather all the information from the
            given text file. This module is required both
            in training and inference stages

            Author: Ullas Bharadwaj
######################################################################
"""

"""
Import all the required packages
"""
import numpy as np
from collections import Counter
import os
from argparse import Namespace
import ipdb
import glob
import pickle

"""
Function to get input data, true label vector, number of words info from the Input file used for training
"""

def ReadPickle(filename):
        f = open(filename,"rb")
        data = pickle.load(f)
        f.close()
        return data

def WritePickle(data, outfile):
        f = open(outfile, "w+b")
        pickle.dump(data, f)
        f.close()

def Get_Info_From_File(train_file, batch_size, seq_size):
    with open(train_file, 'r', encoding='utf-8') as f:
        text = f.read()

    text = text.split()
    word_counts = Counter(text)
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)

    # Getting the integer and word representations from the input
    int_to_vocab = {k: w for k, w in enumerate(sorted_vocab)}
    vocab_to_int = {w: k for k, w in int_to_vocab.items()}
    number_vocab = len(int_to_vocab)

    # Converting the set of words in the input text to integer representations
    int_text = [vocab_to_int[w] for w in text]

    #Calculate the number of batches from the calculated number of words
    number_batches = int(len(int_text) / (seq_size * batch_size))

    #Generate the Input training data in the integer form
    X_text = int_text[:number_batches * batch_size * seq_size]

    #Generate the True Text Vector cooresponding to X_test
    Y_text = np.zeros_like(X_text)
    Y_text[:-1] = X_text[1:]
    Y_text[-1] = X_text[0]
    X_text = np.reshape(X_text, (batch_size, -1))
    Y_text = np.reshape(Y_text, (batch_size, -1))

    return int_to_vocab, vocab_to_int, number_vocab, X_text, Y_text

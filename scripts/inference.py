"""
######################################################################
            Script to perform inference once the text
            generation model is trained

            Author: Ullas Bharadwaj
######################################################################
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import Counter
import os
import argparse
from argparse import Namespace
import ipdb
import glob
import utils
from train import RNNModule
from train import predict

"""
Test routine to generate the texts using the trained LSTM network. The 'Predict' subroutine is invoked from this function.
"""
def Test(initial_words, int_to_vocab, vocab_to_int, n_vocab, number_of_words, number_of_sentences):
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device= 'cpu'

    #Load the latest trained model in the directory storing the trained models
    list_of_models= glob.glob('../checkpoint/*.pth')
    latest_model = max(list_of_models, key=os.path.getctime)

    for i in range(number_of_sentences):
        #Instantiate the RNN module
        net = RNNModule(n_vocab , flags.seq_size, flags.embedding_size, flags.lstm_size)
        #Load the state dictionary of the trained module
        net_trained = net.load_state_dict(torch.load(latest_model))
        #Transfer the model to GPU/CPU based on availability
        net = net.to(device)
        #Put the model in the evaluation mode
        net.eval()

        #Option to chose the starting two words of the sentence

        predict(device, net, initial_words, number_of_words, vocab_to_int, int_to_vocab, top_k=5)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='textGen')

    parser.add_argument(
        '--start_sequence',
        help='Start of sentence',
        type=str,
        default='This is a test'
    )

    parser.add_argument(
        '--num_words',
        help='No of words to predict',
        type=int,
        default='5'
    )

    parser.add_argument(
        '--num_sentences',
        help='No of sentences to predict',
        type=int,
        default='1'
    )

    args = parser.parse_args()

    flags = Namespace(
        train_file='../data/train.txt',
        seq_size=16,
        batch_size=8,
        embedding_size=64,
        lstm_size=64,
        gradients_norm=5,
        initial_words=['There', 'is'],
        predict_top_k=6,
        checkpoint_path='../checkpoint',
    )
    start_seq = args.start_sequence.split()
    num_words = args.num_words
    num_sentences = args.num_sentences

    int_to_vocab = utils.ReadPickle("../keyData/int_to_vocab.pickle")
    vocab_to_int = utils.ReadPickle("../keyData/vocab_to_int.pickle")
    n_vocab = len(int_to_vocab)
    Test(start_seq, int_to_vocab, vocab_to_int, n_vocab, num_words, num_sentences)

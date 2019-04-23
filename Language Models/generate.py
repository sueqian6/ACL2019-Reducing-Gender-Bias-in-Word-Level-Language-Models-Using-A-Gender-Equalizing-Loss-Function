
# coding: utf-8

##############################################################################
#language Modeling on Penn Tree Bank
#
# This file generates new sentences sampled from the language model
#
###############################################################################

import argparse

import torch
from torch.autograd import Variable

import data_v3
import json
import pandas as pd
import preprocess
import os
from sklearn.model_selection import ShuffleSplit
import random


parser = argparse.ArgumentParser(description='PyTorch bbc Language Model')

# Model parameters.
parser.add_argument('--data', type=str, default='./data/preprocessed/data',
                    help='location of the data corpus')
parser.add_argument('--checkpointpath', type=str, default='./savedmodel',
                    help='model checkpoint to use')
parser.add_argument('--outDir', type=str, default='./data/generated',
                    help='number of output file for generated text')
parser.add_argument('--words', type=int, default='500',
                    help='words to generate')
parser.add_argument('--documents', type=int, default='4000',
                    help='number of files to generate')
parser.add_argument('--no-sentence-reset', default=False, 
                    help='do not reset the hidden state in between sentences') #action='store_true',
parser.add_argument('--seed', type=int, default=20190331,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
parser.add_argument('--log-interval', type=int, default=100,
                    help='reporting interval')
parser.add_argument('--emsize', type=int, default=300,
                    help='size of word embeddings')
parser.add_argument('--vocab', type=str, default='./data/preprocessed/VOCAB.txt',
                    help=('preprocessed vocaburary'))
parser.add_argument('--glove_path', type=str, default='./gn_glove/1b-vectors300-0.8-0.8.txt',
                    help='using glove word embedding')
args = parser.parse_args()


# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

if args.temperature < 1e-3:
    parser.error("--temperature has to be greater or equal 1e-3")


#load vocab            
vocab = preprocess.read_vocab(os.path.join(args.vocab))
if 'cda' in args.vocab:
    idx_train = pd.read_json('idx_train_cda.json')
    idx_val = pd.read_json('idx_val_cda.json')
    idx_test = pd.read_json('idx_test_cda.json')
else:
    idx_train = pd.read_json('idx_train.json')
    idx_val = pd.read_json('idx_val.json')
    idx_test = pd.read_json('idx_test.json')

# Load pretrained Embeddings, common token of vocab and gn_glove will be loaded, only appear in vocab will be initialized
vocab.append('<eos>')
ntokens = len(vocab)

words2idx = {item : index for index, item in enumerate(vocab)}

# Load data
corpus = data_v3.Corpus(args.data, vocab, words2idx, idx_train, idx_val, idx_test) #改动2 

def generateFile(outf, model):
    # torch.initial_seed()
    hidden = model.init_hidden(1)
    input = torch.randint(ntokens, (1, 1), dtype=torch.long).to(device)

    with open(outf, 'w') as outf:       
        with torch.no_grad():  # no tracking history
            for i in range(args.words):
                output, hidden = model(input, hidden)
                word_weights = output.squeeze().div(args.temperature).exp().cpu()
                word_idx = torch.multinomial(word_weights, 1)[0]
                input.fill_(word_idx)
                word = corpus.idx2words[word_idx]

                outf.write(word + ('\n' if i % 20 == 19 else ' '))

def generateData(checkpoint, model_out_dir):
    with open(checkpoint, 'rb') as f:
        model = torch.load(f).to(device)
    model.eval()

    for i in range(args.documents):
        outf = model_out_dir + '/' + str(i+1) + '.txt'
        generateFile(outf, model)

        if i % 1000 == 0:
            print('| Generated {}/{} documents'.format(i, args.documents))


modelDir = args.checkpointpath
modelFiles = [m for m in os.listdir(modelDir) if m.endswith('.pt')]


#generate files for each saved models
outDir = args.outDir
if not os.path.isdir(outDir):
    os.makedirs(outDir)
for m in modelFiles:
    print('processing: '+m)
    modelName = '.'.join(m.split('.')[:-1])
    checkpoint = os.path.join(modelDir,m)
    model_out_dir = os.path.join(outDir,modelName)
    if not os.path.isdir(model_out_dir):
        os.makedirs(model_out_dir)
    generateData(checkpoint, model_out_dir)

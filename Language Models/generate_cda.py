
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
parser.add_argument('--data', type=str, default='./data/',
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
args = parser.parse_args()


# Set the random seed manually for reproducibility.
# torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

if args.temperature < 1e-3:
    parser.error("--temperature has to be greater or equal 1e-3")


vocab = preprocess.read_vocab(os.path.join(args.data,'VOCAB_cda.txt'))
idx_train = pd.read_json('idx_train_cda.json')
idx_val = pd.read_json('idx_val_cda.json')
idx_test = pd.read_json('idx_test_cda.json')

# Load pretrained Embeddings, common token of vocab and gn_glove will be loaded, only appear in vocab will be initialized
gn_glove_dir = './gn_glove/1b-vectors300-0.8-0.8.txt' #142527 tokens, last one is '<unk>'
# ntokens = sum(1 for line in open(gn_glove_dir)) + 1
vocab.append('<eos>')
ntokens = len(vocab)

words2idx = {item : index for index, item in enumerate(vocab)}

# Load data
corpus = data_v3.Corpus(os.path.join(args.data,'outputdata_cda'), vocab, words2idx, idx_train, idx_val, idx_test) #改动2 
# hidden = model.init_hidden(1)
# input = Variable(torch.rand(1, 1).mul(ntokens).long(), volatile=True)


def generateFile(outf, model):
    # torch.initial_seed()
    hidden = model.init_hidden(1)
    input = torch.randint(ntokens, (1, 1), dtype=torch.long)
    print(input)
    if args.cuda:
        input.data = input.data.cuda()

    with open(outf, 'w') as outf:       
        with torch.no_grad():  # no tracking history
            for i in range(args.words):
                output, hidden = model(input, hidden)
                word_weights = output.squeeze().div(args.temperature).exp().cpu()
                word_idx = torch.multinomial(word_weights, 1)[0]
                input.fill_(word_idx)
                word = corpus.idx2words[word_idx]

                outf.write(word + ('\n' if i % 20 == 19 else ' '))

                #if i % log_interval == 0:
                #    print('| Generated {}/{} words'.format(i, words))

                if i % args.log_interval == 0:
                    print('| Generated {}/{} words'.format(i, args.words))

def generateData(checkpoint, model_out_dir):
    with open(checkpoint, 'rb') as f:
        model = torch.load(f)
    model.eval()

    if args.cuda:
        model.cuda()
    else:
        model.cpu()

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)

    for i in range(args.documents):
        outf = model_out_dir + '/' + str(i+1) + '.txt'
        generateFile(outf, model)


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

# sentences = []
# sent = []

# with open(args.outf, 'w') as outf:
#     for i in range(args.words):
#         output, hidden = model(input, hidden)
#         word_weights = output.squeeze().data.div(args.temperature).exp().cpu()
#         word_idx = torch.multinomial(word_weights, 1)[0]
#         input.data.fill_(word_idx)
#         word = corpus.idx2words[word_idx]
#         sent.append(word)

#         buf = word
#         if (i + 1) % 20 == 0:
#             buf += '\n'
#         else:
#             buf += ' '
#         outf.write(buf)


#         if word == '.':
#             if not args.no_sentence_reset:
#                 hidden = model.init_hidden(1)
#             sentences.append(sent)
#             sent = []

#         if i % args.log_interval == 0:
#             print('| Generated {}/{} words'.format(i, args.words))


# if sent:
#     sentences.append(sent)

# # Compute bias metrics
# female_cooccur, male_cooccur = cb.get_sentence_list_gender_cooccurrences(sentences)
# bias, bias_norm = cb.compute_gender_cooccurrance_bias(female_cooccur, male_cooccur)
# gdd = cb.compute_gender_distribution_divergence(female_cooccur, male_cooccur)

# print('Gender Co-occurrence Bias: {}'.format(bias))
# print('Gender Co-occurrence Bias (normalized): {}'.format(bias_norm))
# print('Gender Distribution Divergence: {}'.format(gdd))


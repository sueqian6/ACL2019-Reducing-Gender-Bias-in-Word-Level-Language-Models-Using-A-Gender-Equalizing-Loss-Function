# coding: utf-8
import argparse
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable

import data_v3
import model
import preprocess
# import jams
import os
import pandas as pd
from sklearn.model_selection import ShuffleSplit
import json

import random
seed = random.seed(20180330)

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='../data/preprocessed/data',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=300,
                    help='size of word embeddings')
parser.add_argument('--glove', action='store_true',
                    help='use glove')
parser.add_argument('--glove_path', type=str, default='../gn_glove/1b-vectors300-0.8-0.8.txt',
                    help='using glove word embedding')
parser.add_argument('--nhid', type=int, default=300,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=5,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')

parser.add_argument('--bias_reg_encoder', action='store_true',
                    help='use bias regularization encoder')
parser.add_argument('--bias_reg_decoder', action='store_true',
                    help='use bias regularization decoder')
parser.add_argument('--bias_reg_en_factor', type=float, default=1.0,
                    help='bias regularization encoder loss weight factor')
parser.add_argument('--bias_reg_de_factor', type=float, default=1.0,
                    help='bias regularization decoder loss weight factor')

parser.add_argument('--bias_reg_var_ratio', type=float, default=0.5,
                    help=('ratio of variance used for determining size of gender'
                          'subspace for bias regularization'))
parser.add_argument('--unnorm-bias', dest='norm_bias', action='store_false',
                    help='If set, do not normalize embedding weights before computing bias score')

parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=20180330,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str,  default='../model_reg.pt',
                    help='path to save the final model')

parser.add_argument('--vocab', type=str, default='../data/preprocessed/VOCAB.txt',
                    help=('preprocessed vocaburary'))
parser.add_argument('--anneal', type=float, default=4,
                    help='anneal rate of learning rate')

args = parser.parse_args()


# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if args.cuda:
        data = data.cuda()
    return data
# def repackage_hidden(h):
#     #"""Wraps hidden states in new Variables, to detach them from their history."""
#     if type(h) == Variable:
#         return Variable(h.data)
#     else:
#         return tuple(repackage_hidden(v) for v in h)


# 改动1
def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def get_batch(source, i, evaluation=False):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = Variable(source[i:i+seq_len], volatile=evaluation)
    target = Variable(source[i+1:i+1+seq_len].view(-1))
    return data, target

# def evaluate(data_source):
#     # Turn on evaluation mode which disables dropout.
#     model.eval()
#     total_loss = 0
#     ntokens = len(corpus.dictionary)
#     hidden = model.init_hidden(eval_batch_size)
#     for i in range(0, data_source.size(0) - 1, args.bptt):
#         data, targets = get_batch(data_source, i, evaluation=True)
#         output, hidden = model(data, hidden)
#         output_flat = output.view(-1, ntokens)
#         total_loss += len(data) * criterion(output_flat, targets).data
#         hidden = repackage_hidden(hidden)
#     return total_loss[0] / len(data_source)

# def evaluate(data_source):
#     # Turn on evaluation mode which disables dropout.
#     model.eval()
#     total_loss = 0
#     ntokens = len(corpus.dictionary)
#     hidden = model.init_hidden(eval_batch_size)
#     for i in range(0, data_source.size(0) - 1, args.bptt):
#         data, targets = get_batch(data_source, i, evaluation=True)
#         output, hidden = model(data, hidden)
#         output_flat = output.view(-1, ntokens)
#         total_loss += len(data) * criterion(output_flat, targets).data
#         hidden = repackage_hidden(hidden)
#     return total_loss.item() / len(data_source)


# 改动2 according newest pytorch LM tempelator
def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i)
            output, hidden = model(data, hidden)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
            hidden = repackage_hidden(hidden)
    return total_loss / (len(data_source) - 1)



def load_glove_to_dict(path, emsize):
    with open(path, 'r') as fp:
        glove = fp.readlines()
    glove_dict = {}
    for string in glove:
        vec = string.split()
        glove_dict[vec[0]] = []
        for i in range(emsize):
            glove_dict[vec[0]].append(float(vec[i+1]))
    return glove_dict

def glove_dict_to_tensor(word2idx_dict, glove_dict):
    num_words = len(word2idx_dict)
    emb_dim = len(list(glove_dict.values())[0])
    glove_tensor = torch.FloatTensor(num_words, emb_dim)

    for word in word2idx_dict:
        idx = word2idx_dict[word]
        glove_tensor[idx] = torch.FloatTensor(glove_dict[word])

    return glove_tensor


def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        model.zero_grad()
        output, hidden = model(data, hidden)
        loss = criterion(output.view(-1, ntokens), targets)
        
        if args.bias_reg_encoder: 
            bias_loss = bias_regularization_encoder(model, D, N, args.bias_reg_var_ratio, args.bias_reg_en_factor, norm=args.norm_bias) 
            loss = loss + bias_loss ;

        if args.bias_reg_decoder: 
            bias_loss = bias_regularization_decoder(model, D, N, args.bias_reg_var_ratio, args.bias_reg_de_factor, norm=args.norm_bias) 
            loss = loss + bias_loss ;

        # if args.bias_reg:
        #     bias_loss = bias_regularization(model, D, N, args.bias_reg_var_ratio,
        #                                     args.bias_reg_factor)
        #     print(bias_loss)
        #     bias_loss.backward()
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)

        total_loss += loss.data

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval   #total_loss.[0]
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

#load vocab
vocab = preprocess.read_vocab(os.path.join(args.vocab))

#create json file with indexed filename for following separation
# inds = jams.util.find_with_extension(args.data, 'bin')
inds = [os.path.join(args.data, x) for x in os.listdir(args.data) if x.endswith('bin')]

index_train = {}
index_train['id'] = {}
iteration = 0
for ind in inds:
    index_train['id'][iteration] = os.path.basename(ind)
    iteration += 1

with open('ind_train.json', 'w') as fp:
    json.dump(index_train, fp)

#load the json file of indexed filename
with open('ind_train.json', 'r') as fp:
    data = json.load(fp)
idx_train_ = pd.DataFrame(data)

#split test set
splitter_tt = ShuffleSplit(n_splits=1, test_size=0.1,
                               random_state=seed)
bigtrains, tests = next(splitter_tt.split(idx_train_['id'].keys()))

idx_bigtrain = idx_train_.iloc[bigtrains]
idx_test = idx_train_.iloc[tests]

#split train, val sets
splitter_tv = ShuffleSplit(n_splits=1, test_size=0.2,
                               random_state=seed)

trains, vals = next(splitter_tv.split(idx_bigtrain['id'].keys()))

idx_train = idx_bigtrain.iloc[trains]
idx_val = idx_bigtrain.iloc[vals]

#save idx_train, idx_val, idx_test for later use
idx_train.to_json('idx_train_t.json')
idx_val.to_json('idx_val_t.json')
idx_test.to_json('idx_test_t.json')

# Load data
corpus = data_v3.Corpus(args.data, vocab, idx_train, idx_val, idx_test)



female_words = {
    'woman', 'women', 'ladies', 'female', 'females', 'girl', 'girlfriend',
    'girlfriends', 'girls', 'her', 'hers', 'lady', 'she', 'wife', 'wives'
}

male_words = {
    'gentleman', 'man', 'men', 'gentlemen', 'male', 'males', 'boy', 'boyfriend',
    'boyfriends', 'boys', 'he', 'his', 'him', 'husband', 'husbands'
}

gender_words = female_words | male_words

word2idx = corpus.dictionary.word2idx
D = torch.LongTensor([[word2idx[wf], word2idx[wm]]
                       for wf, wm in zip(female_words, male_words)
                       if wf in word2idx and wm in word2idx])

# Probably will want to make this better
N = torch.LongTensor([idx for w, idx in word2idx.items() if w not in gender_words])

if args.cuda:
    D = D.cuda()
    N = N.cuda()

eos_idx = corpus.dictionary.word2idx['<eos>']

def bias_regularization_encoder(model, D, N, var_ratio, lmbda, norm=True):
    """
    Compute bias regularization loss term
    """
    W = model.encoder.weight
    if norm:
        W = W / model.encoder.weight.norm(2, dim=1).view(-1, 1)

    C = []
    # Stack all of the differences between the gender pairs
    for idx in range(D.size()[0]):
        idxs = D[idx].view(-1)
        u = W[idxs[0],:]
        v = W[idxs[1],:]
        C.append(((u - v)/2).view(1, -1))
    C = torch.cat(C, dim=0)

    # Get prinipal components
    U, S, V = torch.svd(C)

    # Find k such that we capture 100*var_ratio% of the gender variance
    var = S**2

    norm_var = var/var.sum()
    cumul_norm_var = torch.cumsum(norm_var, dim=0)
    _, k_idx = cumul_norm_var[cumul_norm_var >= var_ratio].min(dim=0)

    # Get first k components to for gender subspace
    B = V[:, :k_idx.item()+1]
    loss = torch.matmul(W[N], B).norm(2) ** 2

    return lmbda * loss

def bias_regularization_decoder(model, D, N, var_ratio, lmbda, norm=True):
    """
    Compute bias regularization loss term
    """
    W = model.decoder.weight
    if norm:
        W = W / model.decoder.weight.norm(2, dim=1).view(-1, 1)

    C = []
    # Stack all of the differences between the gender pairs
    for idx in range(D.size()[0]):
        idxs = D[idx].view(-1)
        u = W[idxs[0],:]
        v = W[idxs[1],:]
        C.append(((u - v)/2).view(1, -1))
    C = torch.cat(C, dim=0)

    # Get prinipal components
    U, S, V = torch.svd(C)

    # Find k such that we capture 100*var_ratio% of the gender variance
    var = S**2

    norm_var = var/var.sum()
    cumul_norm_var = torch.cumsum(norm_var, dim=0)
    _, k_idx = cumul_norm_var[cumul_norm_var >= var_ratio].min(dim=0)

    # Get first k components to for gender subspace
    B = V[:, :k_idx.item()+1]
    loss = torch.matmul(W[N], B).norm(2) ** 2

    return lmbda * loss


eval_batch_size = 10
train_data = batchify(corpus.train, args.batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)


# # load glove embeddings to tensor
# glove_dict = load_glove_to_dict(args.glove_path, args.emsize)
# glove_tensor = glove_dict_to_tensor(corpus.dictionary.word2idx, glove_dict)


# Build the model
ntokens = len(corpus.dictionary)
# model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.glove, glove_tensor, args.dropout, args.tied)
#改动3
# model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.glove, args.dropout, args.tied)
model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied, args.glove) #glove_tensor
if args.cuda:
    model.cuda()

criterion = nn.CrossEntropyLoss()

# Loop over epochs.
lr = args.lr
best_val_loss = None

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        val_loss = evaluate(val_data)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss)))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 4.0
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)

# Run on test data.
test_loss = evaluate(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)


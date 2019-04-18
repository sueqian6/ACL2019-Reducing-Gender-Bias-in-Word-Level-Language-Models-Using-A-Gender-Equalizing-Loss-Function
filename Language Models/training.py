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
#import jams
import os
import pandas as pd
from sklearn.model_selection import ShuffleSplit
import json

import numpy as np
import random
seed = random.seed(20180330)


parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='./data',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=300,
                    help='size of word embeddings')
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
parser.add_argument('--save', type=str,  default='model.pt',
                    help='path to save the final model')
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

#改动2
def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    ntokens = len(corpus.idx2words)
    hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i)
            output, hidden = model(data, hidden)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
            hidden = repackage_hidden(hidden)
    return total_loss / (len(data_source) - 1)


def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.idx2words)
    hidden = model.init_hidden(args.batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        model.zero_grad()
        output, hidden = model(data, hidden)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()

        

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        ## this is deprecated, no need any more, just use torch.nn.utils.clip_grad_norm
        for p in model.parameters():
            if p.grad is not None:
                p.data.add_(-lr, p.grad.data)
        
        total_loss += loss.data

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss.item() / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()



#load vocab            
vocab = preprocess.read_vocab(os.path.join(args.data,'VOCAB.txt'))


#create json file with indexed filename for following separation
# inds = jams.util.find_with_extension(args.data, 'bin')
# inds = [os.path.join(args.data, x) for x in os.listdir(args.data+'/outputdata') if x.endswith('bin')]
inds = [os.path.join(args.data, x) for x in os.listdir(args.data+'/outputdata') if x.endswith('bin')]


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
idx_train.to_json('idx_train.json')
idx_val.to_json('idx_val.json')
idx_test.to_json('idx_test.json')


# Load pretrained Embeddings, common token of vocab and gn_glove will be loaded, only appear in vocab will be initialized
gn_glove_dir = './gn_glove/1b-vectors300-0.8-0.8.txt' #142527 tokens, last one is '<unk>'
# ntokens = sum(1 for line in open(gn_glove_dir)) + 1
vocab.append('<eos>')
ntokens = len(vocab)
#

with open(gn_glove_dir,'r+', encoding="utf-8") as f: 
    gn_glove_vecs = np.zeros((ntokens, 300))
    words2idx_emb = {}
    idx2words_emb = []
    # ordered_words = []
    for i, line in enumerate(f):
        try:
            s = line.split() 
            gn_glove_vecs[i, :] = np.asarray(s[1:])
            words2idx_emb[s[0]] = i
            idx2words_emb.append(s[0])
            # ordered_words.append(s[0])
        except:
            continue

    words2idx_emb['<eos>'] = i+1
    idx2words_emb.append('<eos>')

#creat new word embeding, word embedding both  in the gn_glove and vocab keeps, only in vocab is initialized
nw = np.zeros((ntokens, args.emsize), dtype=np.float32)
for i,w in enumerate(vocab):#change add start=1
    try:
        r = words2idx_emb[w]
        nw[i] = gn_glove_vecs[r]  
    except:
        continue
words2idx = {item : index for index, item in enumerate(vocab)}
# Load data
corpus = data_v3.Corpus('./data/outputdata', vocab, words2idx, idx_train, idx_val, idx_test) #改动2 
eval_batch_size = 10
train_data = batchify(corpus.train, args.batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

# nw = torch.from_numpy(nw)

# Build the model
model = model.RNNModel(args.model, nw, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied) #ntokens, args.emsize,
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
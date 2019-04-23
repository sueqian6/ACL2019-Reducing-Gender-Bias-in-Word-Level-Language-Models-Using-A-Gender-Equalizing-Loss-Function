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
# seed = random.seed(20180330)

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='./data/preprocessed/data',
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
parser.add_argument('--lamda', type=float, default=1,
                    help='loss parameter')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=20180330,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str,  default='./savedmodel/model.pt',
                    help='path to save the final model')
parser.add_argument('--vocab', type=str, default='./data/preprocessed/VOCAB.txt',
                    help=('preprocessed vocaburary'))
parser.add_argument('--glove', action='store_true',
                    help='use glove')
parser.add_argument('--glove_path', type=str, default='./gn_glove/1b-vectors300-0.8-0.8.txt',
                    help='using glove word embedding')
parser.add_argument('--anneal', type=float, default=4,
                    help='anneal rate of learning rate')
parser.add_argument('--alpha', type=float, default=2,
                    help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
args = parser.parse_args()

print(args)
# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    # else:
    #     torch.cuda.manual_seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()

    return data.to(device)

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


#改动2
def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i)
            output, hidden = model(data, hidden)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output, targets, lamda, f_onehot, m_onehot)
            hidden = repackage_hidden(hidden)
    return total_loss / (len(data_source) - 1)


def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0
    start_time = time.time()
    hidden = model.init_hidden(args.batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        model.zero_grad()
        output, hidden = model(data, hidden)
        #loss = criterion(output.view(-1, ntokens), targets)
        loss = criterion(output, targets, lamda, f_onehot, m_onehot)

        # Activiation Regularization
        if args.alpha:
            loss = loss + sum(args.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])

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

########################


### LOSS CODE ###

# load doc into memory
def load_doc(filename):
    file = open(filename, 'r', encoding='utf-8-sig')
    text = file.read()
    file.close()
    return text

femaleFile = 'gender_words/female_word_file.txt'
maleFile = 'gender_words/male_word_file.txt'

def getGenderIdx(femaleFile, maleFile, word2idx):
    female_word_list = load_doc(femaleFile).split('\n')
    male_word_list = load_doc(maleFile).split('\n')
    pairs = [ (word2idx[f],word2idx[m]) for f,m in zip(female_word_list,male_word_list)  if f in word2idx and m in word2idx]
    femaleIdx = [ f for f,m in pairs]
    maleIdx = [ m for f,m in pairs]
    return femaleIdx,maleIdx

class Custom_Loss(torch.nn.Module):
    
    def __init__(self):
        super(Custom_Loss,self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, output, targets, lamda, femaleIdx, maleIdx):
        cross_ent_loss = self.criterion(output.view(-1, ntokens), targets)
        bias_loss = self.logRatioLossFast(output, femaleIdx, maleIdx) * lamda
        loss = cross_ent_loss + bias_loss     
        return loss
    
    def customLoss_LogRatioGenderPairs(self,output, femaleIdx, maleIdx):
        flato = output.view(-1, ntokens)
        logratio_sum = 0
        for i in range(flato.size()[0]):
            o = flato[i]
            logratio = [abs(torch.log( (torch.exp(o[f]) + 0.00001) / (torch.exp(o[m])+ 0.00001) )) for f,m in zip(femaleIdx, maleIdx)]
            logratio_sum += sum(logratio)    
        return logratio_sum / flato.size()[0]
    
    def logRatioLossFast(self, output, f_onehot, m_onehot):
        flato = output.view(-1, ntokens)
        m1 = torch.matmul(f_onehot,flato.t())
        m2 = torch.matmul(m_onehot,flato.t())
        b_loss = torch.mean(torch.abs(torch.log((torch.exp(m1) + 0.00001) / (torch.exp(m2) + 0.00001))))
        return b_loss


################


#######################

#load vocab            
vocab = preprocess.read_vocab(os.path.join(args.vocab))


#create json file with indexed filename for following separation
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
                               random_state=args.seed)
bigtrains, tests = next(splitter_tt.split(idx_train_['id'].keys()))

idx_bigtrain = idx_train_.iloc[bigtrains]
idx_test = idx_train_.iloc[tests]


#split train, val sets
splitter_tv = ShuffleSplit(n_splits=1, test_size=0.2,
                               random_state=args.seed)

trains, vals = next(splitter_tv.split(idx_bigtrain['id'].keys()))

idx_train = idx_bigtrain.iloc[trains]
idx_val = idx_bigtrain.iloc[vals]


#save idx_train, idx_val, idx_test for later use
idx_train.to_json('idx_train.json')
idx_val.to_json('idx_val.json')
idx_test.to_json('idx_test.json')


# Load pretrained Embeddings, common token of vocab and gn_glove will be loaded, only appear in vocab will be initialized
#142527 tokens, last one is '<unk>'
# ntokens = sum(1 for line in open(gn_glove_dir)) + 1
vocab.append('<eos>')
ntokens = len(vocab)
#

with open(args.glove_path,'r+', encoding="utf-8") as f: 
    gn_glove_vecs = np.zeros((142527, 300)) #april_1
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
        test_i += 1 
    except:
        nw[i] = np.random.normal(scale=0.6, size=(args.emsize, ))

words2idx = {item : index for index, item in enumerate(vocab)}
# Load data
corpus = data_v3.Corpus(args.data, vocab, words2idx, idx_train, idx_val, idx_test) #改动2 
eval_batch_size = 10
train_data = batchify(corpus.train, args.batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

# nw = torch.from_numpy(nw)

######################################

femaleIdx,maleIdx = getGenderIdx(femaleFile, maleFile, corpus.words2idx)


f_onehot = np.zeros((len(femaleIdx), ntokens))
f_onehot[np.arange(len(femaleIdx)), femaleIdx] = 1
f_onehot = torch.tensor(f_onehot, dtype = torch.float).to(device)

m_onehot = np.zeros((len(femaleIdx), ntokens))
m_onehot[np.arange(len(femaleIdx)), maleIdx] = 1
m_onehot = torch.tensor(m_onehot, dtype = torch.float).to(device)



###########################################


# Build the model
model = model.RNNModel(args.model, nw, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied).to(device)

#criterion = nn.CrossEntropyLoss()
criterion = Custom_Loss().to(device)

# Loop over epochs.
lr = args.lr
best_val_loss = None
lamda = args.lamda

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
            if lr<1e-1 and lr>=1e-2:
                lr /= (args.anneal/2)
            elif lr<1e-2 and lr >=1e-3:
                lr /= (args.anneal/3)
            # elif lr<1e-4 and lr >=1e-5:
            #     lr /= (args.anneal/3.5)
            elif lr<1e-3:
                lr /= lr*0.99
            else:
                lr /= args.anneal

            print('new learning rate is {}'.format(lr))
            print('val_loss is {}'.format(val_loss))
            print('best_val_loss is {}'.format(best_val_loss))
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

#!/usr/bin/env python

import argparse
import os

limit = {
        'maxq' : 20,
        'minq' : 0,
        'maxa' : 20,
        'mina' : 3
        }

UNK = '<unk>'
VOCAB_SIZE = 10000

EN_WHITELIST = '0123456789abcdefghijklmnopqrstuvwxyz ' # space is included in whitelist

import random
import sys

import nltk
import itertools
from collections import defaultdict

import numpy as np

import pickle


def ddefault():
    return 1

'''
 read lines from file
     return [list of lines]

'''
def read_lines(filename):
    return open(filename).read().split('\n')[:-1]


'''
 split sentences in one line
  into multiple lines
    return [list of lines]

'''
def split_line(line):
    return line.split('.')


'''
 remove anything that isn't in the vocabulary
    return str(pure ta/en)

'''
def filter_line(line, whitelist):
    return ''.join([ ch for ch in line if ch in whitelist ])


'''
 read list of words, create index to word,
  word to index dictionaries
    return tuple( vocab->(word, count), idx2w, w2idx )

'''
def index_(tokenized_sentences, vocab_size):
    # get frequency distribution
    freq_dist = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    # get vocabulary of 'vocab_size' most used words
    vocab = freq_dist.most_common(vocab_size)
    # index2word
    index2word = ['<pad>'] + [UNK] + [ x[0] for x in vocab ]
    # word2index
    word2index = dict([(w,i) for i,w in enumerate(index2word)] )
    return index2word, word2index, freq_dist


'''
 filter too long and too short sequences
    return tuple( filtered_ta, filtered_en )

'''
def filter_data(sequences, crop):
    filtered_q, filtered_a = [], []
    raw_data_len = len(sequences)//2

    for i in range(0, len(sequences), 2):
        qlen, alen = len(sequences[i].split(' ')), len(sequences[i+1].split(' '))
        if crop:
            if qlen > limit['maxq']:
                cropped_q = sequences[i].split(' ')[:limit['maxq']]
                sequences[i] = (' ').join(cropped_q)
            if alen > limit['maxa']:
                cropped_a = sequences[i+1].split(' ')[:limit['maxa']]
                sequences[i+1] = (' ').join(cropped_a)
            filtered_q.append(sequences[i])
            filtered_a.append(sequences[i+1])
        else:
            if qlen >= limit['minq'] and qlen <= limit['maxq']:
                if alen >= limit['mina'] and alen <= limit['maxa']:
                    filtered_q.append(sequences[i])
                    filtered_a.append(sequences[i+1])

    # print the fraction of the original data, filtered
    filt_data_len = len(filtered_q)
    filtered = int((raw_data_len - filt_data_len)*100/raw_data_len)
    print(str(filtered) + '% filtered from original data')

    return filtered_q, filtered_a





'''
 create the final dataset :
  - convert list of items to arrays of indices
  - add zero padding
      return ( [array_en([indices]), array_ta([indices]) )

'''
def zero_pad(qtokenized, atokenized, w2idx):
    # num of rows
    data_len = len(qtokenized)

    # numpy arrays to store indices
    idx_q = np.zeros([data_len, limit['maxq']], dtype=np.int32)
    idx_a = np.zeros([data_len, limit['maxa']], dtype=np.int32)

    for i in range(data_len):
        q_indices = pad_seq(qtokenized[i], w2idx, limit['maxq'])
        a_indices = pad_seq(atokenized[i], w2idx, limit['maxa'])

        #print(len(idx_q[i]), len(q_indices))
        #print(len(idx_a[i]), len(a_indices))
        idx_q[i] = np.array(q_indices)
        idx_a[i] = np.array(a_indices)

    return idx_q, idx_a


'''
 replace words with indices in a sequence
  replace with unknown if word not in lookup
    return [list of indices]

'''
def pad_seq(seq, lookup, maxlen):
    indices = []
    for word in seq:
        if word in lookup:
            indices.append(lookup[word])
        else:
            indices.append(lookup[UNK])
    return indices + [0]*(maxlen - len(seq))

def filter_too_many_unks(idx_q, idx_a):
    unk_id = 1

    n_unks_per_a = np.sum(idx_a == unk_id, axis=1)
    mask = (n_unks_per_a == 0)
    idx_q = idx_q[mask]
    idx_a = idx_a[mask]

    n_unks_per_q = np.sum(idx_q == unk_id, axis=1)
    mask = (n_unks_per_q <= 2)
    idx_q = idx_q[mask]
    idx_a = idx_a[mask]

    return idx_q, idx_a


def process_data(fname, name, metadata_path, filter_whitelist, filter_unks, crop):

    print('\n>> Read lines from file')
    lines = read_lines(filename=fname)

    # change to lower case (just for en)
    lines = [ line.lower() for line in lines ]

    print('\n:: Sample from read(p) lines')
    print(lines[121:125])

    if filter_whitelist:
        # filter out unnecessary charactersg
        print('\n>> Filter lines')
        lines = [filter_line(line, EN_WHITELIST) for line in lines]
        print(lines[121:125])

    # filter out too long or too short sequences
    print('\n>> 2nd layer of filtering')
    qlines, alines = filter_data(lines, crop)
    print('\nq : {0} ; a : {1}'.format(qlines[0], alines[0]))
    print('\nq : {0} ; a : {1}'.format(qlines[1], alines[1]))


    # convert list of [lines of text] into list of [list of words ]
    print('\n>> Segment lines into words')
    qtokenized = [ wordlist.split(' ') for wordlist in qlines ]
    atokenized = [ wordlist.split(' ') for wordlist in alines ]
    print('\n:: Sample from segmented list of words')
    print('\nq : {0} ; a : {1}'.format(qtokenized[0], atokenized[0]))
    print('\nq : {0} ; a : {1}'.format(qtokenized[1], atokenized[1]))

    if metadata_path is None:
        print("Generating vocabulary...")
        idx2w, w2idx, _ = \
            index_(qtokenized + atokenized, vocab_size=VOCAB_SIZE)
    else:
        print("Loading saved vocabulary from %s..." % metadata_path)
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        idx2w = metadata['idx2w']
        w2idx = metadata['w2idx']
    print("ID to word dictionary: %d items" % len(idx2w))
    print("Word to ID dictionary: %d items" % len(w2idx))

    print('\n >> Zero Padding')
    idx_q, idx_a = zero_pad(qtokenized, atokenized, w2idx)

    if filter_unks:
        idx_q, idx_a = filter_too_many_unks(idx_q, idx_a)

    print('\n >> Save numpy arrays to disk')
    # save them
    np.save('%s_q.npy' % name, idx_q)
    np.save('%s_a.npy' % name, idx_a)

    metadata = {
        'w2idx': w2idx,
        'idx2w': idx2w,
        'limit': limit
    }
    if metadata_path is None:
        # write to disk : data control dictionaries
        with open('metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)

    return idx_q, idx_a, metadata


def load_data(PATH=''):
    # read data control dictionaries
    with open(os.path.join(PATH, 'metadata.pkl'), 'rb') as f:
        metadata = pickle.load(f)
    # read numpy arrays
    idx_q = np.load(PATH + 'idx_q.npy')
    idx_a = np.load(PATH + 'idx_a.npy')
    return metadata, idx_q, idx_a

if __name__ == '__main__':
    desc = """
./data.py Training_Shuffled_Dataset_Tuples.txt training
./data.py --metadata metadata.pkl Validation_Shuffled_Dataset_Tuples.txt validation
"""
    parser = argparse.ArgumentParser(
        description=desc, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('dataset_txt')
    parser.add_argument('dataset_name')
    parser.add_argument('--metadata')
    parser.add_argument('--filter_whitelist', action='store_true')
    parser.add_argument('--filter_unks', action='store_true')
    parser.add_argument('--crop', action='store_true')
    args = parser.parse_args()
    process_data(args.dataset_txt, args.dataset_name, args.metadata,
                 args.filter_whitelist, args.filter_unks, args.crop)

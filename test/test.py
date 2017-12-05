import tensorflow as tf
import numpy as np
import pickle
import sys
import argparse

sys.path.append('..')
import utils
import seq2seq_wrapper

parser = argparse.ArgumentParser()
parser.add_argument('--text', action='store_true')
args = parser.parse_args()

read_ckpt_dir = 'b8d5773'
emb_dim = 1024
batch_size = 16

testX = np.load('test_q.npy')
testY = np.load('test_a.npy')

with open('metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

xseq_len = testX.shape[-1]
yseq_len = testY.shape[-1]
xvocab_size = len(metadata['idx2w'])
yvocab_size = xvocab_size

model = seq2seq_wrapper.Seq2Seq(xseq_len=xseq_len,
                               yseq_len=yseq_len,
                               xvocab_size=xvocab_size,
                               yvocab_size=yvocab_size,
                               ckpt_path='ckpt/',
                               emb_dim=emb_dim,
                               num_layers=3,
                               mode=seq2seq_wrapper.ATTENTION_MODE
                               )
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

checkpoint_file = tf.train.latest_checkpoint(read_ckpt_dir)
saver.restore(sess, checkpoint_file)

start = 0
end = batch_size
while True:
    x = testX[start:end]
    y = testX[start:end]

    if args.text:
        logits = model.predict(sess, x.T)
        for reply in logits:
            print(' '.join([metadata['idx2w'][i] for i in reply]))
    else:
        logits = model.predict(sess, x.T, Y=y.T, argmax=False)
        word_probabilities = utils.softmax(logits, axis=-1)
        sentence_perplexities = utils.perplexity(word_probabilities, reference=y)

        for i in range(0, len(sentence_perplexities), 2):
            print("%.3f %.3f" % (sentence_perplexities[i], sentence_perplexities[i+1]))

    if end == len(testX):
        break

    start += batch_size
    end += batch_size
    if end > len(testX):
        end = len(testX)

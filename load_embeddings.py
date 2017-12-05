from __future__ import print_function
from gensim import models
import numpy as np
import tensorflow as tf


def load_embedding(session, vocab, embs, path, dim_embedding, vocab_length,
                   model=None):
    '''
      session        Tensorflow session object
      vocab          A dictionary mapping token strings to vocabulary IDs
      emb            Embedding tensor of shape vocabulary_size x dim_embedding
      path           Path to embedding file
      dim_embedding  Dimensionality of the external embedding.
    '''
    if model is None:
        print("Loading embeddings from '%s'..." % path, end='')
        model = models.KeyedVectors.load_word2vec_format(path, binary=True)
        print("done!")
    external_embedding = np.zeros(shape=(vocab_length, dim_embedding))
    matches = 0
    for tok, idx in vocab.items():
        if tok in model.vocab:
            external_embedding[idx] = model[tok]
            matches += 1
        else:
            print("Warning: token '%s' not found in embeddings" % tok)
            external_embedding[idx] = \
                np.random.normal(loc=0.0, scale=0.01, size=dim_embedding)

    print("%d words out of %d could be loaded" % (matches, vocab_length))

    pretrained_embeddings = tf.placeholder(tf.float32, [None, None])
    ops = []
    for emb in embs:
        ops.append(emb.assign(pretrained_embeddings))
    session.run(ops, {pretrained_embeddings: external_embedding})

    return model

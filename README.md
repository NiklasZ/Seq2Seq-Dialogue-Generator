# Seq2Seq Dialogue Generator

The following is a [seq2seq](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf) trainer that was designed to create a chat bot. It was trained using a movie dataset and will respond to some questions rather dramatically. Unfortunately, the movie data set cannot be included for intellectual property reasons, but the rest of the model is here.

## Requirements

* Jupyter Notebook (`pip install jupyter`)
* Tensorflow 1.0 (`pip install tensorflow-gpu==1.0`)
* NLTK (`pip install nltk`)

## Installation
The files in the repository function as follows:
* `model.ipynb`, the main notebook for the model
* The `perplexity` and `max_branching_score` score functions in `utils.py`
* In `legacy_seq2seq.py`, in `loop_function`, changes to `prev_symbol`
calculation to enable diversification of answers based on "A
Diversity-Promoting Objective Function for Neural Conversation Models" (Li et
al. 2015)
* Adaptations to `seq2seq_wrapper.py`:
  * In `__graph__`, use Tensorflow's `embedding_attention_seq2seq` with an extra projection
    layer instead of `embedding_rnn_seq2seq`
  * `load_embeddings`: a function to load pretrained word2vec embeddings using
   `load_wrapper.py`
  * Tweaks to `predict` to enable perplexity calculation


## Acknowledgments

I reuse code from several sources:
* The basic sequence-to-sequence model (`seq2seq_wrapper.py`, `data_utils.py`,
  `data/data.py`) is adapted from Suriyadeepan's
  [`practical_seq2seq`](https://github.com/suriyadeepan/practical_seq2seq).
* `legacy_seq2seq.py` is adapted from contrib.legacy_seq2seq from Tensorflow 1.0
  (<https://github.com/tensorflow/tensorflow/blob/r1.0/tensorflow/contrib/legacy_seq2seq/python/ops/seq2seq.py>)
* The softmax code in `utils.py` comes from Nolan Conaway
  (<https://nolanbconaway.github.io/blog/2017/softmax-numpy>)

## Test mode

To test the model run:

```
cd test
./run-test.sh some-triples.txt
```

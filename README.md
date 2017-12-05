# NLU 2017 Task 2

## Requirements

* Tensorflow 1.0 (`pip install tensorflow-gpu==1.0`)
* NLTK (`pip install nltk`)

## Code

We reuse code from several sources:
* The basic sequence-to-sequence model (`seq2seq_wrapper.py`, `data_utils.py`,
  `data/data.py`) is adapted from Suriyadeepan's
  [`practical_seq2seq`](https://github.com/suriyadeepan/practical_seq2seq).
* `legacy_seq2seq.py` is adapted from contrib.legacy_seq2seq from Tensorflow 1.0
  (<https://github.com/tensorflow/tensorflow/blob/r1.0/tensorflow/contrib/legacy_seq2seq/python/ops/seq2seq.py>)
* The softmax code in `utils.py` comes from Nolan Conaway
  (<https://nolanbconaway.github.io/blog/2017/softmax-numpy>)
* `load_embeddings.py` was supplied to us by the TAs for the course, Florian
  Schmidt and Jason Lee.

Our work comprises the following:
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

## Test mode

To run the model in 'test mode':

```
cd test
./run-test.sh some-triples.txt
```

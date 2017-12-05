import numpy as np

# https://nolanbconaway.github.io/blog/2017/softmax-numpy
def softmax(X, theta = 1.0, axis = None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis = axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p


def perplexity(sentences, reference, debug=False, metadata=None):
    """
    Calculates sentence-level perplexity values.
    
    sentences is assumed to be batch size x sequence length x vocabulary length,
    where the last dimension represents softmax probability values.
    
    reference is assumed to be batch size x sequence length,
    where the last dimension is the id of the ground truth word
    """
    pad_id = 0
    perplexities = []
    n_sentences = sentences.shape[0]
    n_words = sentences.shape[1]
    for i in range(n_sentences):
        word_probabilities = np.zeros(reference.shape[-1])
        for j in range(n_words):
            word_probabilities[j] = sentences[i,j,reference[i,j]]
        if np.all(reference == pad_id):
            perp = None
        else:
            logp = np.log2(word_probabilities)[reference[i,:] != pad_id]
            n = len(logp)
            perp = 2 ** (-1 * 1/n * np.sum(logp))
        perplexities.append(perp)
    return perplexities


def perplexity_test():
    w1_p = [0.1, 0.9]
    w2_p = [0.2, 0.8]
    # w3: element 0 has highest probability
    # => w3 is a pad
    w3_p = [0.6, 0.4]
    sentence = [w1_p, w2_p, w3_p]
    batch = [sentence] * 2
    actual = perplexity(batch)
    expected = [2 ** (-1 * 1/2.0 * np.sum(np.log2([0.9, 0.8])))] * 2
    assert np.array_equal(actual, expected)

def max_branching_score(answers):
    # Input: A list of sentences, each sentence is a list of word ids
    # branches contains a tuple of number of occurrences and a set of child nodes
    counts = {}
    for answer in answers:
        for i in range(1, len(answer)+1):
            key = tuple(answer[:i])
            if not key in counts.keys():
                counts[key] = 1
            else:
                counts[key] = counts[key]+1
    sentence_scores = []
    for answer in answers:
        prefix_scores = []
        for i in range(1, len(answer)+1):
            prefix = answer[:i]
            key = tuple(prefix)
            prefix_scores.append(len(prefix)**2*counts[key])
        sentence_scores.append(float(max(prefix_scores))/len(answer)/len(answers))
    return sentence_scores
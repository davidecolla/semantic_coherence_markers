from nltk.lm.preprocessing import padded_everygram_pipeline

import numpy as np
import nltk
from collections import Counter
from nltk.lm import Vocabulary, KneserNeyInterpolated
from nltk import word_tokenize, sent_tokenize
import math


def train_ngrams(text, order, replace_with_unk=True):
    # # ------------------- Build n-grams --------------- #
    tokenized_text = [list(map(str.lower, word_tokenize(sent))) for sent in sent_tokenize(text)]

    # -------------------------------------------------
    # Replace words occurring <= 1 times with UNK token
    # -------------------------------------------------
    if replace_with_unk:
        # Compute word frequencies
        text_words = []

        for t in tokenized_text:
            text_words.extend(t)

        word_freq = Counter(text_words)
        # Replace word with freq < T with UNK
        for i, sentence in enumerate(tokenized_text):
            for j, word in enumerate(tokenized_text[i]):
                if word_freq[word] <= 1:
                    tokenized_text[i][j] = "UNK"
    # -------------------------------------------------

    train_data, padded_vocab = padded_everygram_pipeline(order, tokenized_text)
    # discount = 0.1
    model = KneserNeyInterpolated(order)
    model.fit(train_data, padded_vocab)

    return model


def compute_perplexity(text, order, model, replace_with_unk=True):

    # ------------------------------------------------- #
    #                   No chunking                     #
    # ------------------------------------------------- #
    tokenized_text = [list(map(str.lower, word_tokenize(sent))) for sent in sent_tokenize(text)]


    # -------------------------------------------------
    # Replace words occurring <= 1 times with UNK token
    # -------------------------------------------------
    if replace_with_unk:
        # Compute word frequencies
        text_words = []

        for t in tokenized_text:
            text_words.extend(t)

        # Replace word with freq < T with UNK
        for i, sentence in enumerate(tokenized_text):
            for j, word in enumerate(tokenized_text[i]):
                if model.counts[word] <= 1:
                    tokenized_text[i][j] = "UNK"
    # -------------------------------------------------


    test_data = [nltk.ngrams(t, order, pad_right=True, pad_left=True, left_pad_symbol="<s>", right_pad_symbol="</s>") for t in tokenized_text]
    ppl_scores_no_chunks = []
    for sent_text in test_data:
        sentence_ppl_score = perplexity_base_e(sent_text, model)
        ppl_scores_no_chunks.append(sentence_ppl_score)

    avg_ppl = np.mean(ppl_scores_no_chunks)
    dev_std = np.std(ppl_scores_no_chunks)

    return avg_ppl, dev_std





# ------------------------------------------------------------------------------------------------- #

def log_base_e(score):
    """Convenience function for computing logarithms with base e."""
    NEG_INF = float("-inf")
    POS_INF = float("inf")
    if score == 0.0:
        return NEG_INF
    return math.log(score)

def _mean(items):
    """Return average (aka mean) for sequence of items."""
    return sum(items) / len(items)


def logscore(word, model, context=None):
    """Evaluate the log score of this word in this context.

    The arguments are the same as for `score` and `unmasked_score`.

    """
    return log_base_e(model.unmasked_score(word, context))



def entropy(text_ngrams, model):
    """Calculate cross-entropy of model for given evaluation text.

    :param Iterable(tuple(str)) text_ngrams: A sequence of ngram tuples.
    :rtype: float

    """
    return -1 * _mean(
        [logscore(ngram[-1], model, ngram[:-1]) for ngram in text_ngrams]
    )


def perplexity_base_e(text_ngrams, model):
    """Calculates the perplexity of the given text.

    This is simply 2 ** cross-entropy for the text, so the arguments are the same.

    """
    return math.exp(entropy(text_ngrams, model))
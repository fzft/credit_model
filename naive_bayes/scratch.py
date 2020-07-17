import numpy as np
import pandas as pd
from typing import Callable

from sklearn.feature_extraction.text import CountVectorizer


def make_spam_dataset(show_X=True) -> (pd.DataFrame, np.ndarray, Callable):
    vocab = [
        'secret', 'offer', 'low', 'price', 'valued', 'customer', 'today',
        'dollar', 'million', 'sports', 'is', 'for', 'play', 'healthy', 'pizza'
    ]

    spam = [
        'million dollar offer',
        'secret offer today',
        'secret is secret'
    ]

    not_spam = [
        'low price for valued customer',
        'play secret sports today',
        'sports is healthy',
        'low price pizza'
    ]

    all_messages = spam + not_spam

    vectorizer = CountVectorizer(vocabulary=vocab)
    word_counts = vectorizer.fit_transform(all_messages).toarray()
    df = pd.DataFrame(word_counts, columns=vocab)
    is_spam = [1] * len(spam) + [0] * len(not_spam)
    msg_tx_func = lambda x: vectorizer.transform(x).toarray()

    if show_X:
        print(df)

    return df.to_numpy(), np.array(is_spam), msg_tx_func


# X, y, tx_func = make_spam_dataset()


class NaiveBayes(object):
    """ DIY binary Naive Bayes classifier based on categorical data """

    def __init__(self, alpha=1.0):
        """ """
        self.prior = None
        self.word_counts = None
        self.lk_word = None
        self.alpha = alpha
        self.is_fitted_ = False
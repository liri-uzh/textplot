

import os
import re
import matplotlib.pyplot as plt
import textplot.utils as utils
import numpy as np
import pkgutil

from nltk.stem import PorterStemmer
from sklearn.neighbors import KernelDensity
from collections import OrderedDict, Counter
from scipy.spatial import distance
from scipy import ndimage
from functools import lru_cache


class Text:


    @classmethod
    def from_file(cls, path):

        """
        Create a text from a file.

        Args:
            path (str): The file path.
        """

        # with open(path, 'r', errors='replace') as f:
        #     return cls(f.read())

        pass


    def __init__(self, token_dicts, stopwords=None, pos=['NOUN', 'PROPN'], min_char=2, lang='en'):

        """
        Store the raw text, tokenize.

        Args:
            text (str): The raw text string.
            lower: apply lower-casing
            stopwords (str or list): A custom stopwords path or list
            min_char (int): exclude words with less characters
        """

        self.token_dicts = token_dicts
        #self.lower = lower
        self.load_stopwords(stopwords)
        self.min_char = min_char

        if 'pos' in self.token_dicts[0]:
            self.pos = pos  # filter by given part of speech tags
        else:
            self.pos = None

        #self.tokenizer = utils.Tokenizer(lang=lang, lower=lower)
        self.tokenize()


    def load_stopwords(self, stopwords):

        """
        Load a set of stopwords.

        Args:
            stopwords (str or list): The stopwords file path or list of stopwords.
        """

        if hasattr(stopwords, '__iter__'):
            # check if stopwords is an iterable
            self.stopwords = stopwords

        elif os.path.exists(stopwords):
            with open(stopwords, 'r') as f:
                self.stopwords = set(f.read().splitlines())
        else:
            self.stopwords = set(
                pkgutil
                .get_data('textplot', 'data/stopwords.txt')
                .decode('utf8')
                .splitlines()
            )


    def tokenize(self):

        """
        Tokenize the text.
        """

        self.tokens = []
        self.terms = OrderedDict()

        # Generate tokens.
        for token_dict in self.token_dicts:

            # filter if in stopword list.
            if token_dict['original'].lower() in self.stopwords:
                self.tokens.append(None)
                continue

            # filter by token length  
            elif len(token_dict['original']) < self.min_char:
                self.tokens.append(None)
                continue

            # filter by part of speech 
            elif self.pos is not None and token_dict['pos'] not in self.pos:
                self.tokens.append(None)
                continue


            else:
                # Token:
                self.tokens.append(token_dict)

                # Term:
                offsets = self.terms.setdefault(token_dict['normalized'], [])
                offsets.append(token_dict['offset'])


    def term_counts(self):

        """
        Returns:
            OrderedDict: An ordered dictionary of term counts.
        """

        counts = OrderedDict()
        for term in self.terms:
            counts[term] = len(self.terms[term])

        return utils.sort_dict(counts)


    def term_count_buckets(self):

        """
        Returns:
            dict: A dictionary that maps occurrence counts to the terms that
            appear that many times in the text.
        """

        buckets = {}
        for term, count in self.term_counts().items():
            if count in buckets: buckets[count].append(term)
            else: buckets[count] = [term]

        return buckets


    def most_frequent_terms(self, depth):

        """
        Get the X most frequent terms in the text, and then probe down to get
        any other terms that have the same count as the last term.

        Args:
            depth (int): The number of terms.

        Returns:
            set: The set of frequent terms.
        """

        counts = self.term_counts()

        # Get the top X terms and the instance count of the last word.
        top_terms = set(list(counts.keys())[:depth])
        end_count = list(counts.values())[:depth][-1]

        # Merge in all other words with that appear that number of times, so
        # that we don't truncate the last bucket - eg, half of the words that
        # appear 5 times, but not the other half.

        bucket = self.term_count_buckets()[end_count]
        return top_terms.union(set(bucket))


    def unnormalize(self, term):

        """
        Given a normalized term, get the most common unnormalized variant.

        Args:
            term (str): A normalized term.

        Returns:
            str: The most common original token.
        """

        originals = []
        for i in self.terms[term]:
            i_int = int(i)
            originals.append(self.tokens[i_int]['original'])

        mode = Counter(originals).most_common(1)
        return mode[0][0]


    @lru_cache(maxsize=None)
    def kde(self, term, bandwidth=2000, samples=1000, kernel='gaussian'):

        """
        Estimate the kernel density of the instances of term in the text.

        Args:
            term (str): A normalized term.
            bandwidth (int): The kernel bandwidth.
            samples (int): The number of evenly-spaced sample points.
            kernel (str): The kernel function.

        Returns:
            np.array: The density estimate.
        """

        # Get the offsets of the term instances.
        terms = np.array(self.terms[term])[:, np.newaxis]

        # Fit the density estimator on the terms.
        kde = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(terms)

        # Score an evely-spaced array of samples.
        x_axis = np.linspace(0, len(self.tokens), samples)[:, np.newaxis]
        scores = kde.score_samples(x_axis)

        # Scale the scores to integrate to 1.
        return np.exp(scores) * (len(self.tokens) / samples)


    def score_intersect(self, term1, term2, **kwargs):

        """
        Compute the geometric area of the overlap between the kernel density
        estimates of two terms.

        Args:
            term1 (str)
            term2 (str)

        Returns: float
        """

        t1_kde = self.kde(term1, **kwargs)
        t2_kde = self.kde(term2, **kwargs)

        # Integrate the overlap.
        overlap = np.minimum(t1_kde, t2_kde)
        return np.trapz(overlap)


    def score_cosine(self, term1, term2, **kwargs):

        """
        Compute a weighting score based on the cosine distance between the
        kernel density estimates of two terms.

        Args:
            term1 (str)
            term2 (str)

        Returns: float
        """

        t1_kde = self.kde(term1, **kwargs)
        t2_kde = self.kde(term2, **kwargs)

        return 1-distance.cosine(t1_kde, t2_kde)


    def score_braycurtis(self, term1, term2, **kwargs):

        """
        Compute a weighting score based on the "City Block" distance between
        the kernel density estimates of two terms.

        Args:
            term1 (str)
            term2 (str)

        Returns: float
        """

        t1_kde = self.kde(term1, **kwargs)
        t2_kde = self.kde(term2, **kwargs)

        return 1-distance.braycurtis(t1_kde, t2_kde)


    def plot_term_kdes(self, words, **kwargs):

        """
        Plot kernel density estimates for multiple words.

        Args:
            words (list): A list of unnormalized terms.

        #TODO : change this to work with a custom type of normalization
        """

        stem = PorterStemmer().stem

        for word in words:
            kde = self.kde(stem(word), **kwargs)
            plt.plot(kde)

        plt.show()

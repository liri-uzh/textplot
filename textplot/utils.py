

import re
import numpy as np
import functools
import copy

from collections import OrderedDict
from nltk.stem import PorterStemmer
from itertools import islice

import spacy
from nltk.stem.snowball import SnowballStemmer

class SpacyTokenizer(object):
    def __init__(self, lang='de', lower=False):

        if lang == 'de':
            spacy_model = 'de_core_news_sm'
        elif lang == 'en':
            spacy_model = 'en_core_web_sm'
        else:
            print(f'Selected language not yet supported: {lang}')
            raise NotImplementedError
        print(f'Initializing tokenizer: Spacy tokenizer with model {spacy_model}')
        self.nlp = spacy.load(spacy_model)
        self.lower = lower

        self.global_offset = 0

    def tokenize(self, collection):

        for text in collection:

            for tok in self.nlp(text):

                base = tok.lemma_
                if self.lower is True:
                    full = tok.lower_
                else:
                    full = tok.text

                offset = copy.deepcopy(self.global_offset)
                self.global_offset+=1

                yield { # Emit the token.
                'normalized':      base,
                'original':    full,
                'offset':       offset
            }

class SplitTokenizer(object):
    def __init__(self, sep=' ', lower=False, normalizer=None):
        self.lower = lower
        self.sep = sep
        self.normalizer = normalizer
        self.global_offset = 0

    def tokenize(self, collection):
        for text in collection:
            for tok in text.split(sep):
                if normalizer is not None:
                    base = normalizer.normalize(tok)
                else:
                    base = tok

                offset = copy.deepcopy(self.global_offset)
                self.global_offset+=1

                yield { # Emit the token.
                'normalized':      base,
                'original':    tok,
                'offset':       offset
            }

class ListTokenizer(object):
    pass 




def tokenize(text, normalizer=PorterStemmer().stem):

    """
    Yield tokens.

    Args:
        text (str): The original text.

    Yields:
        dict: The next token.
    """

    tokens = text.split()

    for offset, token in enumerate(tokens):

        yield { # Emit the token.
            'normalized':      normalize(token),
            'original':    token,
            'offset':       offset
        }


def sort_dict(d, desc=True):

    """
    Sort an ordered dictionary by value, descending.

    Args:
        d (OrderedDict): An ordered dictionary.
        desc (bool): If true, sort desc.

    Returns:
        OrderedDict: The sorted dictionary.
    """

    sort = sorted(d.items(), key=lambda x: x[1], reverse=desc)
    return OrderedDict(sort)


def window(seq, n=2):

    """
    Yield a sliding window over an iterable.

    Args:
        seq (iter): The sequence.
        n (int): The window width.

    Yields:
        tuple: The next window.
    """

    it = iter(seq)
    result = tuple(islice(it, n))

    if len(result) == n:
        yield result

    for token in it:
        result = result[1:] + (token,)
        yield result

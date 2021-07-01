

import re
import numpy as np
import functools
import copy

from collections import OrderedDict
from nltk.stem import PorterStemmer
from itertools import islice

import spacy

class Tokenizer(object):
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

                prev_length = len(text)
                base = tok.lemma_
                if self.lower is True:
                    full = tok.lower_
                else:
                    full = tok.text

                offset = copy.deepcopy(self.global_offset)
                self.global_offset+=1

                yield { # Emit the token.
                'stemmed':      tok.lemma_,
                'unstemmed':    full,
                'offset':       offset
            }


def tokenize(text):

    """
    Yield tokens.

    Args:
        text (str): The original text.

    Yields:
        dict: The next token.
    """

    stem = PorterStemmer().stem
    #tokens = re.finditer('[a-z]+', text.lower())

    tokens = text.split()

    for offset, match in enumerate(tokens):

        # Get the raw token.
        #unstemmed = match.group(0)
        unstemmed = match

        yield { # Emit the token.
            'stemmed':      stem(unstemmed),
            'unstemmed':    unstemmed,
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

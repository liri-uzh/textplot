#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Optional
from collections import OrderedDict
from itertools import islice

def load_words_from_file(file_path: str) -> list:
    """
    Load words (e.g. stopwords, connectors, etc.) from a file into a list.
    """
    with open(file_path, 'r') as f:
        return [line.strip() for line in f]

def parse_wordlists(wordlist_file: Optional[str] = None, wordlist: Optional[list] = None) -> list:
    words = set()
    if wordlist_file:
        words.update(load_words_from_file(wordlist_file))
    if wordlist:
        words.update(wordlist)
    return words

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

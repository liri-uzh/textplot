import matplotlib.pyplot as plt
import textplot.utils as utils
from textplot.tokenization import LegacyTokenizer, PhrasalTokenizer
import numpy as np
import logging
import re
import io

from pathlib import Path

from nltk.stem import PorterStemmer
from sklearn.neighbors import KernelDensity
from collections import OrderedDict, Counter
from scipy.spatial import distance
from functools import lru_cache


logger = logging.getLogger(__name__)

# try:
# 	from memory_profiler import profile # Import the profile decorator
# except ImportError:
# 	def profile(func): # Create a dummy decorator if memory_profiler is not available
# 		return func
class Text:

    def __init__(self, corpus_like_object, **kwargs):
        """
        Initialize a Text object from various input sources, supporting efficient streaming.

        Args:
            corpus_like_object: Raw text, file path, directory path, file-like object, or list of strings
        """
        self.input = corpus_like_object
        self.input_type = self._detect_input_type(corpus_like_object, **kwargs)
        self.kwargs = kwargs
        self.tokens = None
        self.terms = None
        self.chunk_size = self.kwargs.pop("chunk_size", 1000)
        self.chunk_by = self.kwargs.pop("chunk_by", "line")
        logger.debug(f"Initializing Text object with input type: {self.input_type}")
        logger.debug(f"Chunk size: {self.chunk_size}, Chunk by: {self.chunk_by}")
        self.tokenize(**kwargs)


    def _detect_input_type(self, obj, **kwargs):
        if isinstance(obj, str):
            path = Path(obj)
            if path.exists():
                if path.is_file():
                    return 'file'
                elif path.is_dir():
                    return 'directory'
            return 'string'
        elif isinstance(obj, list) and all(isinstance(i, str) for i in obj):
            return 'list'
        elif hasattr(obj, 'read'):
            return 'filelike'
        else:
            raise ValueError("Unsupported input type.")

    def iter_lines(self):
        """
        Yield lines of text from the input source, streaming efficiently.
        """
        if self.input_type == 'file':
            with open(self.input, 'r', encoding='utf-8') as f:
                for line in f:
                    yield line
        elif self.input_type == 'directory':
            file_pattern = self.kwargs.get('file_pattern', '*.txt')
            recursive = self.kwargs.get('recursive', True)
            glob_pattern = '**/' + file_pattern if recursive else file_pattern
            for file_path in Path(self.input).glob(glob_pattern):
                if file_path.is_file():
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            yield line
        elif self.input_type == 'list':
            for text in self.input:
                for line in text.splitlines():
                    yield line
        elif self.input_type == 'filelike':
            self.input.seek(0)
            for line in self.input:
                yield line
        elif self.input_type == 'string':
            for line in io.StringIO(self.input):
                yield line

    # @profile
    def iter_chunks(self):
        """
        Yield text chunks from the input source, streaming efficiently.
        """
        if self.chunk_by == "word":
            chunk = []
            for line in self.iter_lines():
                tokens = re.split(r"\s+", line.strip())
                for token in tokens:
                    if token:
                        chunk.append(token)
                    if len(chunk) >= self.chunk_size:
                        yield " ".join(chunk[:self.chunk_size])
                        chunk = chunk[self.chunk_size:]
            if chunk:
                yield " ".join(chunk)
        elif self.chunk_by == "line":
            chunk = []
            for line in self.iter_lines():
                chunk.append(line.rstrip("\n"))
                if len(chunk) >= self.chunk_size:
                    yield "\n".join(chunk)
                    chunk = []
            if chunk:
                yield "\n".join(chunk)
        else:
            raise ValueError(f"Unknown chunking mode: {self.chunk_by}. Expected 'word' or 'line'.")
    
    def tokenize(self, **kwargs):
        """
        Tokenize the text using the provided tokenizer and streaming chunks.
        """
        if kwargs.get("tokenizer") in ["spacy", "phrasal"]:
            self.tokenizer = PhrasalTokenizer(**kwargs)
            logger.debug("Using PhrasalTokenizer for tokenization.")
        else:
            self.tokenizer = LegacyTokenizer(**kwargs)
            logger.debug("Using LegacyTokenizer for tokenization.")

        self.tokens = []
        self.terms = OrderedDict()
        
        # Initialize chunk iterator factory which will be used by tokenizer
        # this allows for multiple iterations of the chunks 
        # (one for learning phrases and one for tokenization)
        def chunk_iter_factory():
            return self.iter_chunks()
    
        for token in self.tokenizer.tokenize(chunk_iter_factory, **kwargs):
            self.tokens.append(token)
            offsets = self.terms.setdefault(token["stemmed"], [])
            offsets.append(token["offset"])


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
            if count in buckets:
                buckets[count].append(term)
            else:
                buckets[count] = [term]

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

    def unstem(self, term):
        """
        Given a stemmed term, get the most common unstemmed variant.

        Args:
            term (str): A stemmed term.

        Returns:
            str: The unstemmed token.
        """

        originals = []
        for i in self.terms[term]:
            originals.append(self.tokens[i]["unstemmed"])

        mode = Counter(originals).most_common(1)
        return mode[0][0]

    def kde(self, term, bandwidth=2000, samples=1000, kernel="gaussian", **kwargs):
        """
        Wrapper for _kde that handles unhashable kwargs.
        """
        # Convert any unhashable kwargs to hashable versions
        hashable_kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, set):
                hashable_kwargs[k] = frozenset(v)
            else:
                hashable_kwargs[k] = v

        return self._kde(term, bandwidth, samples, kernel, **hashable_kwargs)

    @lru_cache(maxsize=None)
    def _kde(self, term, bandwidth=2000, samples=1000, kernel="gaussian", **kwargs):
        """
        Estimate the kernel density of the instances of term in the text.

        Args:
            term (str): A stemmed term.
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

        return 1 - distance.cosine(t1_kde, t2_kde)

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

        return 1 - distance.braycurtis(t1_kde, t2_kde)

    def plot_term_kdes(self, words, **kwargs):
        """
        Plot kernel density estimates for multiple words.

        Args:
            words (list): A list of unstemmed terms.
        """

        stem = PorterStemmer().stem

        for word in words:
            kde = self.kde(stem(word), **kwargs)
            plt.plot(kde)

        plt.show()

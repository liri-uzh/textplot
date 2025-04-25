import matplotlib.pyplot as plt
import textplot.utils as utils
from textplot.tokenization import LegacyTokenizer, PhrasalTokenizer
import numpy as np

from pathlib import Path
from tqdm import tqdm

from nltk.stem import PorterStemmer
from sklearn.neighbors import KernelDensity
from collections import OrderedDict, Counter
from scipy.spatial import distance
from functools import lru_cache

from textplot.constants import BAR_FORMAT


class Text:
    @staticmethod
    def _read_file(file_path, **kwargs):
        """Helper method to read a single file."""
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    @staticmethod
    def _read_directory(dir_path, file_pattern="*.txt", recursive=True, **kwargs):
        """Helper method to read files from a directory."""
        path = Path(dir_path)
        texts = []
        glob_pattern = "**/" + file_pattern if recursive else file_pattern

        for file_path in tqdm(
            list(path.glob(glob_pattern)),
            desc="Loading files...",
            bar_format=BAR_FORMAT,
        ):
            try:
                texts.append(Text._read_file(file_path))
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

        return texts

    @staticmethod
    def _combine_texts(texts, separator="\n\n", **kwargs):
        """Helper method to combine multiple texts."""
        return separator.join(texts)

    @classmethod
    def from_file(cls, path, **kwargs):
        """Create a text from a file."""
        content = cls._read_file(path)
        return cls(content, **kwargs)

    @classmethod
    def from_directory(
        cls, directory_path, file_pattern="*.txt", recursive=True, **kwargs
    ):
        """Create a text from multiple files in a directory."""
        texts = cls._read_directory(directory_path, file_pattern, recursive)
        return cls.from_texts(texts, **kwargs)

    @classmethod
    def from_texts(cls, texts, separator="\n\n", **kwargs):
        """Create a text from a list of text strings."""
        combined_text = cls._combine_texts(texts, separator)
        return cls(combined_text, **kwargs)

    def __init__(self, corpus_like_object, **kwargs):
        """
        Initialize a Text object from various input sources.

        Args:
            corpus_like_object: Raw text, file path, directory path, file-like object, or list of strings
            stopwords: Path to stopwords file
            tokenizer: Name of tokenizer to use
        """
        self.text = self._process_input(corpus_like_object, **kwargs)
        self.tokens = None
        self.terms = None

        if kwargs.get("tokenizer") in ["spacy", "phrasal"]:
            self.tokenizer = PhrasalTokenizer(**kwargs)
        else:
            self.tokenizer = LegacyTokenizer(**kwargs)

        self.tokenize(**kwargs)

    def _process_input(self, corpus_like_object, **kwargs):
        """
        Process various input types to extract text content.

        Args:
            corpus_like_object: Raw text, file path, directory path, file-like object, or list of strings

        Returns:
            str: Extracted text content
        """

        # String input (raw text or path)
        if isinstance(corpus_like_object, str):
            try:
                path = Path(corpus_like_object)
                if path.exists():
                    if path.is_file():
                        return self.__class__._read_file(path)
                    elif path.is_dir():
                        texts = self.__class__._read_directory(path, **kwargs)
                        return self.__class__._combine_texts(texts)
                # Not a valid path, treat as raw text
                return corpus_like_object
            except (OSError, TypeError):
                # Not a valid path, treat as raw text
                return corpus_like_object

        # List of text strings
        elif isinstance(corpus_like_object, list) and all(
            isinstance(i, str) for i in corpus_like_object
        ):
            return self.__class__._combine_texts(corpus_like_object)

        # File-like object
        elif hasattr(corpus_like_object, "read"):
            return corpus_like_object.read()

        else:
            raise ValueError(
                "Unsupported input type. Please provide a file path, directory path, "
                "file object, or text string."
            )

    def tokenize(self, **kwargs):
        """
        Tokenize the text.
        """
        self.tokens = []
        self.terms = OrderedDict()

        # Generate tokens.

        # for token in utils.tokenize(self.text):
        for token in self.tokenizer.tokenize(self.text, **kwargs):
            # Gather the tokens.
            self.tokens.append(token)

            # Gather the terms and their offsets.
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

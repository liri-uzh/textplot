

import os
from pathlib import Path
import re
import matplotlib.pyplot as plt
import textplot.utils as utils
import numpy as np
import pkgutil
from typing import Union, List, Optional


from nltk.stem import PorterStemmer
from sklearn.neighbors import KernelDensity
from collections import OrderedDict, Counter
from scipy.spatial import distance
from scipy import ndimage
from functools import lru_cache


class Text:

    @classmethod
    def load_corpus(cls, 
        corpus_like_object: Union[str, Path, List[Union[str, Path]]],
        recursive: bool = False,
        combine_separator: str = "\n\n"
        ):
        """
        Universal method to create a Text instance from various input types.

        Args:
            corpus_like_object: The source data in one of these formats:
                - str: Either a text string or a file/directory path
                - Path: A file or directory path
                - List[str/Path]: A list of text strings or file paths
            recursive (bool): If True and input is a directory, include files in subdirectories.
            combine_separator (str): Separator used when combining multiple texts.

        Returns:
            Text: A Text instance with the combined text from all sources.
        """

        # Case 1: Path object
        if isinstance(corpus_like_object, Path):
            if corpus_like_object.is_file():
                return cls._from_single_file(corpus_like_object)
            elif corpus_like_object.is_dir():
                return cls._from_directory(corpus_like_object, recursive=recursive)
            else:
                raise ValueError(f"Path does not exist: {corpus_like_object}")
        
        # Case 2: String (either file path or text content)
        elif isinstance(corpus_like_object, str):
            # Check if it's a file path
            path_obj = Path(corpus_like_object)
            if path_obj.exists():
                if path_obj.is_file():
                    return cls._from_single_file(path_obj)
                elif path_obj.is_dir():
                    return cls._from_directory(path_obj, recursive=recursive)
            else:
                # Treat as text content
                if not corpus_like_object.strip():
                    raise ValueError("Input string is empty or contains only whitespace")
                
                token_dicts = cls.tokenize_text(corpus_like_object)
                return cls(token_dicts=token_dicts)
        
        # Case 3: List (could be list of strings or file paths)
        elif isinstance(corpus_like_object, list):
            if not corpus_like_object:
                raise ValueError("Empty list provided")
            
            # Check if we have file paths or text strings
            first_item = corpus_like_object[0]
            if isinstance(first_item, (str, Path)) and Path(first_item).is_file():
                # List of file paths
                return cls._from_file_list(corpus_like_object)
            else:
                # List of strings
                valid_texts = [t for t in corpus_like_object if isinstance(t, str) and t.strip()]
                if not valid_texts:
                    raise ValueError("No valid text content in the provided list")
                
                combined_text = combine_separator.join(valid_texts)
                token_dicts = cls.tokenize_text(combined_text)
                return cls(token_dicts=token_dicts)
        
        else:
            raise ValueError(
                "Invalid input type. Must be a string, Path object, or list of strings/paths."
            )

    # For backward compatibility
    @classmethod
    def from_file(cls, path_input, recursive=False):
        """Alias for load_data for backward compatibility"""
        return cls.load_data(path_input, recursive=recursive)

    @classmethod
    def _from_single_file(cls, file_path: Path):
        """
        Create a text from a single file.
        
        Args:
            file_path (Path): The file path.
            
        Returns:
            Text: A Text instance with the file's content.
        """
        text = file_path.read_text(errors='replace')
        token_dicts = cls.tokenize_text(text)
        return cls(token_dicts=token_dicts)
        
    @classmethod
    def _from_directory(cls, dir_path: Path, extension: str = "txt", recursive: bool = False):
        """
        Create a text from all text files in a directory.
        
        Args:
            dir_path (Path): The directory path.
            recursive (bool): Whether to search subdirectories.
            
        Returns:
            Text: A Text instance with the combined content.
        """
        text_files = []
        pattern = f"**/*.{extension}" if recursive else f"*.{extension}"
        text_files.extend(dir_path.glob(pattern))
        if not text_files:
            raise ValueError(f"No files with extension '{extension}' found in the input directory '{dir_path}'")
            
        return cls._from_file_list(text_files)
    
    @classmethod
    def _from_file_list(cls, file_paths: List[Union[str, Path]]):
        """
        Create a text from a list of file paths.
        
        Args:
            file_paths (List[Union[str, Path]]): List of file paths.
            
        Returns:
            Text: A Text instance with the combined content.
        """
        text_content = []
        
        for file_path in file_paths:
            path_obj = file_path if isinstance(file_path, Path) else Path(file_path)
            
            if not path_obj.is_file():
                raise ValueError(f"Not a valid file path: {path_obj}")
                
            text_content.append(path_obj.read_text(errors='replace'))
            
        if not text_content:
            raise ValueError("No valid files provided in the list")
            
        combined_text = "\n\n".join(text_content)
        token_dicts = cls.tokenize_text(combined_text)
        return cls(token_dicts=token_dicts)


    def __init__(self, text=None, token_dicts=None, stopwords=None, pos=['NOUN', 'PROPN'], min_char=2, lang='en'):

        """
        Store the raw text, tokenize.

        Args:
            text (str): The raw text string.
            lower: apply lower-casing
            stopwords (str or list): A custom stopwords path or list
            min_char (int): exclude words with less characters
        """
        if text is not None:
            token_dicts = self.tokenize_text(text)

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

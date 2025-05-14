#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import string
import pkgutil
import logging
import gc

from tqdm import tqdm
from pathlib import Path
from typing import List, Set, Generator, Dict, Optional, Callable, Iterable

from gensim.models.phrases import Phrases

import spacy
from spacy.attrs import ORTH  # for special token cases
from spacy.tokens import Doc, Token

# Import your connector word sets or define them here
from textplot.constants import CONNECTOR_WORDS, BAR_FORMAT, ALLOWED_UPOS
from textplot.utils import parse_wordlists

# Set up logging
logger = logging.getLogger(__name__)

# try:
# 	from memory_profiler import profile # Import the profile decorator
# except ImportError:
# 	def profile(func): # Create a dummy decorator if memory_profiler is not available
# 		return func

class PhrasalTokenizer:
    """
    A class that combines spaCy tokenization with phrase detection using gensim's Phrases.

    This tokenizer identifies multi-word expressions in text and annotates them with
    IOB tags to represent phrases as single tokens during further processing.
    """

    def __init__(
        self,
        lang: str = "en",
        skip_phraser: bool = False,
        min_count: int = 3,
        threshold: float = 0.6,
        scoring: str = "npmi",
        phrase_lemma: str = "full",  # NOTE: "full" seems to work best
        custom_stopwords: Optional[List[str]] = None,
        custom_stopwords_file: Optional[str] = None,
        custom_connector_words: Optional[List[str]] = None,
        custom_connector_words_file: Optional[str] = None,
        labels: Optional[List[str]] = None,
        allowed_upos: Optional[Set[str]] = None,
        disable: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Initialize the PhrasalTokenizer.

        Args:
            lang: Language code ('en', 'de', 'fr', 'it')
            min_count: Minimum count for phrases
            threshold: Threshold for phrase detection
            scoring: Scoring method for phrases ('npmi', 'default', etc.)
            allowed_upos: Set of UPOS tags to exclude from tokenization
            connector_words: Set of words that can appear inside phrases
            disable: spaCy pipeline components to disable
        """
        # spacy nlp arguments
        self.lang = lang
        # optional spacy arguments
        self.allowed_upos = allowed_upos or ALLOWED_UPOS
        self.labels = labels or []
        self.disable = disable or ["ner", "textcat"]  # Default components to disable

        # phrase detection arguments
        self.skip_phraser = skip_phraser
        self.min_count = min_count
        self.threshold = threshold
        self.scoring = scoring
        self.phrase_lemma = phrase_lemma
        self.phraser_model = None

        self.nlp = self._load_spacy_model()
        self._add_tokenizer_exceptions()

        self._register_extensions()

        self._load_custom_stopwords(custom_stopwords, custom_stopwords_file)
        self._load_connector_words(custom_connector_words, custom_connector_words_file)

        logger.debug(f"Allowed UPOS tags: {self.allowed_upos}")

    def _load_spacy_model(self):
        """Load the appropriate spaCy model for the specified language"""
        try:
            if self.lang == "en":
                nlp = spacy.load("en_core_web_sm", disable=self.disable)
            elif self.lang == "de":
                nlp = spacy.load("de_core_news_sm", disable=self.disable)
            elif self.lang == "fr":
                nlp = spacy.load("fr_core_news_sm", disable=self.disable)
            elif self.lang == "it":
                nlp = spacy.load("it_core_news_sm", disable=self.disable)
            else:
                raise ValueError(f"Unsupported language: {self.lang}")
        except OSError:
            raise OSError(
                f"Language model for '{self.lang}' not found. "
                f"Please install the appropriate spaCy model. "
                f"Example: `python -m spacy download {self.lang}_core_news_sm`"
            )

        logger.info(f"Initialized PhrasalTokenizer with language: {self.lang}")
        return nlp

    def _add_tokenizer_exceptions(self):
        """Add special cases to the tokenizer for the provided labels"""

        for label in self.labels:
            self.nlp.tokenizer.add_special_case(label, [{ORTH: label}])
            logger.debug(f"Added special case for token: {label}")

    def _register_extensions(self):
        """Register custom extensions for tokens"""
        if not Token.has_extension("phrase_iob"):
            Token.set_extension("phrase_iob", default="O")

    def _load_custom_stopwords(
        self,
        custom_stopwords: Optional[List[str]] = None,
        custom_stopwords_file: Optional[str] = None,
    ):
        """Load custom stopwords if a file or list is provided"""

        custom_stopwords = parse_wordlists(
            wordlist_file=custom_stopwords_file,
            wordlist=custom_stopwords,
        )

        # Update the nlp pipeline with user-defined stopwords
        self.nlp.Defaults.stop_words.update(custom_stopwords)
        logger.debug(f"Stopwords: {self.nlp.Defaults.stop_words}")

    def _load_connector_words(
        self,
        custom_connector_words: Optional[List[str]] = None,
        custom_connector_words_file: Optional[str] = None,
    ):
        """Load custom connector words if a file or list is provided"""

        custom_connector_words = parse_wordlists(
            wordlist_file=custom_connector_words_file,
            wordlist=custom_connector_words,
        )

        # Update the connector words with user-defined connector words
        self.connector_words = CONNECTOR_WORDS[self.lang]
        # Extend connector words with punctuation and custom connector words
        self.connector_words.update(string.punctuation)
        self.connector_words.update(custom_connector_words)
        logger.debug(f"Connector words: {self.connector_words}")


    def add_phrase_iob_annotations(
        self, doc: Doc, sentences_ngrams: List[List[str]], verbose: bool = False
    ) -> Doc:
        """
        Add IOB annotations to tokens in a spaCy Doc indicating whether they are part
        of a multi-token phrase. Works at the sentence level.

        Args:
            doc: A spaCy Doc object
            sentences_ngrams: A list of lists, where each inner list contains ngrams for a sentence
                            Multi-token phrases are represented with words joined by '_'
            verbose: Whether to print token annotations for debugging

        Returns:
            Doc with phrase_iob annotations
        """
        # Process each sentence in the Doc
        for sent_idx, sent in enumerate(doc.sents):
            # Skip if we don't have ngrams for this sentence
            if sent_idx >= len(sentences_ngrams):
                continue

            # Get the ngrams for this sentence
            ngrams = sentences_ngrams[sent_idx]

            # Get tokens for this sentence
            sent_tokens = list(sent)

            # Process each ngram in this sentence
            for ngram in ngrams:
                if "_" in ngram:  # This is a multi-token phrase
                    words = ngram.split("_")

                    # Find the starting index of this phrase in the sentence tokens
                    for i in range(len(sent_tokens) - len(words) + 1):
                        if all(
                            sent_tokens[i + j].text == words[j]
                            for j in range(len(words))
                        ):
                            # Set B tag for the first token
                            sent_tokens[i]._.phrase_iob = "B"

                            # Set I tag for subsequent tokens in the phrase
                            for j in range(1, len(words)):
                                sent_tokens[i + j]._.phrase_iob = "I"

            if verbose:
                # Print the tokens with their IOB tags (for verification)
                for token in sent_tokens:
                    print(f"{token.text}: {token._.phrase_iob}")

        # Return the Doc with annotations added
        return doc

    # @profile
    def learn_phrases(self, chunk_iter, verbose: bool = False, **kwargs):
        """
        Learn phrases from a streaming input of text chunks using the learned phraser model.

        Args:
            chunk_iter: Iterable yielding text chunks (strings)
            verbose: Debug output

        Returns:
            A Phrases object containing the learned phrases
        """
        
        bigram_model = Phrases(
            min_count=self.min_count,
            threshold=self.threshold,
            scoring=self.scoring,
            connector_words=self.connector_words,
        )
        trigram_model = Phrases(
            min_count=self.min_count,
            threshold=self.threshold,
            scoring=self.scoring,
            connector_words=self.connector_words,
        )

        # Stream over chunks, extract sentences, and update the phrase models
        for chunk in tqdm(chunk_iter, desc="Learning phrases...", bar_format=BAR_FORMAT):
            for doc in self.nlp.pipe([chunk], n_process=1):
                doc_sent_tokens = [[token.text for token in sent] for sent in doc.sents]
                bigram_model.add_vocab(doc_sent_tokens)
                trigram_model.add_vocab(bigram_model[doc_sent_tokens])

        bigram_model.freeze()
        trigram_model.freeze()
        
        self.phraser_model = trigram_model

        # Explicitly free memory
        del bigram_model
        gc.collect()

    def apply_phraser(self, doc: Doc, verbose: bool = False) -> Doc:
        """
        Apply the phraser model to a spaCy Doc object.

        Args:
            doc: A spaCy Doc object
            verbose: Whether to print token annotations for debugging

        Returns:
            Doc with phrase_iob annotations
        """
        if self.skip_phraser or self.phraser_model is None:
            # do nothing
            # logging.warning("No phraser model available. Skipping phrase extraction.")
            return doc

        doc_sent_tokens = [
            [token.text for token in sent if not token.is_space] for sent in doc.sents
        ]
        # Apply the phraser model to the sentences
        doc_phrases = self.phraser_model[doc_sent_tokens]
        # Create a new Doc object with the updated tokens
        doc = self.add_phrase_iob_annotations(doc, doc_phrases, verbose=verbose)

        return doc

    def extract_phrase_spans(self, doc: Doc) -> Generator[Dict[str, str], None, None]:
        """
        Extracts tokens and phrases from a Doc with IOB phrase annotations.

        For tokens that are part of a phrase (marked with B/I tags), it yields a dict with:
        - unstemmed: The full phrase text with spaces between tokens
        - stemmed: The lemma of the last token in the phrase if phrase_lemma is "last", otherwise the full phrase text in lowercase

        For tokens outside of phrases (marked with O tag), it yields a dict with:
        - unstemmed: The token's text
        - stemmed: The token's lemma

        This function skips stopwords, punctuation, spaces, and tokens with UPOS tags not in the allowed set.

        Args:
            doc: A spaCy Doc with "phrase_iob" extension

        Yields:
            Dict containing the text and lemma for each token or phrase
        """
        # Ensure the phrase_iob extension exists
        if not Token.has_extension("phrase_iob"):
            raise ValueError(
                "Doc tokens don't have 'phrase_iob' extension. Run add_phrase_iob_annotations first."
            )

        def is_valid_token(token: Token) -> bool:
            """
            Determine if a token is valid for extraction based on specific criteria.

            Invalid tokens are:
                - stopwords
                - punctuation
                - spaces
                - digits
                - tokens with UPOS tags not in the allowed set
            Valid tokens are:
                - tokens marked as labels
                - tokens with UPOS tags in the allowed set
            """

            # if the token is a label, return True
            # (this is to ensure that labels are not excluded by the allowed_upos)
            if self.labels and token.text in self.labels:
                return True

            # if the token meets any of the invalid criteria, return False
            if token.is_stop:
                return False

            if token.is_punct:
                return False

            if token.is_space:
                return False

            if token.is_digit:
                return False

            # if the token is less than 3 characters, return False
            if len(token.text) < 3:
                return False

            if len(self.allowed_upos) > 0 and token.pos_ not in self.allowed_upos:
                return False

            # otherwise, it's a valid token
            return True

        i = 0
        while i < len(doc):
            token = doc[i]

            # If token is outside any phrase
            if token._.phrase_iob == "O":
                if is_valid_token(token):
                    yield {
                        "unstemmed": token.text,
                        "stemmed": token.lemma_,
                        "pos": token.pos_,
                    }
                i += 1

            # If token is the beginning of a phrase
            elif token._.phrase_iob == "B" and token.text not in self.labels:
                # Find the end of this phrase
                phrase_tokens = [token]
                j = i + 1
                while (
                    j < len(doc)
                    and doc[j]._.phrase_iob == "I"
                    and token.text not in self.labels
                ):
                    phrase_tokens.append(doc[j])
                    j += 1

                # Create the phrase text with spaces between tokens
                phrase_text = " ".join(t.text for t in phrase_tokens)

                if self.phrase_lemma == "full":
                    phrase_lemma = phrase_text.lower()
                    phrase_pos = phrase_tokens[0].pos_
                elif self.phrase_lemma == "last":
                    # Get the lemma of the last token in the phrase
                    phrase_lemma = phrase_tokens[-1].lemma_
                    phrase_pos = phrase_tokens[-1].pos_
                else:
                    raise ValueError(
                        "Invalid value for phrase_lemma. Use 'full' or 'last'."
                    )

                yield {
                    "unstemmed": phrase_text,
                    "stemmed": phrase_lemma,
                    "pos": phrase_pos,
                }

                # Move to the token after the phrase
                i = j

            # If we encounter an "I" without a preceding "B", treat it as an "O"
            # (this handles potential errors in IOB tagging)
            else:  # token._.phrase_iob == "I"
                yield {
                    "unstemmed": token.text,
                    "stemmed": token.lemma_,
                    "pos": token.pos_,
                }
                i += 1

    # @profile
    def tokenize(self, chunk_iter_factory: Callable[[], Iterable[str]], verbose: bool = False, **kwargs):
        """
        Tokenize text using spaCy and apply the learned phraser model in a streaming way.

        Args:
            chunk_iter_factory: Callable returning an iterable of text chunks (strings)
            verbose: Debug output

        Yields:
            Dict containing the text and lemma for each token or phrase
        """
        if self.skip_phraser:
            logger.warning("--skip_phraser is set, skipping phrase extraction.")
        else:
            self.learn_phrases(chunk_iter_factory(), verbose=verbose, **kwargs)

        i = 0
        for doc in self.nlp.pipe(chunk_iter_factory(), n_process=1):
            doc = self.apply_phraser(doc, verbose=verbose)
            for token in self.extract_phrase_spans(doc):
                token["offset"] = i
                yield token
                i += 1


class LegacyTokenizer:
    """
    A simple tokenizer that uses regex to extract words and applies Porter stemming.

    This tokenizer provides backward compatibility with older code that relied on
    regex-based word extraction and stemming with NLTK's PorterStemmer.

    It is not as sophisticated as the PhrasalTokenizer and does not support.
    """

    def __init__(self, stopwords: str = None, **kwargs):
        """
        Initialize the LegacyTokenizer.

        Args:
            **kwargs: Additional arguments (for compatibility with other tokenizers)
        """
        from nltk.stem import PorterStemmer

        self.stemmer = PorterStemmer()

        self.load_stopwords(stopwords)

    def load_stopwords(self, path):
        """
        Load a set of stopwords.

        Args:
            path (str): The stopwords file path.
        """

        if path:
            with open(path) as f:
                self.stopwords = set(f.read().splitlines())
        else:
            self.stopwords = set(
                pkgutil.get_data("textplot", "data/stopwords.txt")
                .decode("utf8")
                .splitlines()
            )


    def tokenize(self, chunk_iter_factory: Callable[[], Iterable[str]], verbose: bool = False, **kwargs) -> Generator[Dict[str, str], None, None]:
        """
        Tokenize text using spaCy and apply the learned phraser model in a streaming way.

        Args:
            chunk_iter_factory: Callable returning an iterable of text chunks (strings)
            verbose: Debug output

        Yields:
            Dict containing the text and lemma for each token or phrase
        """

        offset = 0
        
        for chunk in tqdm(chunk_iter_factory(), desc="Extracting tokens...", bar_format=BAR_FORMAT):
            # Extract word tokens using regex (skip numbers, punctuation, etc.)
            tokens = re.finditer(r"[\w']+", chunk.lower())

            for match in tokens:
                # Get the raw token
                unstemmed = match.group(0)
                stemmed = self.stemmer.stem(unstemmed)

                # Skip stopwords
                if unstemmed in self.stopwords:
                    continue

                yield {
                    "stemmed": stemmed,  # Legacy field
                    "unstemmed": unstemmed,  # Legacy field
                    "offset": offset,
                }

                offset += 1
            

# Example usage
if __name__ == "__main__":
    corpus = Path("data/corpora/human_rights.txt")
    text = corpus.read_text(encoding="utf-8")

    # Create tokenizer with the same settings as the original code
    tokenizer = PhrasalTokenizer(
        lang="en",
        min_count=3,
        threshold=0.8,
        allowed_upos={"NOUN", "PROPN", "ADJ", "VERB"},
    )

    # Print the tokenized output
    for token in tokenizer.tokenize(text, verbose=True):
        print(token)

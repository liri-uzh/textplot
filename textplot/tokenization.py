#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import os
import string
import pkgutil
import logging

from tqdm import tqdm
from pathlib import Path
from typing import List, Set, Generator, Dict, Optional, Union

from gensim.models.phrases import Phrases

import spacy
from spacy.tokens import Doc, Token
from spacy.language import Language

# Import your connector word sets or define them here
from textplot.constants import CONNECTOR_WORDS, BAR_FORMAT

# Set up logging
logger = logging.getLogger(__name__)

class PhrasalTokenizer:
    """
    A class that combines spaCy tokenization with phrase detection using gensim's Phrases.
    
    This tokenizer identifies multi-word expressions in text and annotates them with
    IOB tags to represent phrases as single tokens during further processing.
    """
    
    def __init__(
        self,
        lang: str = None,
        min_count: int = 3,
        threshold: float = 0.6,
        scoring: str = "npmi",
        stopwords: str = None,
        allowed_upos: Optional[Set[str]] = None,
        connector_words: Optional[Set[str]] = None,
        disable: Optional[List[str]] = None,
        phrase_lemma: Optional[str] = "full", # NOTE: "full" seems to work best
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
        self.lang = lang
        self.min_count = min_count
        self.threshold = threshold
        self.scoring = scoring
        self.allowed_upos = allowed_upos or set()
        self.phraser_model = None
        self.phrase_lemma = phrase_lemma
                
        # Set default disabled components if none provided
        if disable is None:
            disable = ['ner', 'textcat']
        
        # Load appropriate spaCy model and connector words based on language
        if lang is None:
            logger.warning(
                "Language not specified when initializing PhrasalTokenizer. Defaulting to English. " \
                "If you want to use a different language, please specify it using the '--lang' parameter."
                )
            lang = 'en'

        try:
            if lang == 'en':
                self.nlp = spacy.load('en_core_web_sm', disable=disable)
            elif lang == 'de':
                self.nlp = spacy.load('de_core_news_sm', disable=disable)
            elif lang == 'fr':
                self.nlp = spacy.load('fr_core_news_sm', disable=disable)        
            elif lang == 'it':
                self.nlp = spacy.load('it_core_news_sm', disable=disable)
            else:
                raise ValueError(f"Unsupported language: {lang}")
        except OSError:
            raise OSError(
                f"Language model for '{lang}' not found. " \
                f"Please install the appropriate spaCy model. " \
                f"Example: `python -m spacy download {lang}_core_news_sm`"
            )
        
        logger.info(f"Initialized PhrasalTokenizer with language: {lang}")
        logger.info(f"Hyperparameters for Phrase Detection: min_count: {min_count}, threshold: {threshold}, scoring: {scoring}")
        logger.info(f"Allowed UPOS tags: {allowed_upos}")
        
        # Load stopwords if provided
        if stopwords and Path(stopwords).is_file():
            with open(stopwords) as f:
                user_stopwords = set(f.read().splitlines())
        else:
            user_stopwords = set()
        
        # update the nlp pipeline with user-defined stopwords
        self.nlp.Defaults.stop_words.update(user_stopwords)
        logger.info(f"Loaded custom stopwords from {stopwords}" if stopwords else "No custom stopwords file provided.")
        logger.info(f"Number of stopwords: {len(self.nlp.Defaults.stop_words)}")

        self.connector_words = connector_words or CONNECTOR_WORDS[lang]
        # Extend connector words with punctuation
        self.connector_words.update(string.punctuation)

        logger.info(f"Connector words: {self.connector_words}")
        
        # Register the phrase_iob extension if it doesn't exist
        if not Token.has_extension("phrase_iob"):
            Token.set_extension("phrase_iob", default="O")
    
    def learn_phrases(self, docs: List[Doc]) -> Phrases:
        """
        Learn phrases from a list of spaCy Doc objects.
        
        Args:
            docs: List of spaCy Doc objects
            
        Returns:
            A Phrases object containing the learned phrases
        """
        bigram_model = Phrases(
            min_count=self.min_count, 
            threshold=self.threshold, 
            scoring=self.scoring, 
            connector_words=self.connector_words
        )
        
        trigram_model = Phrases(
            min_count=self.min_count, 
            threshold=self.threshold, 
            scoring=self.scoring, 
            connector_words=self.connector_words
        )
        
        for doc in tqdm(docs, total=len(docs), desc="Learning phrases...", bar_format=BAR_FORMAT):
            doc_sent_tokens = [[token.text for token in sent if not token.is_space] for sent in doc.sents]
            # Apply the bigram model to the sentences
            bigram_model.add_vocab(doc_sent_tokens)
            # Apply the trigram model to the sentences
            trigram_model.add_vocab(bigram_model[doc_sent_tokens])

        # Freeze the models for faster processing
        bigram_model.freeze()
        trigram_model.freeze()
        
        return trigram_model
    
    def add_phrase_iob_annotations(self, doc: Doc, sentences_ngrams: List[List[str]], verbose: bool = False) -> Doc:
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
                if '_' in ngram:  # This is a multi-token phrase
                    words = ngram.split('_')
                    
                    # Find the starting index of this phrase in the sentence tokens
                    for i in range(len(sent_tokens) - len(words) + 1):
                        if all(sent_tokens[i+j].text == words[j] for j in range(len(words))):
                            # Set B tag for the first token
                            sent_tokens[i]._.phrase_iob = "B"
                            
                            # Set I tag for subsequent tokens in the phrase
                            for j in range(1, len(words)):
                                sent_tokens[i+j]._.phrase_iob = "I"
        
            if verbose:
                # Print the tokens with their IOB tags (for verification)
                for token in sent_tokens:
                    print(f"{token.text}: {token._.phrase_iob}")

        # Return the Doc with annotations added
        return doc
    
    @staticmethod
    def split_text_into_chunks(text: str, chunk_size: int = 500) -> List[str]:
        """
        Split a text into chunks of approximately equal size.
        
        Args:
            text: Text to split into chunks
            chunk_size: Maximum number of whitespace-separated tokens per chunk
            
        Returns:
            List of text chunks
        """
        # Split text on whitespace
        whitespace_split_text = re.split(r'\s+', text)
        
        # Group tokens into chunks of specified size
        chunks = [
            ' '.join(whitespace_split_text[i:i+chunk_size]) 
            for i in range(0, len(whitespace_split_text), chunk_size)
        ]
        
        return chunks

    def apply_phraser(self, doc: Doc, verbose: bool = False) -> Doc:
        """
        Apply the phraser model to a spaCy Doc object.
        
        Args:
            doc: A spaCy Doc object
            verbose: Whether to print token annotations for debugging
            
        Returns:
            Doc with phrase_iob annotations
        """
        if self.phraser_model is None:
            raise ValueError("No phraser model available. Run fit() first.")
            
        doc_sent_tokens = [[token.text for token in sent if not token.is_space] for sent in doc.sents]
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
            raise ValueError("Doc tokens don't have 'phrase_iob' extension. Run add_phrase_iob_annotations first.")
        
        i = 0
        while i < len(doc):
            token = doc[i]
            
            # If token is outside any phrase
            if token._.phrase_iob == "O":
                # Skip token if it is a stopword
                if token.is_stop or token.is_punct or token.is_space:
                    i += 1
                    continue

                # Skip token if it is an excluded UPOS tag
                if self.allowed_upos and token.pos_ not in self.allowed_upos:
                    i += 1
                    continue

                yield {"unstemmed": token.text, "stemmed": token.lemma_}
                i += 1
                
            # If token is the beginning of a phrase
            elif token._.phrase_iob == "B":
                # Find the end of this phrase
                phrase_tokens = [token]
                j = i + 1
                while j < len(doc) and doc[j]._.phrase_iob == "I":
                    phrase_tokens.append(doc[j])
                    j += 1
                
                # Create the phrase text with spaces between tokens
                phrase_text = " ".join(t.text for t in phrase_tokens)
                
                if self.phrase_lemma == "full":
                    phrase_lemma = phrase_text.lower()
                elif self.phrase_lemma == "last":
                    # Get the lemma of the last token in the phrase
                    phrase_lemma = phrase_tokens[-1].lemma_
                else:
                    raise ValueError("Invalid value for phrase_lemma. Use 'full' or 'last'.")
                
                yield {"unstemmed": phrase_text, "stemmed": phrase_lemma}
                
                # Move to the token after the phrase
                i = j
            
            # If we encounter an "I" without a preceding "B", treat it as an "O"
            # (this handles potential errors in IOB tagging)
            else:  # token._.phrase_iob == "I"
                yield {"unstemmed": token.text, "stemmed": token.lemma_}
                i += 1
    
    def tokenize(
        self, 
        text: str, 
        chunk_size: int = 500, 
        verbose: bool = False,
        **kwargs
    ) -> Generator[Dict[str, str], None, None]:
        """
        Tokenize text using spaCy and apply the learned phraser model.
        
        Args:
            text: Input text to tokenize
            chunk_size: Size of chunks to split the text into for processing
            verbose: Whether to print token annotations for debugging
            
        Yields:
            Dict containing the text and lemma for each token or phrase
        """
            
        # Split text into chunks for processing
        chunks = self.split_text_into_chunks(text, chunk_size)
        
        # Process chunks with spaCy
        docs = list(self.nlp.pipe(chunks, n_process=os.cpu_count()-1))
        
        # Learn phrases from the documents
        self.phraser_model = self.learn_phrases(docs)

        # Apply the phraser model to each document
        for doc in tqdm(docs, desc="Extracting tokens...", total=len(docs), bar_format=BAR_FORMAT):
            # Apply the phraser model to the document
            doc = self.apply_phraser(doc, verbose=verbose)
            
            # Extract tokens and phrases
            for i, token in enumerate(self.extract_phrase_spans(doc)):
                token["offset"] = i
                yield token
    

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
                pkgutil
                .get_data('textplot', 'data/stopwords.txt')
                .decode('utf8')
                .splitlines()
            )


    def tokenize(self, text: str, **kwargs) -> Generator[Dict[str, str], None, None]:
        """
        Tokenize text using regex and apply Porter stemming.
        
        Args:
            text: Input text to tokenize
            **kwargs: Additional arguments (for API compatibility)
            
        Yields:
            Dict containing token information with keys:
            - stemmed: The stemmed token (legacy format)
            - unstemmed: The unstemmed token (legacy format)
            - offset: Token position in the sequence
        """
        # Extract word tokens using regex (skip numbers, punctuation, etc.)
        tokens = re.finditer(r'[^\W\d_]+', text.lower())

        offset = 0

        for match in tqdm(tokens, desc="Extracting tokens...", bar_format=BAR_FORMAT):
            # Get the raw token
            unstemmed = match.group(0)
            stemmed = self.stemmer.stem(unstemmed)
            
            # Skip stopwords
            if unstemmed in self.stopwords:
                continue

            yield {
                'stemmed': stemmed,    # Legacy field
                'unstemmed': unstemmed,  # Legacy field
                'offset': offset
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
    
    # # # Example of saving and loading a model
    # # # tokenizer.save("phraser_model.pkl")
    # # # new_tokenizer = PhrasalTokenizer(lang="en")
    # # # new_tokenizer.load("phraser_model.pkl")

    # # # Example of using the LegacyTokenizer
    # tokenizer = LegacyTokenizer()
    # tokenizer.fit(text)
    # for token in tokenizer.tokenize(text):
    #     print(token)
    
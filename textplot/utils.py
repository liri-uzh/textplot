
import re
import numpy as np
import functools
import os

from collections import OrderedDict

from itertools import islice
from tqdm import tqdm


def _tokenize_legacy(text, **kwargs):

    """
    Yield tokens.

    Args:
        text (str): The original text.

    Yields:
        dict: The next token.
    """

    import re
    from nltk.stem import PorterStemmer

    stem = PorterStemmer().stem
    tokens = re.finditer(r'[^\W\d_]+', text.lower())

    for offset, match in tqdm(enumerate(tokens), desc="Tokenizing text"):

        # Get the raw token.
        unstemmed = match.group(0)

        yield { # Emit the token.
            'stemmed':      stem(unstemmed),
            'unstemmed':    unstemmed,
            'offset':       offset
            }


def _tokenize_with_spacy(text, lang: str = 'en', chunk_size: int = 500, **kwargs):
    """
    Tokenize using spaCy's efficient pipeline.
    """

    # first we do a regex split to get the chunks
    # TODO: this could be improved by using a paragraph tokenizer
    whitespace_split_text = re.split(r'\s+', text)
    chunks = [' '.join(whitespace_split_text[i:i+chunk_size]) for i in range(0, len(whitespace_split_text), chunk_size)]
    
    # Load spaCy model
    import spacy
    if lang == 'en':
        nlp = spacy.load('en_core_web_sm', disable=['tok2vec', 'ner', 'parser', 'textcat']) # keep parser active for noun chunks
    elif lang == 'de':
        nlp = spacy.load('de_core_news_sm', disable=['tok2vec', 'ner', 'parser', 'textcat'])
    elif lang == 'fr':
        nlp = spacy.load('fr_core_news_sm', disable=['tok2vec', 'ner', 'parser', 'textcat'])
    elif lang == 'it':
        nlp = spacy.load('it_core_news_sm', disable=['tok2vec', 'ner', 'parser', 'textcat'])
    else:
        raise ValueError(f"Unsupported language: {lang}")

    exclude_upos = kwargs.get('exclude_upos', set())
    print(f"Excluding UPOS tags: {exclude_upos}")

    print(f"Tokenizing {len(chunks)} chunks of size {chunk_size}...")
    docs = list(nlp.pipe(chunks, n_process=os.cpu_count()-1))
    
    offset = 0
    for doc in tqdm(docs, desc="Tokenizing with spaCy"):

        if kwargs.get('noun_chunks', False):
            # Use noun chunks
            for chunk in doc.noun_chunks:
                if kwargs.get('filter_stopwords', False) and chunk.root.is_stop:
                    continue
                if chunk.root.is_punct or chunk.root.is_space:
                    continue
                yield { # Emit the token.
                    'stemmed':      chunk.root.lemma_,
                    'unstemmed':    chunk.text,
                    'offset':       offset
                    }
                offset += 1

        for token in doc:

            # skip token if it is a stopword
            if kwargs.get('filter_stopwords', False) and token.is_stop:
                continue

            # skip token if it is a punctuation or space
            if token.is_punct or token.is_space:
                continue
            
            if token.pos_ in exclude_upos:
                continue
                
            yield { # Emit the token.
                'stemmed':      token.lemma_,
                'unstemmed':    token.text,
                'offset':       offset
                }

            offset += 1


def _tokenize_with_gensim(text, **kwargs):

    """
    Tokenize using Gensim's efficient pipeline.
    """
    
    import gensim
    from gensim.utils import simple_preprocess
    
    # NOTE: we are using the nltk sentence splitter as gensim's get_sentences was removed in v4.0
    import nltk

    import spacy
    from spacy.tokens import Doc

    nlp = spacy.load('en_core_web_sm', disable=['tok2vec', 'ner', 'parser', 'textcat'])

    sentences = nltk.sent_tokenize(text)
    sentences = [simple_preprocess(sent, deacc=False, min_len=3) for sent in sentences]
        
    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(sentences, min_count=5, threshold=100) # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[sentences], threshold=100)  

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    
    offset = 0

    for sent_tokens in sentences:
        # Apply the models to the sentence
        sent_bigrams = bigram_mod[sent_tokens]
        sent_trigrams = trigram_mod[sent_bigrams]

        # Create a Doc object from the list of tokens
        doc = Doc(nlp.vocab, words=sent_trigrams)
        # Apply the pipeline components to the Doc object in order to get lemmatized tokens
        for name, component in nlp.pipeline:
            doc = component(doc)

        for token in doc:
            # skip token if it is a stopword
            if kwargs.get('filter_stopwords', False) and token.is_stop:
                continue
            
            # skip token if it is a punctuation or space
            if token.is_punct or token.is_space:
                continue
            
            # skip token if it is an excluded UPOS tag
            if token.pos_ in kwargs.get('exclude_upos', set()):
                continue
            
            yield { # Emit the token.
                'stemmed':      token.lemma_,
                'unstemmed':    token.text,
                'offset':       offset
                }

            offset += 1



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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Demonstrates how to preprocess a corpus with spaCy and build a graph from it.

Example call:
    python -m examples.preprocess_with_spacy
"""

import sys
from pathlib import Path
from textplot.helpers import build_graph

import spacy
from spacy.tokens import Doc
import numpy as np

nlp = spacy.load("en_core_web_sm")



docs = list(nlp.pipe(texts))

# # nlp.vocab.strings.add("coffee")
# # coffee_hash = nlp.vocab.strings["coffee"]
# # print(coffee_hash)
# # coffee_string = nlp.vocab.strings[coffee_hash]
# # print(coffee_string)

# # Initialize optimized tokenization-only pipeline
# nlp = spacy.blank("en")
# nlp.max_length = 10**8  # Increase max text length limit
# Doc.set_extension("token_ids", default=None)  # For efficient storage

# # Configure tokenization parameters
# tokenizer = nlp.tokenizer
# tokenizer.token_match = None  # Disable URL matching if not needed
# tokenizer.url_match = None

# def batch_tokenize(texts, batch_size=10000):
#     """Optimized tokenization generator for large datasets"""
#     for batch in batch_generator(texts, size=batch_size):
#         yield from nlp.pipe(
#             batch,
#             batch_size=batch_size,
#             n_process=-1,  # Use all available cores
#             disable=["tagger", "parser", "ner", "lemmatizer"],
#             as_tuples=False
#         )

# def batch_generator(items, size=1000):
#     """Memory-efficient batch generator"""
#     for i in range(0, len(items), size):
#         yield items[i:i+size]

# # Usage with memory mapping for large files
# file_path = Path(sys.argv[1])
# with open(file_path, "r", encoding="utf-8") as f:
#     texts = (line.strip() for line in f)
    
#     # Process 10M tokens at a time (~1GB RAM usage)
#     for doc in batch_tokenize(texts):
#         tokens = np.array([token.text for token in doc], dtype=object)
#         del doc  # Explicit memory release
#         # Process tokens or write to disk

# # # Usage with memory mapping for large files
# # input_file_or_dir = Path(sys.argv[1])
# # if input_file_or_dir.is_dir():
# #     file_paths = input_file_or_dir.glob("*.txt")
# # else:
# #     file_paths = [input_file_or_dir]

# # for file_path in file_paths:
# #     with open
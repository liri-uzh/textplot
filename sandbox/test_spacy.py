#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Test the tokenizers

python sandbox/test_spacy.py -i /Users/tannon/liri/projects_data/engage/articles/xml_md --limit 200 --batch_size 100 --disable tok2vec parser ner

"""

import sys
from pathlib import Path
from typing import List
import copy
import argparse

import spacy
from spacy.tokens import Doc
import multiprocessing
import time
import os
from pathlib import Path
import json
from typing import List, Dict, Any, Generator, Tuple, Optional, Iterator
import logging
from itertools import islice

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def set_args():
    parser = argparse.ArgumentParser(description="Preprocess text corpus with spaCy")
    parser.add_argument("--input", "-i", required=True, help="Input file or directory containing text")
    parser.add_argument("--output", "-o", required=False, help="Output directory for processed files")
    parser.add_argument("--model", "-m", default="en_core_web_sm", help="spaCy model to use")
    parser.add_argument("--batch_size", "-b", type=int, default=100, help="Batch size for processing")
    parser.add_argument("--disable", "-d", nargs="+", default=[], 
                        help="Pipeline components to disable (e.g., ner parser)")
    parser.add_argument("--log_level", "-l", default="INFO", 
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set logging level")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of documents to process")
    return parser.parse_args()

def lazy_file_reader(input_path: str, limit: Optional[int] = None) -> Generator[Tuple[str, str], None, None]:
    """
    Lazily read text content from either a single file or all files in a directory.
    
    Args:
        input_path: Path to a file or directory containing text files
        limit: Optional limit on number of documents to process
        
    Yields:
        Tuples of (text_content, file_id)
    """
    path = Path(input_path)
    
    # Check if path exists
    if not path.exists():
        raise ValueError(f"Input path {input_path} does not exist")
    
    # Single file case
    if path.is_file():
        logger.info(f"Processing single file: {path}")
        try:
            with open(path, "r", encoding="utf-8") as f:
                if limit == 1 or limit is None:
                    # Process entire file as one document
                    text = f.read()
                    yield (text, path.stem)
                else:
                    # Process file as multiple chunks if limit > 1
                    # This handles large files by breaking them into chunks
                    chunk_size = 100000  # Adjust based on expected document size
                    chunks_processed = 0
                    
                    while True:
                        text = f.read(chunk_size)
                        if not text:
                            break
                        
                        yield (text, f"{path.stem}_chunk{chunks_processed}")
                        chunks_processed += 1
                        
                        if limit is not None and chunks_processed >= limit:
                            break
        except Exception as e:
            logger.error(f"Error reading {path}: {e}")
    
    # Directory case (existing functionality)
    elif path.is_dir():
        input_files = list(path.glob("*"))
        
        if limit is not None:
            input_files = input_files[:limit]
        
        logger.info(f"Found {len(input_files)} files in {input_path}")
        
        for file_path in input_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                logger.debug(f"Read file: {file_path}")
                yield (text, file_path.stem)
            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")
    
    else:
        raise ValueError(f"Input path {input_path} is neither a file nor a directory")


def preprocess_corpus(
    input_dir: str,
    model_name: str = "en_core_web_sm",
    batch_size: int = 100,
    n_process: Optional[int] = None,
    disable: List[str] = None,
    limit: Optional[int] = None
) -> None:
    """
    Preprocess a corpus of text files using spaCy with multiprocessing.
    
    Args:
        input_dir: Directory containing text files
        model_name: spaCy model to use
        batch_size: Number of documents to process in each batch
        n_process: Number of processes to use (defaults to CPU count)
        disable: Pipeline components to disable for faster processing
    """
    start_time = time.time()
        
    logger.info(f"Loading spaCy model: {model_name}")
    nlp = spacy.load(model_name)
    
    # Enable sentence segmentation if not already enabled
    if "parser" not in nlp.pipe_names and (not disable or "parser" not in disable):
        nlp.add_pipe("sentencizer")
    
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Disabling pipeline components: {disable}")

    corpus = []
    total_docs = 0

    for doc in nlp.pipe(lazy_file_reader(input_dir, limit), batch_size=batch_size, disable=disable, as_tuples=True):
        total_docs += 1
        corpus.extend(tokenize(doc[0]))


    elapsed_time = time.time() - start_time
    logger.info(f"Completed processing {total_docs} documents in {elapsed_time:.2f} seconds")
    if total_docs > 0:
        logger.info(f"Average processing time: {elapsed_time/total_docs:.4f} seconds per document")

    return corpus


def tokenize(doc: spacy.tokens.doc.Doc):

    for tok in doc:
        if tok.is_punct or tok.is_space or tok.is_stop or tok.is_digit:
            # print(f"Skipping token: {tok}\t{tok.is_punct}\t{tok.is_space}\t{tok.is_stop}\t{tok.is_digit}")
            continue

        yield {
            'normalized': tok.lemma_,
            'original': tok.text,
        }
        


if __name__ == "__main__":
    
    args = set_args()
        
    # Set logging level
    logger.setLevel(getattr(logging, args.log_level))
    
    corpus = preprocess_corpus(
        input_dir=args.input,
        model_name=args.model,
        batch_size=args.batch_size,
        disable=args.disable,
        limit=args.limit
    )

    print(corpus[:10])

    # breakpoint()

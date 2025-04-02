

import logging
from pathlib import Path
from argparse import ArgumentParser
from typing import Optional, List, Dict, Any

import click

from textplot.text import Text
from textplot.graphs import Skimmer
from textplot.matrix import Matrix
import networkx as nx

from pyvis.network import Network

"""
Example usage:

    python -m textplot.helpers \
        data/corpora/human_rights.txt \
        --tokenizer spacy \
        --lang en
        

"""

def set_args():
    parser = ArgumentParser(description="Build a graph from a text corpus.")
    
    # corpus options
    parser.add_argument("corpus", help="Path to file/directory/string containing the text corpus.")
    parser.add_argument("--file_pattern", default="*.txt", type=str, help="File pattern for directory input.")
    
    # tokenization options
    parser.add_argument("--tokenizer", default=None, type=str, choices=["spacy", "legacy", None], help="Tokenization method. If None, use the legacy regex tokenizer.")
    parser.add_argument("--lang", default=None, type=str, help="Language for spacy tokenizer. If not provided, we assume English.")
    parser.add_argument("--stopwords", default=None, type=str, help="Path to stopwords file. Note, if using spacy, stopwords in this file will be added to the spacy stopwords list.")
    parser.add_argument("--allowed_upos", nargs="*", default=["ADJ", "ADV", "INTJ", "NOUN", "PROPN", "VERB"], help="List of UPOS tags to exclude. See list here: https://universaldependencies.org/u/pos/. By default, we use open-class words only.")
    parser.add_argument("--chunk_size", default=1000, type=int, help="Chunk size for tokenization. If using spacy, this is the number of words to process at a time.")
    
    # phrase extraction options
    parser.add_argument("--phrase_min_count", default=3, type=int, help="Minimum occurrence count for a candidate phrase.")
    parser.add_argument("--phrase_threshold", default=0.8, type=float, help="Threshold for phrase scorer.")
    parser.add_argument("--phrase_scoring", default="npmi", type=str, choices=["npmi", "default"], help="Scoring method for phrases.")
    
    # logging options
    parser.add_argument('-d', '--debug', action="store_const", dest="loglevel", const=logging.DEBUG, default=logging.WARNING, help="Print DEBUG level messages")
    parser.add_argument('-v', '--verbose', help="Print INFO level messages", action="store_const", dest="loglevel", const=logging.INFO)

    # graph options
    parser.add_argument("--term_depth", default=1000, type=int, help="Consider the N most frequent terms.")
    parser.add_argument("--skim_depth", default=10, type=int, help="Connect each word to the N closest siblings.")
    parser.add_argument("--d_weights", action="store_true", help="If true, give 'close' nodes low weights.")

    # plotting options
    parser.add_argument("--height", default=1500, type=int, help="Height of the graph in px.")
    parser.add_argument("--width", default=100, type=int, help="Width of the graph as a percentage.")
    parser.add_argument("--directed", action="store_true", help="If true, plot a directed graph.")
    parser.add_argument("--notebook", action="store_true", help="If true, plot in a notebook.")
    parser.add_argument("--node_size", default=20, type=int, help="Node size.")
    parser.add_argument("--font_size", default=80, type=int, help="Font size.")
    parser.add_argument("--output_file", default=None, type=str, help="Output file name for the html graph.")


    return parser.parse_args()

def build_graph(
    corpus_like_object: str, 
    term_depth: int = 1000, 
    skim_depth: int = 10,
    d_weights: bool = False, 
    **kwargs: Any) -> Skimmer:

    """
    Tokenize a text, index a term matrix, and build out a graph.

    Args:
        corpus_like_object (str): The file path or text content.
        term_depth (int): Consider the N most frequent terms.
        skim_depth (int): Connect each word to the N closest siblings.
        d_weights (bool): If true, give "close" nodes low weights.

    Returns:
        Skimmer: The indexed graph.
    """

    # Load the text and tokenize    
    t = Text(corpus_like_object, **kwargs)
    logging.info(f'Extracted {len(t.tokens)} tokens')

    m = Matrix()

    # Index the term matrix.
    m.index(t, t.most_frequent_terms(term_depth), **kwargs)

    g = Skimmer()

    # Construct the network.
    g.build(t, m, skim_depth, d_weights)

    return g


def plot_graph(
    g: Skimmer,
    height: int = 1500,
    width: int = 100, 
    directed: bool = False, 
    notebook: bool = False,
    node_size: int = 20, 
    font_size: int = 80, 
    output_file: Optional[str] = None
    ) -> None:

    """
    Plot a graph using pyvis.
    Args:
        g (Skimmer): The graph to plot.
        height (int): Height of the graph in px.
        width (int): Width of the graph as a percentage.
        directed (bool): If true, plot a directed graph.
        notebook (bool): If true, plot in a notebook.
        node_size (int): Node size.
        font_size (int): Font size.
        output_file (str): Output file name for the html graph.
    Returns:
        None
    """

    # Create a pyvis network object
    nt = Network(
        height=f"{height}px",
        width=f"{width}%",
        directed=directed,
        notebook=notebook,
        # cdn_resources="remote",
        )

    nt.from_nx(g.graph)

    for n in nt.nodes:
        n["size"] = node_size
        n["font"] = {"size": font_size}

    nt.force_atlas_2based()  # this method showed the best visualisation result
    nt.toggle_physics(True)

    if output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        # generate the visualization and save it as an HTML file
        nt.show(str(output_file), notebook=notebook)
    else:
        print("No output file specified. Graph not saved.")
    return


if __name__ == "__main__":

    args = set_args()
    
    logging.basicConfig(level=args.loglevel, format='%(asctime)s - %(levelname)s - %(message)s')

    if args.tokenizer != "spacy":
        if args.lang is not None:
            logging.warning(f"Language argument is given, but tokenizer is set to {args.tokenizer}. Did you mean to use `--tokenizer spacy`...?")
        if args.allowed_upos is not None:
            logging.warning(f"Allowed UPOS argument is given, but tokenizer is set to {args.tokenizer}. Did you mean to use `--tokenizer spacy`...?")
        if args.phrase_min_count is not None:
            logging.warning(f"Phrase min count argument is given, but tokenizer is set to {args.tokenizer}. Did you mean to use `--tokenizer spacy`...?")
        if args.phrase_threshold is not None:
            logging.warning(f"Phrase threshold argument is given, but tokenizer is set to {args.tokenizer}. Did you mean to use `--tokenizer spacy`...?")
        if args.phrase_scoring is not None:
            logging.warning(f"Phrase scoring argument is given, but tokenizer is set to {args.tokenizer}. Did you mean to use `--tokenizer spacy`...?")
        
    g = build_graph(
        args.corpus, 
        tokenizer=args.tokenizer, 
        lang=args.lang, 
        stopwords=args.stopwords,
        allowed_upos=set(args.allowed_upos),
        chunk_size=args.chunk_size,
        file_pattern=args.file_pattern,
        phrase_min_count=args.phrase_min_count,
        phrase_threshold=args.phrase_threshold,
        phrase_scoring=args.phrase_scoring,
        )

    # Save the graph to a file
    # print(nx.degree_centrality(g.graph))
    plot_graph(
        g, 
        height=args.height,
        width=args.width,
        directed=args.directed,
        notebook=args.notebook,
        node_size=args.node_size,
        font_size=args.font_size,
        output_file=args.output_file,
        )

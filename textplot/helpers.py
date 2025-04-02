

import click
from pathlib import Path
from argparse import ArgumentParser

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

def build_graph(corpus_like_object, term_depth=1000, skim_depth=10,
                d_weights=False, **kwargs):

    """
    Tokenize a text, index a term matrix, and build out a graph.

    Args:
        path (str): The file path.
        term_depth (int): Consider the N most frequent terms.
        skim_depth (int): Connect each word to the N closest siblings.
        d_weights (bool): If true, give "close" nodes low weights.

    Returns:
        Skimmer: The indexed graph.
    """

    # Load the text and tokenize    
    t = Text(corpus_like_object, **kwargs)
    click.echo(f'Extracted {len(t.tokens)} tokens')

    m = Matrix()

    # Index the term matrix.
    # click.echo('\nIndexing terms:')
    m.index(t, t.most_frequent_terms(term_depth), **kwargs)

    g = Skimmer()

    # Construct the network.
    # click.echo('\nGenerating graph:')
    g.build(t, m, skim_depth, d_weights)

    return g


def plot_graph(g, height="1500px", width="100%", directed=False, notebook=False,
    node_size=20, font_size=80, output_file=None):

    nt = Network(
        height=height,
        width=width,
        directed=directed,
        notebook=notebook,
        # cdn_resources="remote",
        )

    nt.from_nx(g.graph)

    for n in nt.nodes:
        n["size"] = node_size
        n["font"] = {"size": 80}

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

    parser = ArgumentParser(description="Build a graph from a text corpus.")
    parser.add_argument("corpus", help="Path to file/directory/string containing the text corpus.")
    parser.add_argument("--file_pattern", default="*.txt", type=str, help="File pattern for directory input.")
    parser.add_argument("--tokenizer", default=None, type=str, choices=["spacy", "gensim", "legacy", None], help="Tokenizer to use.")
    parser.add_argument("--lang", default=None, type=str, help="Language for the tokenizer.")
    parser.add_argument("--stopwords", default=None, type=str, help="Path to stopwords file.")
    parser.add_argument("--allowed_upos", nargs="*", 
        default=["ADJ", "ADV", "INTJ", "NOUN", "PROPN", "VERB"], # open-class words
        help="List of UPOS tags to exclude. See list here: https://universaldependencies.org/u/pos/"
        )
    parser.add_argument("--noun_chunks", action="store_true", help="Use noun chunks.")
    parser.add_argument("--chunk_size", default=1000, type=int, help="Chunk size for processing.")
    parser.add_argument("--output_file", default=None, type=str, help="Output file name.")

    args = parser.parse_args()
    
    g = build_graph(
        args.corpus, 
        tokenizer=args.tokenizer, 
        lang=args.lang, 
        stopwords=args.stopwords,
        allowed_upos=set(args.allowed_upos),
        noun_chunks=args.noun_chunks,
        chunk_size=args.chunk_size,
        file_pattern=args.file_pattern,
        )


    # Save the graph to a file
    # print(nx.degree_centrality(g.graph))
    plot_graph(
        g, 
        height="1500px", 
        width="100%", 
        directed=False, 
        notebook=False,
        node_size=20, 
        font_size=80, 
        output_file=args.output_file,
        )

import logging
from pathlib import Path
from argparse import ArgumentParser
from typing import Dict, Any, List, Optional


from textplot.text import Text
from textplot.graphs import Skimmer
from textplot.matrix import Matrix



"""
Example usage:

    python -m textplot.helpers \
        data/corpora/human_rights.txt \
        --tokenizer spacy \
        --lang en
        

"""


def set_args() -> ArgumentParser:
    """
    Set the arguments for the script.

    Returns:
        ArgumentParser: The argument parser.
    """

    parser = ArgumentParser(description="Build a graph from a text corpus.")

    # corpus options
    parser.add_argument(
        "corpus", help="Path to file/directory/string containing the text corpus."
    )
    parser.add_argument(
        "--file_pattern",
        default="*.txt",
        type=str,
        help="File pattern for directory input.",
    )
    parser.add_argument(
        "--target_column",
        default=None,
        type=str,
        help="Target column for CSV input.",
    )
    parser.add_argument(
        "--sort_by",
        nargs="*",
        default=None,
        type=str,
        help="Column to sort by for CSV input.",
    )

    # tokenization options
    parser.add_argument(
        "--tokenizer",
        default=None,
        type=str,
        choices=["spacy", "legacy", None],
        help="Tokenization method. If None, use the legacy regex tokenizer.",
    )

    parser.add_argument(
        "--lang",
        default=None,
        type=str,
        help="Language for spacy tokenizer. If not provided, we assume English.",
    )

    parser.add_argument(
        "--custom_stopwords_file",
        default=None,
        type=str,
        help="Path to custom stopwords file. Note, if using spacy, words provided through this argument will be added to the spacy stopwords list.",
    )
    parser.add_argument(
        "--custom_stopwords",
        nargs="*",
        default=None,
        type=str,
        help="List of custom stopwords. Note, if using spacy, words provided through this argument will be added to the spacy stopwords list.",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        type=str,
        help="List of labels if used to annotate the texts. These words will be retained even if they are not in the allowed_upos list.",
    )
    parser.add_argument(
        "--allowed_upos",
        nargs="*",
        default=None,
        type=str,
        help="List of UPOS tags to exclude. See list here: https://universaldependencies.org/u/pos/. By default, we use open-class words only.",
    )
    parser.add_argument(
        "--chunk_size",
        default=1,
        type=int,
        help="Number of words or lines per chunk (default: 1000)."
    )
    parser.add_argument(
        "--chunk_by",
        default="line",
        type=str,
        choices=["word", "line"],
        help="Chunking mode for tokenization. 'word' chunks by words, 'line' chunks by lines.",
    )

    # phrase extraction options
    parser.add_argument(
        "--skip_phraser",
        action="store_true",
        help="If true, skip phrase extraction.",
    )
    
    parser.add_argument(
        "--phrase_min_count",
        default=3,
        type=int,
        help="Minimum occurrence count for a candidate phrase.",
    )
    parser.add_argument(
        "--phrase_threshold",
        default=0.8,
        type=float,
        help="Threshold for phrase scorer.",
    )
    parser.add_argument(
        "--phrase_scoring",
        default="npmi",
        type=str,
        choices=["npmi", "default"],
        help="Scoring method for phrases.",
    )
    parser.add_argument(
        "--custom_connector_words_file",
        default=None,
        type=str,
        help="Path to custom connector words file, which will be added to the default connector words list (see textplot/constants.py for more info).",
    )
    parser.add_argument(
        "--custom_connector_words",
        nargs="*",
        default=None,
        type=str,
        help="List of custom connector words, which will be added to the default connector words list (see textplot/constants.py for more info).",
    )

    # logging options
    parser.add_argument(
        "-d",
        "--debug",
        action="store_const",
        dest="loglevel",
        const=logging.DEBUG,
        default=logging.WARNING,
        help="Print DEBUG level messages",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="Print INFO level messages",
        action="store_const",
        dest="loglevel",
        const=logging.INFO,
    )

    # graph options
    parser.add_argument(
        "--term_depth",
        default=1000,
        type=int,
        help="Consider the N most frequent terms.",
    )
    parser.add_argument(
        "--skim_depth",
        default=10,
        type=int,
        help="Connect each word to the N closest siblings.",
    )
    parser.add_argument(
        "--d_weights",
        action="store_true",
        help="If true, give 'close' nodes low weights.",
    )
    parser.add_argument(
        "--bandwidth",
        default=2000,
        type=int,
        help="Bandwidth for the graph. This is the number of nodes to consider when calculating the distance between nodes.",
    )

    # plotting options
    parser.add_argument(
        "--height", default=1500, type=int, help="Height of the graph in px."
    )
    parser.add_argument(
        "--width", default=100, type=int, help="Width of the graph as a percentage."
    )
    parser.add_argument(
        "--directed", action="store_true", help="If true, plot a directed graph."
    )
    parser.add_argument(
        "--notebook", action="store_true", help="If true, plot in a notebook."
    )
    parser.add_argument("--node_size", default=20, type=int, help="Node size.")
    parser.add_argument("--font_size", default=80, type=int, help="Font size.")
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        help="Output directory to save all files. File names will be inferred from the input file name and args provided",
    )

    return parser.parse_args()


def build_graph(
    corpus_like_object: str,
    term_depth: int = 1000,
    skim_depth: int = 10,
    d_weights: bool = False,
    preprocessing_kwargs: Dict[str, Any] = {},
    **kwargs: Any,
) -> Skimmer:
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

    # Initialise the lazy text object and tokenizer
    t = Text(corpus_like_object, **preprocessing_kwargs)
    # if preprocessing_kwargs.get("tokenizer") in ["spacy", "phrasal"]:
    #     tokenizer = PhrasalTokenizer(**preprocessing_kwargs)
    # else:
    #     tokenizer = LegacyTokenizer(**preprocessing_kwargs)

    # t.tokenize(tokenizer, chunk_size=preprocessing_kwargs.get("chunk_size"), chunk_by=preprocessing_kwargs.get("chunk_by"))
    logging.info(f"Extracted {len(t.tokens)} tokens")

    m = Matrix()

    # Index the term matrix.
    m.index(t, t.most_frequent_terms(term_depth), **kwargs)

    g = Skimmer()

    # Construct the network.
    g.build(t, m, skim_depth, d_weights)

    return g


def prepare_corpus_from_csv(csv_file: str, output_dir: str, target_column: str, sort_by: Optional[List[str]] = None) -> None:
    """
    Prepare the CSV file for output.

    Args:
        csv_file (str): The CSV file to prepare.
        output_dir (str): The output directory to save the corpus with which we build the graph.
    """

    import pandas as pd

    if Path(csv_file).suffix == ".csv":
        df = pd.read_csv(csv_file, sep=",", header=0)
    elif Path(csv_file).suffix == ".tsv":
        df = pd.read_csv(csv_file, sep="\t", header=0)
    else:
        raise ValueError(
            f"File {csv_file} is not a CSV or TSV file. Please provide a valid file."
        )
    
    # Check if the target column exists
    if target_column not in df.columns:
        raise ValueError(
            f"Column {target_column} not found in the CSV file. Please provide a valid column name."
        )
    
    # Sort the dataframe by the target column
    if sort_by is not None:
        for col in sort_by:
            if col not in df.columns:
                raise ValueError(
                    f"Column {sort_by} not found in the CSV file. Please provide a valid column name."
                )
        df = df.sort_values(by=sort_by)
    
    # deduplicate the dataframe
    df = df.drop_duplicates(subset=[target_column])

    # Save the target column to a new CSV file
    output_file = Path(output_dir) / f"{Path(csv_file).stem}_{target_column}.csv"
    # make the output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    # save the target column to a new CSV file
    df[[target_column]].to_csv(output_file, index=False, header=False)
    logging.info(f"Saved the target column {target_column} to {output_file}")

    return str(output_file)


def infer_output_filename_from_args(args: ArgumentParser) -> Path:
    """
    Infer the output filename from the arguments.
    Args:
        args (Namespace): The arguments.
    Returns:
        str: The output filename.
    """
    if args.output_dir is None:
        output_dir = Path(args.corpus).parent
    else:
        output_dir = Path(args.output_dir)

    # make the output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file_stem = Path(args.corpus).stem
    output_file_stem += f"-td{args.term_depth}"
    output_file_stem += f"-sd{args.skim_depth}"
    output_file_stem += f"-bw{args.bandwidth}"
    output_file_stem += f"-dw{args.d_weights}"

    return output_dir / output_file_stem


if __name__ == "__main__":
    args = set_args()

    logging.basicConfig(
        level=args.loglevel, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    if args.tokenizer != "spacy":
        if args.lang is not None:
            logging.warning(
                f"Language argument is given, but tokenizer is set to {args.tokenizer}. Did you mean to use `--tokenizer spacy`...?"
            )
        if args.allowed_upos is not None:
            logging.warning(
                f"Allowed UPOS argument is given, but tokenizer is set to {args.tokenizer}. Did you mean to use `--tokenizer spacy`...?"
            )
        if args.phrase_min_count is not None:
            logging.warning(
                f"Phrase min count argument is given, but tokenizer is set to {args.tokenizer}. Did you mean to use `--tokenizer spacy`...?"
            )
        if args.phrase_threshold is not None:
            logging.warning(
                f"Phrase threshold argument is given, but tokenizer is set to {args.tokenizer}. Did you mean to use `--tokenizer spacy`...?"
            )
        if args.phrase_scoring is not None:
            logging.warning(
                f"Phrase scoring argument is given, but tokenizer is set to {args.tokenizer}. Did you mean to use `--tokenizer spacy`...?"
            )
    
    if args.corpus.endswith(".csv") or args.corpus.endswith(".tsv"):
        # Prepare the corpus from the CSV file
        output_file = prepare_corpus_from_csv(
            args.corpus,
            args.output_dir,
            target_column=args.target_column,
            sort_by=args.sort_by,
        )
        args.corpus = output_file
    
    if args.chunk_by == "line" and args.chunk_size > 1:
        # If chunking by line, set the chunk size to 1
        logging.warning(
            f"Chunking by line, but chunk size is set to {args.chunk_size}. Setting chunk size to 1."
        )
        args.chunk_size = 1
    elif args.chunk_by == "word" and args.chunk_size == 1:
        # If chunking by word, set the chunk size to 1000
        logging.warning(
            f"Chunking by word, but chunk size is set to {args.chunk_size}. Setting chunk size to 1000."
        )
        args.chunk_size = 1000


    # returns a Skimmer object
    g = build_graph(
        args.corpus,
        term_depth=args.term_depth,
        skim_depth=args.skim_depth,
        d_weights=args.d_weights,
        bandwidth=args.bandwidth,
        preprocessing_kwargs={
            "tokenizer": args.tokenizer,
            "lang": args.lang,
            "custom_stopwords_file": args.custom_stopwords_file,
            "custom_stopwords": args.custom_stopwords,
            "labels": args.labels,
            "allowed_upos": args.allowed_upos,
            "chunk_size": args.chunk_size,
            "chunk_by": args.chunk_by,
            "file_pattern": args.file_pattern,
            "skip_phraser": args.skip_phraser,
            "phrase_min_count": args.phrase_min_count,
            "phrase_threshold": args.phrase_threshold,
            "phrase_scoring": args.phrase_scoring,
            "custom_connector_words_file": args.custom_connector_words_file,
            "custom_connector_words": args.custom_connector_words,
        },
    )

    logging.info(
        f"Graph built with {len(g.graph.nodes)} nodes and {len(g.graph.edges)} edges"
    )

    # Infer the output filename from the arguments
    output_file_path = infer_output_filename_from_args(args)

    # Write the graph to a GML file
    g.write_gml(f"{output_file_path}.gml")  # GML format
    print(f"Graph written to {output_file_path}.gml.")
    
    # Write the graph to a GraphML file
    # g.write_graphml(f"{output_file_path}.graphml") # XML format
    # print(f"Graph written to {output_file_path}.graphml.")

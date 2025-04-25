#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import logging
import itertools
from pathlib import Path
from argparse import ArgumentParser
from typing import Dict, Optional, List

from tqdm import tqdm
import seaborn as sns
from matplotlib import pyplot as plt

import networkx as nx


def plot_graph(
    G: nx.Graph,
    subgraph_nodes: Optional[List[str]] = None,
    layout_kwargs: Dict = {},
    plotting_kwargs: Dict = {},
    label_subgraph_nodes: bool = False,
    save_as: str = None,
    verbose: bool = False,
):
    """
    Highlights a subgraph within the context of the entire graph with separated layout and plotting arguments.

    Args:
        G: The entire NetworkX graph.
        subgraph_nodes: A list or set of node labels that constitute the subgraph.
        layout_algorithm: The NetworkX layout algorithm to use for positioning nodes.
        layout_kwargs: Keyword arguments to pass to the layout algorithm.
        plotting_kwargs: Keyword arguments for plotting customization.
        label_subgraph_nodes: Whether to label the subgraph nodes.
        save_as: Filename to save the plot.
    """

    # 1. Set default values for plotting arguments
    figsize = plotting_kwargs.get("figsize", (10, 10))
    font_size = plotting_kwargs.get("font_size", 10)
    font_color = plotting_kwargs.get("font_color", "black")
    node_size = plotting_kwargs.get("node_size", 50)
    negative_alpha = plotting_kwargs.get("negative_alpha", 0.5)
    palette = plotting_kwargs.get("palette", "viridis")
    main_graph_color = plotting_kwargs.get("main_graph_color", "grey")

    plt.figure(figsize=figsize)

    if not subgraph_nodes:
        subgraph_nodes = []

    # 2. Set default values for layout arguments
    # TODO: improve handling for alg-specific arguments
    if not layout_kwargs:
        layout_kwargs = {"algorithm": "spring_layout"}

    layout_algorithm = layout_kwargs.pop("algorithm", "spring_layout")
    if layout_algorithm == "spring_layout":
        layout_algorithm = nx.spring_layout
    elif layout_algorithm in ["forceatlas2_layout", "fa2"]:
        layout_algorithm = nx.forceatlas2_layout
    else:
        raise ValueError(f"Unsupported layout algorithm: {layout_algorithm}")
    layout_kwargs["seed"] = layout_kwargs.get(
        "seed", 42
    )  # Set a default seed for reproducibility
    if layout_algorithm == nx.spring_layout:
        layout_kwargs["k"] = layout_kwargs.get("k", 0.1)
        if "max_iter" in layout_kwargs:
            layout_kwargs["iterations"] = layout_kwargs.pop("max_iter")
        else:
            layout_kwargs["iterations"] = layout_kwargs.get("iterations", 200)
    elif layout_algorithm == nx.forceatlas2_layout:
        layout_kwargs["pos"] = layout_kwargs.get("pos", None)
        if "iterations" in layout_kwargs:
            layout_kwargs["max_iter"] = layout_kwargs.pop("iterations")
        else:
            layout_kwargs["max_iter"] = layout_kwargs.get("max_iter", 200)

    # 2. Calculate the layout for the entire graph
    # we also check that layout_kwargs are supported for the layout_algorithm
    try:
        pos = layout_algorithm(G, **layout_kwargs)
    except TypeError as e:
        raise Exception(
            " Check that the layout_kwargs are compatible with the chosen algorithm (see the documentation:"
            " https://networkx.org/documentation/stable/reference/generated/networkx.drawing.layout.spring_layout.html)."
        ).with_traceback(e.__traceback__)
    except Exception as e:
        raise Exception(
            f"An error occurred while calculating the layout: {e}"
        ).with_traceback(e.__traceback__)

    if verbose:
        print(
            f"Layout calculated using {layout_algorithm.__name__} with parameters: {layout_kwargs}"
        )

    # 3. Identify the nodes for the main graph (not in the subgraph)
    nodes = [node for node in G.nodes() if node not in subgraph_nodes]
    if nodes and verbose:
        print(f"Plotting graph with {len(nodes)} nodes")

    # Use a single color for non-subgraph nodes
    node_color = main_graph_color
    edge_color = main_graph_color

    # Draw the main graph
    # Nodes
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=nodes,
        node_color=node_color,
        alpha=(1 - negative_alpha) * 0.2,
        node_size=node_size,
    )
    # Edges
    nx.draw_networkx_edges(
        G, pos, alpha=(1 - negative_alpha) * 0.4, edge_color=edge_color
    )
    # Labels (text)
    nx.draw_networkx_labels(
        G,
        pos,
        labels={node: node for node in nodes},
        font_size=font_size,
        font_color=font_color,
        alpha=1.0 - negative_alpha,
    )

    # Optionally, draw highlight the subgraph nodes if provided
    if subgraph_nodes:
        num_subgraph_nodes = len(subgraph_nodes)
        colors = sns.color_palette(palette, n_colors=num_subgraph_nodes)
        if verbose:
            print(f"Using {num_subgraph_nodes} colors from the {palette} palette")

        # Create a color map for the subgraph nodes
        node_color_map = {
            node: colors[i % num_subgraph_nodes]
            for i, node in enumerate(subgraph_nodes)
        }
        subgraph_node_colors = [node_color_map[node] for node in subgraph_nodes]

        # Identify the edges within the subgraph
        subgraph_edges = [
            (u, v) for u, v in G.edges() if u in subgraph_nodes and v in subgraph_nodes
        ]
        if subgraph_edges and verbose:
            print(f"Subgraph nodes: {subgraph_nodes}")
            print(f"Subgraph edges: {subgraph_edges}")

        # subgraph_edge_color = (
        #     colors[0] if colors else main_graph_color
        # )  # Use first color or default
        # subgraph_node_color = (
        #     colors[0] if colors else main_graph_color
        # )  # Use first color or default

        # Draw the subgraph with more emphasis
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=subgraph_nodes,
            node_color=subgraph_node_colors,
            node_size=node_size * 10,
            alpha=0.8,
        )

        if label_subgraph_nodes:
            nx.draw_networkx_labels(
                G,
                pos,
                labels={node: node for node in subgraph_nodes},
                font_size=font_size,
                font_weight="bold",
                alpha=1.0,
            )

    plt.axis("off")

    if save_as:
        plt.savefig(save_as, bbox_inches="tight", dpi=300)
        print(f"Graph saved as {save_as}")
    else:
        plt.show()

    return


# def plot_graph(
#     g: Skimmer,
#     height: int = 1500,
#     width: int = 100,
#     directed: bool = False,
#     notebook: bool = False,
#     node_size: int = 20,
#     font_size: int = 80,
#     output_file: Optional[str] = None
#     ) -> None:

#     """
#     Plot a graph using pyvis.
#     Args:
#         g (Skimmer): The graph to plot.
#         height (int): Height of the graph in px.
#         width (int): Width of the graph as a percentage.
#         directed (bool): If true, plot a directed graph.
#         notebook (bool): If true, plot in a notebook.
#         node_size (int): Node size.
#         font_size (int): Font size.
#         output_file (str): Output file name for the html graph.
#     Returns:
#         None
#     """

#     # Create a pyvis network object
#     nt = Network(
#         height=f"{height}px",
#         width=f"{width}%",
#         directed=directed,
#         notebook=notebook,
#         # cdn_resources="remote",
#         )

#     nt.from_nx(g.graph)

#     for n in nt.nodes:
#         n["size"] = node_size
#         n["font"] = {"size": font_size}

#     nt.force_atlas_2based()  # this method showed the best visualisation result
#     nt.toggle_physics(True)

#     if output_file:
#         Path(output_file).parent.mkdir(parents=True, exist_ok=True)
#         # generate the visualization and save it as an HTML file
#         nt.show(str(output_file), notebook=notebook)
#     else:
#         print("No output file specified. Graph not saved.")
#     return


def tuple_type(strings):
    strings = strings.replace("(", "").replace(")", "")
    mapped_int = map(int, strings.split(","))
    return tuple(mapped_int)


def set_args():
    """
    Set the arguments for the script.
    Returns:
        Namespace: The arguments.
    """

    parser = ArgumentParser(description="Plot a graph with subgraph highlighting.")
    parser.add_argument(
        "input_file", type=str, help="Input GML file"
    )  # "data/outputs/examples/8set_ALL.name_text_source_ASCII_cleaned_w_sentiment_t200_skim5_bw20k.gml"
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output file name. If not provided, will be inferred from input file",
    )  # "data/outputs/examples/8set_ALL.name_text_source_ASCII_cleaned_w_sentiment_t200_skim5_bw20k.png"

    parser.add_argument(
        "--subgraph_nodes", type=str, nargs="*", help="List of subgraph nodes"
    )

    # layout arguments
    parser.add_argument(
        "--layout_algorithm",
        type=str,
        default="spring_layout",
        choices=[
            "spring_layout",
            "forceatlas2_layout",
            "fa2",  # alias for forceatlas2_layout
        ],  # Add more layout algorithms as needed
        help="Layout algorithm to use",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=200,
        help="Maximum iterations for layout algorithm",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=200,
        dest="max_iter",
        help="Maximum iterations for layout algorithm (alias for max_iter)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for layout algorithm"
    )
    parser.add_argument(
        "--k", type=float, default=0.1, help="Spring layout parameter k"
    )

    # plotting arguments
    parser.add_argument(
        "--palette",
        type=str,
        default="blend:#7AB,#EDA",
        help="Color palette for subgraph nodes. See seaborn palettes.",
    )
    parser.add_argument(
        "--label_subgraph_nodes", action="store_true", help="Label subgraph nodes"
    )
    parser.add_argument(
        "--font_size", type=int, default=12, help="Font size for labels"
    )
    parser.add_argument("--node_size", type=int, default=10, help="Node size")
    parser.add_argument(
        "--negative_alpha",
        type=float,
        default=0.5,
        help="Alpha value for non-subgraph nodes",
    )
    parser.add_argument(
        "--figsize", type=tuple_type, default=(12, 12), help="Figure size"
    )
    parser.add_argument(
        "--main_graph_color",
        type=str,
        default="grey",
        help="Color for main graph nodes and edges",
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

    parser.add_argument(
        "--no_trials",
        action="store_true",
        help="Do not run trials, use the provided parameters only."
        " Otherwise, we run a set of trials for different combinations of default parameters."
        " This is useful for exploring the best parameters for plotting.",
    )

    args = parser.parse_args()

    return args


def main():
    """
    Main function to run the script.
    """
    args = set_args()

    logging.basicConfig(
        level=args.loglevel, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    input_file = Path(args.input_file)

    g = nx.read_gml(input_file)

    if args.no_trials:
        search_grid = {
            "layout_algorithm": [args.layout_algorithm],
            "max_iter": [args.max_iter],
            "seed": [args.seed],
        }
    else:
        search_grid = {
            "layout_algorithm": ["forceatlas2_layout"],
            "max_iter": [50, 100, 200, 300, 400, 500],
            "seed": [42, 123, 456, 789, 101112],
        }

    # get the set of all possible combinations of the parameters
    keys, values = zip(*search_grid.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # breakpoint()
    for i, params in tqdm(enumerate(permutations_dicts), total=len(permutations_dicts)):
        logging.info(f"Trial {i + 1}/{len(permutations_dicts)}: {params}")

        perm_id = f"{i + 1:003}"
        output_file = input_file.parent / f"{input_file.stem}-{perm_id}.png"
        settings_file = input_file.parent / f"{input_file.stem}-{perm_id}.json"

        layout_kwargs = {
            "algorithm": params["layout_algorithm"],
            "max_iter": params["max_iter"],
            "seed": params["seed"],
        }

        plotting_kwargs = {
            "palette": args.palette,
            "label_subgraph_nodes": args.label_subgraph_nodes,
            "font_size": args.font_size,
            "node_size": args.node_size,
            "negative_alpha": args.negative_alpha,
            "figsize": args.figsize,
            "main_graph_color": args.main_graph_color,
        }

        plot_graph(
            g,
            subgraph_nodes=args.subgraph_nodes,
            layout_kwargs=layout_kwargs,
            plotting_kwargs=plotting_kwargs,
            label_subgraph_nodes=True,
            verbose=True,
            save_as=output_file,
        )

        # Save the settings to a JSON file
        with open(settings_file, "w", encoding="utf8") as f:
            json.dump(
                {
                    "input_file": str(input_file),
                    "output_file": str(output_file),
                    "layout_kwargs": layout_kwargs,
                    "plotting_kwargs": plotting_kwargs,
                },
                f,
                indent=4,
                ensure_ascii=False,
            )
            logging.info(f"Settings saved to {settings_file}")

    logging.info(f"See results in {input_file.parent}")
    return


if __name__ == "__main__":
    main()

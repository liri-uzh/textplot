# Textplot

This repository contains an extended version of David McClure's `textplot` package, which is a tool for visualizing the structure of a text document. It uses kernel density estimation to create a network of terms based on their co-occurrence in the document.

## What's New?

In this version, we've added the following features:

- Text preprocessing with [SpaCy](https://spacy.io/): The text is tokenized and lemmatized using SpaCy, which allows us to benefit from its advanced NLP capabilities and support for multiple languages. Currently, the package supports English, German, French, and Italian, but you can easily add support for other languages by installing the appropriate SpaCy model and updating the code.
- Phrase detection: The package now includes a phrase detection feature based on [Gensim](https://radimrehurek.com/gensim/models/phrases.html) that allows you to identify and visualize multi-word expressions in the text. This is particularly useful for analyzing texts with complex terminology or idiomatic expressions.
- Filtering by part-of-speech: The package now allows you to filter the terms included in the network based on their [UPOS](https://universaldependencies.org/u/pos/) tags. This can help you focus on specific types of words, such as nouns or verbs, and improve the quality of your analysis.
- Support for multiple input formats: The package now supports multiple input formats, including a single plain text file, a directory of files and a pre-loaded list of strings.

## Setup

To install the package, clone the repository and install the required dependencies:

```bash
# Clone the repository
git clone git@github.com:liri-uzh/textplot.git
cd textplot

# Create a virtual environment (we recommend conda)
conda create -n textplot python=3.11
conda activate textplot
# Install the dependencies
pip install -r requirements.txt
# Install textplot for command line usage
pip install .

# Install the language models for SpaCy
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
python -m spacy download fr_core_news_sm
python -m spacy download it_core_news_sm
```

## Usage

### Command Line

To use the package, you can either run the `textplot` command line tool or import the package in your Python code.
The command line tool provides a simple interface for generating a gml file from a text file.

```bash
textplot generate data/corpora/war-and-peace/war-and-peace.txt data/outputs/war-and-peace.gml
```

### Python

Alternatively, you can run it as a Python module, using the `textplot/helpers.py` script to process the text and compute the network:

```bash
python -m textplot.helpers \
    data/corpora/human_rights/en/human_rights.txt \
    --tokenizer spacy \
    --lang en \
    --allowed_upos NOUN \
    --custom_stopwords_file textplot/data/stopwords.txt \
    --custom_stopwords "article" \
    --phrase_min_count 6 --phrase_threshold 0.6 \
    --bandwidth 2000 --term_depth 200 --skim_depth 5 -d \
    --output_dir data/outputs/human_rights \
```

This command processes the text file `data/corpora/human_rights.txt` using SpaCy for tokenization and lemmatization, filters the terms based on their UPOS tags (in this case, only nouns), and applies phrase detection with Gensim. 
By default, this will create a single output file in the output directory with the same name as the input file, but with the `.gml` extension.


## Plotting

Once you have generated a `.gml` file, you can visualize the network using the `textplot/plotting.py` script or import it into a graph visualization tool like Gephi.

For example, you can use the following command to visualize the network using the `plotting.py` script:

```bash
python -m textplot.plotting data/outputs/human_rights-td200-sd5-bw2000-dwFalse.gml
```

By default, the script runs a series of layout hyperparameters and saves the output png and json files in the same directory as the input file. 
This allows for quick exploration of potential layouts and visualizations for a given network.
If you do not want to explore the layout hyperparameters, you can specify `--no_trials`, in which case, the script will only generate a single plot with the layout parameters specified in the command line.
In this case, the `--iterations` parameter controls the number of iterations for the force-directed layout algorithm, and the `--layout_algorithm` parameter specifies which layout algorithm to use. In this case, we are using the ForceAtlas2 algorithm. Run `python -m textplot.plotting --help` for more options.

An example of the resulting network with `pyvis` is shown below:

![Example network for human rights text](./examples/human_rights-td200-sd5-bw2000-dwFalse.png)

For a full list of options, run `python -m textplot.helpers --help`.

## Working with labelled data

If you have labelled data, you can use the `--labels` option to specify the labels used. 
This ensures that labels are included in the network even if they are tagged as stopwords or filtered out by the part-of-speech filter.
For example, the 8set dataset contains texts labelled with positive and negative sentiment (`sentinegative` and `sentipositive`).

```bash
python -m textplot.helpers \
    data/corpora/8set/8set_ALL.name_text_source_ASCII_cleaned_w_sentiment_h1k.txt \
    --tokenizer spacy \
    --lang en \
    --labels "sentinegative" "sentipositive" \
    --phrase_min_count 6 --phrase_threshold 0.6 \
    --bandwidth 20000 --term_depth 200 --skim_depth 5 -d \
    --output_dir data/outputs/8set
```


## TODOs

- [ ] ~~pre vs. post filtering for POS, key terms, sudo words etc.~~
- [x] add support for sudo words (these are words that need to be kept for the analysis, but are ultimately visualised in the network)
- [x] ~~intermediate output of .gml files for networkx~~
- [x] ~~remove numbers from outputs~~
- [x] improve visualisations with [forceatlas](https://github.com/bhargavchippada/forceatlas2) or post-hoc processing with [gephi](https://gephi.org/)
- [ ] re-scoring and filtering with TF-IDF (needs document boundaries or reference corpus)

## Acknowledgements

Textplot is the work of David McClure, who created the original version of the package. 

The extended version was developed by LiRI at UZH and is based largely on David's work. We would like to thank him for making his code available to the community.

Texplot uses **[numpy](http://www.numpy.org)**, **[scipy](http://www.scipy.org)**, **[scikit-learn](http://scikit-learn.org)**, **[matplotlib](http://matplotlib.org)**, **[networkx](http://networkx.github.io)**, and **[clint](https://github.com/kennethreitz/clint)**.

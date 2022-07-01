# Knowledge Base Construction from Pre-trained Language Models (LM-KBC)

This repository contains dataset for the LM-KBC challenge at ISWC 2022.
We also provide:

- ``evaluate.py``: a script to evaluate the performance of LMs' predictions
- ``baseline.py``: our baseline for this challenge
- ``getting_started.ipynb``: a notebook to help you get started

## Clone or download the dataset

```
$ git clone https://github.com/lm-kbc/dataset.git
$ cd dataset
```

## Install necessary dependencies

Tested on Python 3.9.12, should work on any Python 3.7+.

```
$ pip install -r requirements.txt
```

## Usage

1. Run the baseline on the dev set

```
$ python baseline.py -i data/dev.jsonl -o data/dev.pred.jsonl
```

Required parameters:

- ``-i``: input file path
- ``-o``: output file path

You can also modify the following parameters of the baseline:

- ``--model``: the name of the HuggingFace model to use (
  default: ``bert-large-cased``)
- ``--top_k``: the number of top-k predictions that the model should output (
  default: ``100``)
- ``--threshold``: the probability threshold for choosing final output (
  default: ``0.5``)
- ``--gpu``: id of the GPU to use (default: ``-1``, i.e. using CPU)

2. Run the evaluation script

``` 
$ python evaluate.py -g data/dev.jsonl -p data/dev.pred.jsonl
```

Parameters of the evaluation script:

- ``-g``: the ground truth file
- ``-p``: the prediction file
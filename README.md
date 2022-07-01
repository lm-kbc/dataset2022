# Knowledge Base Construction from Pre-trained Language Models (LM-KBC)

This repository contains dataset for the LM-KBC challenge at ISWC 2022.
We also provide:

- ``evaluate.py``: a script to evaluate the performance of LMs' predictions
- ``baseline.py``: our baseline for this challenge
- ``getting_started.ipynb``: a notebook to help you get started with LM-probing
  and the baseline method

🌟 Please directly go to [**File format**](#data-format) if you are ready to
start with your own solutions.

## Clone or download the dataset

```
$ mkdir lm-kbc ; cd lm-kbc
$ git clone https://github.com/lm-kbc/dataset.git
$ cd dataset
```

## Install necessary dependencies

Tested on Python 3.9.12, should work on any Python 3.7+.

```
$ pip install -r requirements.txt
```

## Usage

### Step 1: Run the baseline on the dev set

```
$ python baseline.py -i data/dev.jsonl -o data/dev.pred.jsonl
```

Required parameters:

- ``-i``: input file path
- ``-o``: output file path

You can also modify the following parameters of the baseline:

- ``--model``: the name of the HuggingFace model to use (
  default: ``bert-large-cased``)
- ``--top_k``: the number of top predictions that the model should output (
  default: ``100``)
- ``--threshold``: the probability threshold for choosing final output (
  default: ``0.5``)
- ``--gpu``: id of the GPU to use (default: ``-1``, i.e. using CPU)

### Step 2: Run the evaluation script

``` 
$ python evaluate.py -g data/dev.jsonl -p data/dev.pred.jsonl
```

Parameters of the evaluation script:

- ``-g``: the ground truth file
- ``-p``: the prediction file

## Data format

We use the json-lines (*.jsonl) format (https://jsonlines.org/).
Please take a look at [``file_io.py``](file_io.py) for how we read the files.

### Ground truth ([``data/dev.jsonl``](data/dev.jsonl) and [``data/train.jsonl``](data/train.jsonl) and our private test files)

Each line of a ground-truth file contains a JSON object with the following
fields:

- ``SubjectEntity``: the subject entity (string)
- ``Relation``: the relation (string)
- ``ObjectEntity``: ground truth object entities

The ``ObjectEntity`` filed could be an empty list (``[]``) such as:

```json
{
  "SubjectEntity": "Hwang Chansung",
  "Relation": "PersonInstrument",
  "ObjectEntity": []
}
```

Otherwise, it must be a list of objects, each of which is a list of an
entity's aliases, such as:

```json
{
  "SubjectEntity": "Dominican republic",
  "Relation": "CountryBordersWithCountry",
  "ObjectEntity": [
    [
      "usa",
      "united states of america"
    ],
    [
      "venezuela"
    ],
    [
      "haiti"
    ]
  ]
}
```

### 🌟🌟🌟 YOUR prediction file

Your prediction file should be in the jsonl format as described
above.
Each line of a valid prediction file contains a JSON object which must
contain at least 3 fields to be used by the evaluation script:

- ``SubjectEntity``: the subject entity (string)
- ``Relation``: the relation (string)
- ``ObjectEntity``: the predicted object entities, which should be a list of entities (string).

You can take a look at the [example prediction file](data/dev.pred.jsonl) to
see how a valid prediction file should look like.

This is how we write our prediction file:

```python
import json

# Fake predictions
predictions = [
    {
        "SubjectEntity": "Dominican republic",
        "Relation": "CountryBordersWithCountry",
        "ObjectEntity": ["Haiti"]
    },
    {
        "SubjectEntity": "Eritrea",
        "Relation": "CountryBordersWithCountry",
        "ObjectEntity": ["Ethiopia"]
    },
    {
        "SubjectEntity": "Estonia",
        "Relation": "CountryBordersWithCountry",
        "ObjectEntity": []
    }

]

fp = "/path/to/your/prediction/file.jsonl"

with open(fp, "w") as f:
    for pred in predictions:
        f.write(json.dumps(pred) + "\n")
```
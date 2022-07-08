# Knowledge Base Construction from Pre-trained Language Models (LM-KBC)

This repository contains dataset for the LM-KBC challenge at ISWC 2022.
We also provide:

- [``evaluate.py``](evaluate.py): a script to evaluate the performance of LMs'
  predictions
- [``baseline.py``](baseline.py): our baseline for this challenge
- [``getting_started.ipynb``](getting_started.ipynb): a notebook to help you get
  started with LM probing
  and the baseline method

🌟 Please directly go to [**Data format**](#data-format) if you are ready to
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

- ``--model``: the name of the HuggingFace model to
  use (default: ``bert-large-cased``)
- ``--top_k``: the number of top predictions that the model should output
  (default: ``100``)
- ``--threshold``: the probability threshold for choosing final output
  (default: ``0.5``)
- ``--gpu``: id of the GPU to use (default: ``-1``, i.e. using CPU)

### Step 2: Run the evaluation script

``` 
$ python evaluate.py -g data/dev.jsonl -p data/dev.pred.jsonl
```

Required parameters of the evaluation script:

- ``-g``: the ground truth file
- ``-p``: the prediction file

Results for our baseline should be:

```text
                               p      r     f1
ChemicalCompoundElement    0.960  0.060  0.083
CompanyParentOrganization  0.960  0.680  0.680
CountryBordersWithCountry  1.000  0.087  0.122
CountryOfficialLanguage    0.957  0.703  0.752
PersonCauseOfDeath         0.880  0.520  0.420
PersonEmployer             1.000  0.000  0.000
PersonInstrument           1.000  0.340  0.340
PersonLanguage             0.900  0.412  0.431
PersonPlaceOfDeath         0.980  0.500  0.500
PersonProfession           1.000  0.000  0.000
RiverBasinsCountry         0.960  0.342  0.381
StateSharesBorderState     0.900  0.000  0.000
*** Average ***            0.958  0.304  0.309
```

## Data format

We use the json-lines (*.jsonl) format (https://jsonlines.org/).
Please take a look at [``file_io.py``](file_io.py) for how we read the files.

### Ground truth ([``data/train.jsonl``](data/train.jsonl) and [``data/dev.jsonl``](data/dev.jsonl) and our private test files)

Each line of a ground-truth file contains a JSON object with the following
fields:

- ``SubjectEntity``: the subject entity (string)
- ``Relation``: the relation (string)
- ``ObjectEntities``: ground truth object entities

The ``ObjectEntities`` field could be an empty list (``[]``) such as:

```json
{
  "SubjectEntity": "Hwang Chansung",
  "Relation": "PersonInstrument",
  "ObjectEntities": []
}
```

Otherwise, it will be a list of objects. In case of multi-token objects, a list of an entity's aliases will be given whenever possible. For example:

```json
{
  "SubjectEntity": "Dominican Republic",
  "Relation": "CountryBordersWithCountry",
  "ObjectEntities": [
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
- ``ObjectEntities``: the predicted object entities, which should be a list of
  entities (strings).

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
        "ObjectEntities": ["haiti", "venezuela", "usa", "germany"]
    },
    {
        "SubjectEntity": "Eritrea",
        "Relation": "CountryBordersWithCountry",
        "ObjectEntities": ["ethiopia"]
    },
    {
        "SubjectEntity": "Estonia",
        "Relation": "CountryBordersWithCountry",
        "ObjectEntities": []
    }

]

fp = "/path/to/your/prediction/file.jsonl"

with open(fp, "w") as f:
    for pred in predictions:
        f.write(json.dumps(pred) + "\n")
```
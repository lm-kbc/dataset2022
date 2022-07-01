## Knowledge Base Construction from Pre-trained Language Models (LM-KBC)

Dataset for the LM-KBC challenge at ISWC 2022

### Download the data

```
wget https://github.com/lm-kbc/dataset/zipball/master.zip
unzip master.zip
cd lm-kbc*
```

### Usage

```
pip install -r requirements.txt
```

To run the baseline script:

```
python baseline.py [-h] [--model_type MODEL_TYPE] [--input_dir INPUT_DIR] 
                  [--prompt_output_dir PROMPT_OUTPUT_DIR] 
                  [--baseline_output_dir BASELINE_OUTPUT_DIR]

optional arguments:
  -h, --help            show this help message and exit
  --model_type MODEL_TYPE
                        HuggingFace model name
  --input_dir INPUT_DIR
                        input directory containing the subject-entities for each relation 
                        to probe the language model
  --prompt_output_dir PROMPT_OUTPUT_DIR
                        output directory to store the prompt output
  --baseline_output_dir BASELINE_OUTPUT_DIR
                        output directory to store the baseline output
```

To run the evaluation script:

```
python evaluate.py [-h] [--input_dir INPUT_DIR] [--ground_truth_dir GROUND_TRUTH_DIR] 
                  [--results_dir RESULTS_DIR]
                  
optional arguments:
  -h, --help            show this help message and exit
  --input_dir INPUT_DIR
                        input directory containing the baseline or your method output
  --ground_truth_dir GROUND_TRUTH_DIR
                        ground truth directory containing true object-entities for the 
                        subject-entities for which the LM was probed and then baseline 
                        or your method was applied
  --results_dir RESULTS_DIR
                        results directory for storing the F1 scores for baseline or your 
                        method
```
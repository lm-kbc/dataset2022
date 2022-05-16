# Knowledge Base Construction from Pre-trained Language Models (LM-KBC)

Dataset for the LM-KBC challenge at ISWC 2022

## Download the data

wget https://github.com/lm-kbc/dataset/zipball/master.zip
unzip master.zip
cd lm_kbc*
## Usage

```
pip install -r requirements.txt

python baseline.py 
--model_type "bert-large-cased" 
--input_dir "./dev/"
--prompt_output_dir "./prompt_output_bert_large_cased/"
--baseline_output_dir "./baseline/"

python evaluate.py 
--input_dir "./baseline/" 
--ground_truth_dir "./dev/"
```
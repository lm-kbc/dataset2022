import pandas as pd
import argparse
import os
from ast import literal_eval
import numpy as np
import pprint
import glob

### Precision = how much of the LM's predictions match with ground truth - |LM intersection GT| / |LM|
def precision(x, y):
    count  = 0
    for pred in x:
        count += 1 if any(pred in string for string in y) else 0
    return (count/len(x))

### Recall = how of the LM's predictions match are within ground truth - |LM intersection GT| / |WD|
def recall(x, y):
    count  = 0
    for pred in x:
        count += 1 if any(pred in string for string in y) else 0
    return (count/len(y))

#### ref: https://en.wikipedia.org/wiki/F-score
#### F1 = (2 * P * R)) / (P + R)
def f1_score(x, y):
    p = precision(x, y)
    r = recall(x, y)
    if (p == r == 0):
        return 0
    return ( ( 2 * p * r ) / ( p + r ) )

def clean_predictions(x):
    return x.lower().strip().replace('.', '').replace(',', '').replace('-', '')

def evaluate(input_dir, ground_truth_dir, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    average_f1 = {}
    for fname in glob.glob(input_dir+'*.csv'):
        prompt_df = pd.read_csv(fname)
        relation = fname.split('/')[-1].split('.')[0] ### getting the relation name from the file name
        ground_truth_df = pd.read_csv(ground_truth_dir+relation+'.csv')
        ground_truth_df['ObjectEntity'] = ground_truth_df['ObjectEntity'].apply(literal_eval)
        
        res_df = []
        for entity, ground_truth_objects in ground_truth_df.groupby(['SubjectEntity'])['ObjectEntity']:
            ground_truth_objects = ground_truth_objects.tolist()
            predictions = prompt_df[prompt_df['SubjectEntity']==entity]['ObjectEntity'].tolist()
            # print ('SubjectEntity: %s' % entity, 'Ground Truth: %s' % ground_truth_objects, 'Predictions: %s' % predictions)
            if (len(predictions) == 0):
                res_df.append({'SubjectEntity':entity, 'Relation': relation, 'F1-score': 0})
            else:
                predictions = [clean_predictions(x) for x in predictions] 
                f1 = f1_score(predictions, ground_truth_objects)
                res_df.append({'SubjectEntity':entity, 'Relation': relation, 'F1-score': f1})
                
        res_df = pd.DataFrame(res_df)
        res_df.to_csv(results_dir+relation+'_results.csv', index=False)
        average_f1[relation] = res_df['F1-score'].mean()
        f1 = round(np.mean(list(average_f1.values()))*100, 2)
                   
    print('averagef F1-score for each relation:')
    pprint.pprint(average_f1)
    print ('Final F1-score: {} %'.format(f1))
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default='./baseline/')
    parser.add_argument("--ground_truth_dir", type=str, default='./dev/')
    args = parser.parse_args()
    print (args)
    
    results_dir = './results/'
    input_dir = args.input_dir
    gt_dir = args.ground_truth_dir
    evaluate(input_dir, gt_dir, results_dir)

if __name__ == '__main__':
    main()
import pandas as pd
import argparse
import os
from ast import literal_eval
import numpy as np
import pprint
import glob

### Precision = how much of the LM's predictions match with ground truth (GT) - |LM intersection GT| / |LM|
def precision(x, y):
    count  = 0
    for pred in x:
        ### if the prediction is substring of any of the ground truth object-entity strings, count is incremented
        count += 1 if any(pred in string for string in y) else 0 
    return (count/len(x))

### Recall = how much of the LM's predictions match are within ground truth (GT) - |LM intersection GT| / |GT|
def recall(x, y):
    count  = 0
    for pred in x:
        ### if the prediction is substring of any of the ground truth object-entity strings, count is incremented
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

### converting the LM predictions into lower case and removing punctuations
def clean_predictions(x):
    return x.lower().strip().replace('.', '').replace(',', '').replace('-', '')

def evaluate(input_dir, ground_truth_dir, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    
    ### dictionary (key:relation, val:F1-score) to store average F1 scores across all subject-entities for a given relation
    average_f1 = {} 
    
    ### looping over all the files in the input directory
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
                ### if no predictions obtained for the subject-entity, then F1-score is 0
                res_df.append({'SubjectEntity':entity, 'Relation': relation, 'F1-score': 0}) 
            else:
                predictions = [clean_predictions(x) for x in predictions] 
                f1 = f1_score(predictions, ground_truth_objects) ### calculating F1-score for the given subject-entity
                res_df.append({'SubjectEntity':entity, 'Relation': relation, 'F1-score': f1})
                
        ### storing the results separately for each relation
        res_df = pd.DataFrame(res_df)
        res_df.to_csv(results_dir+relation+'_results.csv', index=False)
        
        ### calculating the averaged F1-score across all subject-entities
        average_f1[relation] = res_df['F1-score'].mean()
        
    ### calculating the final F1-score averaged across all the relations 
    ### NOTE: this score will be used to rank the participating systems
    f1 = round(np.mean(list(average_f1.values()))*100, 2)
                   
    print('averagef F1-score for each relation:')
    pprint.pprint(average_f1)
    
    print ('Final F1-score: {} %'.format(f1))
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default='./baseline/', 
                        help='input directory containing the baseline or your method output')
    parser.add_argument("--ground_truth_dir", type=str, default='./dev/', 
                        help='ground truth directory containing true object-entities for the subject-entities for which the LM was probed and then baseline or your method was applied')
    parser.add_argument("--results_dir", type=str, default='./results/', 
                        help='results directory for storing the F1 scores for baseline or your method')
    args = parser.parse_args()
    print (args)
    
    input_dir = args.input_dir
    gt_dir = args.ground_truth_dir
    results_dir = args.results_dir
    evaluate(input_dir, gt_dir, results_dir)

if __name__ == '__main__':
    main()
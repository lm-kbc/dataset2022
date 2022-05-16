import pandas as pd 
from transformers import AutoModelForMaskedLM, AutoTokenizer, pipeline
import torch, argparse, os

device = torch.device(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
torch.manual_seed(1000)

def prompt_lm(model_type, top_k, relation, entities, output_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    model = AutoModelForMaskedLM.from_pretrained(model_type).to(device)
    device_id  = -1 if device == torch.device("cpu") else 0
    nlp = pipeline('fill-mask', model=model, tokenizer=tokenizer, top_k=top_k, device=device_id) ### -1 device is for cpu
    res = []
    for ent in entities:
        print ('Probing the {} language model for {} (subject-entity) and {} relation'.format(model_type, ent, relation))
        if(relation == 'CountryBordersWithCountry'):
            prompt = ent + " shares border with {}.".format(tokenizer.mask_token)
        elif(relation == 'CountryOfficialLanguage'):
            prompt = "The official language of " + ent + " is {}.".format(tokenizer.mask_token)
        elif(relation == 'StateSharesBorderState'):
            prompt = ent + " shares border with {} state.".format(tokenizer.mask_token)
        elif(relation == 'RiverBasinsCountry'):
            prompt = ent + " river basins in {}.".format(tokenizer.mask_token)
        elif(relation == 'ChemicalCompoundElement'):
            prompt = ent + " consits of {}, which is an element.".format(tokenizer.mask_token)
        elif(relation == 'PersonLanguage'):
            prompt = ent + " speaks in {}.".format(tokenizer.mask_token)
        elif(relation == 'PersonProfession'):
            prompt = ent + " is a {} by profession.".format(tokenizer.mask_token)
        elif(relation == 'PersonInstrument'):
            prompt = ent + " plays {}, which is an instrument.".format(tokenizer.mask_token)
        elif(relation == 'PersonEmployer'):
            prompt = ent + " is an employer at {}, which is a company.".format(tokenizer.mask_token)
        elif(relation == 'PersonPlaceOfDeath'):
            prompt = ent + " died at {}.".format(tokenizer.mask_token)
        elif(relation == 'PersonCauseOfDeath'):
            prompt = ent + " died due to {}.".format(tokenizer.mask_token)
        elif(relation == 'CompanyParentOrganization'):
            prompt = "The parent organization of " + ent + " is {}.".format(tokenizer.mask_token)
        outputs = nlp(prompt)
        for sequence in outputs:
            res.append({'Prompt': prompt, 'SubjectEntity': ent, 'Relation': relation, 'ObjectEntity' : sequence['token_str'], 'Probability' : round(sequence['score'], 4)})
    res_df = pd.DataFrame(res).sort_values(by=['SubjectEntity', 'Probability'], ascending = (True, False))
    os.makedirs(output_dir, exist_ok=True)
    res_df.to_csv(output_dir+relation+'.csv', index=False)

def baseline(relations, input_dir, output_dir):
    print ('Running the baseline method ...')
    for relation in relations:
        df = pd.read_csv(input_dir+relation+'.csv')
        df = df[df['Probability'] >= 0.5]
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(output_dir+relation+'.csv', index=False)
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default='bert-large-cased')
    parser.add_argument("--input_dir", type=str, default='./dev/')
    parser.add_argument("--prompt_output_dir", type=str, default='./prompt_output_bert_large_cased/')
    parser.add_argument("--baseline_output_dir", type=str, default='./baseline/')
    args = parser.parse_args()
    print (args)

    model_type = args.model_type
    input_dir = args.input_dir
    prompt_output_dir = args.prompt_output_dir
    baseline_output_dir = args.baseline_output_dir
    
    top_k = 100
    relations = ['CountryBordersWithCountry', 
                 'CountryOfficialLanguage', 
                 'StateSharesBorderState', 
                 'RiverBasinsCountry',
                 'ChemicalCompoundElement', 
                 'PersonLanguage', 
                 'PersonProfession', 
                 'PersonInstrument', 
                 'PersonEmployer',
                 'PersonPlaceOfDeath',
                 'PersonCauseOfDeath',
                 'CompanyParentOrganization']
  
    for relation in relations:
        entities = pd.read_csv(input_dir+relation+'.csv')['SubjectEntity'].drop_duplicates(keep='first').tolist()
        prompt_lm(model_type, top_k, relation, entities, prompt_output_dir)    
    
    baseline(relations, prompt_output_dir, baseline_output_dir)
    
if __name__ == '__main__':
    main()
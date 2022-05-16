import argparse
import os
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, pipeline

### using GPU if available
device = torch.device(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
torch.manual_seed(1000)


RELATIONS = [
    "CountryBordersWithCountry",
    "CountryOfficialLanguage",
    "StateSharesBorderState",
    "RiverBasinsCountry",
    "ChemicalCompoundElement",
    "PersonLanguage",
    "PersonProfession",
    "PersonInstrument",
    "PersonEmployer",
    "PersonPlaceOfDeath",
    "PersonCauseOfDeath",
    "CompanyParentOrganization",
]


def prompt_lm(model_type, top_k, relation, entities, output_dir: Path):
    ### using the HuggingFace pipeline to initialize the model and its corresponding tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    model = AutoModelForMaskedLM.from_pretrained(model_type).to(device)
    device_id = (
        -1 if device == torch.device("cpu") else 0
    )  ### -1 device is for cpu, 0 for gpu
    nlp = pipeline(
        "fill-mask", model=model, tokenizer=tokenizer, top_k=top_k, device=device_id
    )  ### top_k defines the number of ranked output tokens to pick in the [MASK] position

    ### for every subject-entity in the entities list, we probe the LM using the below sample prompts
    res = []
    for ent in entities:
        print(
            "Probing the {} language model for {} (subject-entity) and {} relation".format(
                model_type, ent, relation
            )
        )

        ### depending on the relation, we fix the prompt
        if relation == "CountryBordersWithCountry":
            prompt = ent + " shares border with {}.".format(tokenizer.mask_token)
        elif relation == "CountryOfficialLanguage":
            prompt = (
                "The official language of "
                + ent
                + " is {}.".format(tokenizer.mask_token)
            )
        elif relation == "StateSharesBorderState":
            prompt = ent + " shares border with {} state.".format(tokenizer.mask_token)
        elif relation == "RiverBasinsCountry":
            prompt = ent + " river basins in {}.".format(tokenizer.mask_token)
        elif relation == "ChemicalCompoundElement":
            prompt = ent + " consits of {}, which is an element.".format(
                tokenizer.mask_token
            )
        elif relation == "PersonLanguage":
            prompt = ent + " speaks in {}.".format(tokenizer.mask_token)
        elif relation == "PersonProfession":
            prompt = ent + " is a {} by profession.".format(tokenizer.mask_token)
        elif relation == "PersonInstrument":
            prompt = ent + " plays {}, which is an instrument.".format(
                tokenizer.mask_token
            )
        elif relation == "PersonEmployer":
            prompt = ent + " is an employer at {}, which is a company.".format(
                tokenizer.mask_token
            )
        elif relation == "PersonPlaceOfDeath":
            prompt = ent + " died at {}.".format(tokenizer.mask_token)
        elif relation == "PersonCauseOfDeath":
            prompt = ent + " died due to {}.".format(tokenizer.mask_token)
        elif relation == "CompanyParentOrganization":
            prompt = (
                "The parent organization of "
                + ent
                + " is {}.".format(tokenizer.mask_token)
            )

        outputs = nlp(prompt)

        ### saving the top_k outputs and the likelihood scores received with the sample prompt
        for sequence in outputs:
            res.append(
                {
                    "Prompt": prompt,
                    "SubjectEntity": ent,
                    "Relation": relation,
                    "ObjectEntity": sequence["token_str"],
                    "Probability": round(sequence["score"], 4),
                }
            )

    ### saving the prompt outputs separately for each relation type
    res_df = pd.DataFrame(res).sort_values(
        by=["SubjectEntity", "Probability"], ascending=(True, False)
    )

    if output_dir.exists():
        assert output_dir.is_dir()
    else:
        output_dir.mkdir(exist_ok=True, parents=True)

    res_df.to_csv(output_dir / f"{relation}.csv", index=False)


def baseline(relations, input_dir, output_dir: Path):
    print("Running the baseline method ...")

    ### for each relation, we run the baseline method
    for relation in relations:
        df = pd.read_csv(input_dir / f"{relation}.csv")
        df = df[
            df["Probability"] >= 0.5
        ]  ### all the output tokens with >= 0.5 likelihood are chosen and the rest are discarded

        if output_dir.exists():
            assert output_dir.is_dir()
        else:
            output_dir.mkdir(exist_ok=True, parents=True)
        
        df.to_csv(
            output_dir / f"{relation}.csv", index=False
        )  ### save the selected output tokens separately for each relation


def main():
    parser = argparse.ArgumentParser(
        description="Probe a Language Model and Run the Baseline Method on Prompt Outputs"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="bert-large-cased",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="./dev/",
        help="input directory containing the subject-entities for each relation to probe the language model",
    )
    parser.add_argument(
        "--prompt_output_dir",
        type=str,
        default="./prompt_output_bert_large_cased/",
        help="output directory to store the prompt output",
    )
    parser.add_argument(
        "--baseline_output_dir",
        type=str,
        default="./baseline/",
        help="output directory to store the baseline output",
    )
    args = parser.parse_args()
    print(args)

    model_type = args.model_type
    input_dir = Path(args.input_dir)
    prompt_output_dir = Path(args.prompt_output_dir)
    baseline_output_dir = Path(args.baseline_output_dir)

    top_k = 100  ### picking the top 100 ranked prompt outputs in the [MASK] position

    ### call the prompt function to get output for each (subject-entity, relation)
    for relation in RELATIONS:
        entities = (
            pd.read_csv(input_dir / f"{relation}.csv")["SubjectEntity"]
            .drop_duplicates(keep="first")
            .tolist()
        )
        prompt_lm(model_type, top_k, relation, entities, prompt_output_dir)

    ### run the baseline method on the prompt outputs
    baseline(RELATIONS, prompt_output_dir, baseline_output_dir)


if __name__ == "__main__":
    main()

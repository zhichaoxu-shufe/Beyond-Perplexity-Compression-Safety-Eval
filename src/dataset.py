import os
import sys
import time
import json

def read_jsonl(fname):
    with open(fname, "r") as fin:
        json_list = list(fin)
        json_list = [json.loads(i) for i in json_list]
        return json_list

def dataset_type_mapping(dataset_name):
    mapping = {
        "AdvPromptSet": "continuation",
        "realtoxicityprompts": "continuation",
        "toxigen_v2": "continuation",
        "bold": "continuation",
        "regard": "continuation",
    }
    return mapping[dataset_name]


def read_dataset(dataset_name, home_dir="/uufs/chpc.utah.edu/common/home/u1320595/toxicity_eval"):
    input_dir = os.path.join(home_dir, f"raw_datasets/{dataset_name}/")

    if dataset_name == "realtoxicityprompts":
        lines = read_jsonl(os.path.join(input_dir, "realtoxicityprompts_filtered.jsonl"))
        res = []
        for i, line in enumerate(lines):
            line_ = {
                "input_pretokenized": line["prompt"]["text"],
                "group": [],
                "sub_group": [],
            }
            res.append(line_)
        return res

    elif dataset_name == "AdvPromptSet":
        sub_groups = ["male", "female", "transgender", "other_gender", "heterosexual", \
        "homosexual_gay_or_lesbian", "bisexual", "other_sexual_orientation", \
        "christian", "jewish", "muslim", "hindu", "buddhist", "atheist", "other_religion",\
        "black", "white", "asian", "latino", "other_race_or_ethnicity", "physical_disability",\
        "intellectual_or_learning_disability", "psychiatric_or_mental_illness", "other_disability"]

        group_mapping = {
            ("male", "female", "transgender", "other_gender"): "gender",
            ("heterosexual", "homosexual_gay_or_lesbian", "bisexual", "other_sexual_orientation"): "sextuality",
            ("christian", "jewish", "muslim", "hindu", "buddhist", "atheist", "other_religion"): "religion",
            ("black", "white", "asian", "latino", "other_race_or_ethnicity"): "race",
            ("physical_disability", "intellectual_or_learning_disability", "psychiatric_or_mental_illness", "other_disability"): "disability"
        }

        lines = read_jsonl(os.path.join(input_dir, "advpromptset_final_10k.jsonl"))
        res = []
        for i, line in enumerate(lines):
            line_ = {
                "input_pretokenized": line["comment_text"],
                "group": [],
                "sub_group": [],
                "sentiment": ""
            }
            for sub_group in sub_groups:
                if sub_group in line:
                    if line[sub_group] != 0:
                        line_["sub_group"].append(sub_group)
                        for k, v in group_mapping.items():
                            if sub_group in k:
                                line_["group"].append(v)
            
            line_["sentiment"] = "benigh" if line["toxicity"] == 0 else "toxic"
            res.append(line_)
        return res

    elif dataset_name == "bold":
        lines = read_jsonl(os.path.join(input_dir, "bold.jsonl"))
        res = []
        for i, line in enumerate(lines):
            for prompt in line["prompts"]:
                line_ = {
                    "input_pretokenized": prompt,
                    "group": [line["domain"]],
                    "sub_group": [line["category"]]
                }
                res.append(line_)
        return res

    elif dataset_name == "regard":
        lines = read_jsonl(os.path.join(input_dir, "prompts.jsonl"))
        res = []
        for i, line in enumerate(lines):
            line_ = {
                "input_pretokenized": line["prompt_text"],
                "group": [],
                "sub_group": []
            }
            res.append(line_)
        return res 
    
    elif dataset_name == "holistic_bias":
        lines = read_jsonl(os.path.join(input_dir, "prompts.jsonl"))
        res = []
        for i, line in enumerate(lines):
            line_ = {
                "input_pretokenized": line["prompt_text"],
                "group": [line["axis"]],
                # "sub_group": [line["noun_phrase"]]
                "sub_group": [],  # currently disabled, as we have too many noun phrases
            }
            res.append(line_)
        return res
    
    elif dataset_name == "toxigen":
        lines = read_jsonl(os.path.join(input_dir, "toxigen_filtered.jsonl"))  # the robbie paper has a mismatch between paper and codebase, TODO, contact the authors probably
        res = []
        for i, line in enumerate(lines):
            line_ = {
                "input_pretokenized": line["text"],
                "group": [line["target_group"]],
                "sub_group": []
            }
            res.append(line_)
        return res
    
    else:
        raise Exception("unsupported dataset name")
    

def read_multiqa_dataset(dataset_name, category, split=0):
    if dataset_name == "unqover":
        input_dir = "/uufs/chpc.utah.edu/common/home/u1320595/toxicity_eval/raw_datasets/unqover/dataset"
        if category in ["country", "ethnicity", "religion", "gender_occupation"]:
            fname = os.path.join(input_dir, f"{category}.json")
        else:
            raise Exception("unrecognized unqover category, force exit!")

        fin = json.load(open(fname, "r"))
        print(f"category {category} contains in total {len(fin)} pairs of questions")

        return fin
    
    elif dataset_name == "bbq":
        input_dir = "/uufs/chpc.utah.edu/common/home/u1320595/toxicity_eval/raw_datasets/bbq"
        if category in ["age", "disability_status", "gender_identity", "nationality", \
            "physical_appearance", "race_ethnicity", "race_x_gender", "race_x_ses", \
            "religion", "ses", "sexual_orientation"]:
            fname = os.path.join(input_dir, f"{category}.jsonl")
        else:
            raise Exception("unrecognized bbq category, force exit!")

        fin = read_jsonl(fname)
        print(f"category {category} contains in total {len(fin)} questions")

        return fin

    elif dataset_name == "bbq_fewshot":
        input_dir = "/uufs/chpc.utah.edu/common/home/u1320595/toxicity_eval/raw_datasets/bbq_fewshot"
        if category in ["age", "disability_status", "gender_identity", "nationality", \
            "physical_appearance", "race_ethnicity", "race_x_gender", "race_x_ses", \
            "religion", "ses", "sexual_orientation"]:
            train_fname = os.path.join(input_dir, f"{category}_train_split_{split}.jsonl")
            test_fname = os.path.join(input_dir, f"{category}_test_split_{split}.jsonl")
        else:
            raise Exception("unrecognized bbq category, force exit!")
        train = read_jsonl(train_fname)
        test = read_jsonl(test_fname)
        return train, test    
    
    elif dataset_name == "bbq_lite":
        input_dir = "/uufs/chpc.utah.edu/common/home/u1320595/toxicity_eval/raw_datasets/bbq_lite_json"
        category_list = [
            "age_ambig", "age_disambig", "disability_status_ambig", "disability_status_disambig",
            "gender_identity_ambig", "gender_identity_disambig", "nationality_ambig", "nationality_disambig",
            "physical_appearance_ambig", "physical_appearance_disambig", "race_ethnicity_ambig", "race_ethnicity_disambig",
            "religion_ambig", "religion_disambig", "ses_ambig", "ses_disambig", 
            "sexual_orientation_ambig", "sexual_orientation_disambig"
        ]
        assert category in category_list, "unrecognized bbq_lite category, force exit!"
        fname = os.path.join(input_dir, f"{category}/task.json")
        return json.load(open(fname, "r"))
    
    else:
        raise Exception("unsupported dataset name")

def read_mmlu_dataset(subjects, supersubject=None):
    mapping = {
        "abstract_algebra": "stem",
        "anatomy": "stem",
        "astronomy": "stem",
        "business_ethics": "other",
        "clinical_knowledge": "other",
        "college_biology": "stem",
        "college_chemistry": "stem",
        "college_computer_science": "stem",
        "college_mathematics": "stem",
        "college_medicine": "other",
        "college_physics": "stem",
        "computer_security": "stem",
        "conceptual_physics": "stem",
        "econometrics": "social_sciences",
        "electrical_engineering": "stem",
        "elementary_mathematics": "stem",
        "formal_logic": "humanities",
        "global_facts": "other",
        "high_school_biology": "stem",
        "high_school_chemistry": "stem",
        "high_school_computer_science": "stem",
        "high_school_european_history": "humanities",
        "high_school_geography": "social_sciences",
        "high_school_government_and_politics": "social_sciences",
        "high_school_microeconomics": "social_sciences",
        "high_school_macroeconomics": "social_sciences",
        "high_school_mathematics": "stem",
        "high_school_physics": "stem",
        "high_school_psychology": "social_sciences",
        "high_school_statistics": "stem",
        "high_school_us_history": "humanities",
        "high_school_world_history": "humanities",
        "human_aging": "other",
        "human_sexuality": "social_sciences",
        "international_law": "humanities",
        "jurisprudence": "humanities",
        "logical_fallacies": "humanities",
        "machine_learning": "stem",
        "management": "other",
        "marketing": "other",
        "medical_genetics": "other",
        "miscellaneous": "other",
        "moral_disputes": "humanities",
        "moral_scenarios": "humanities",
        "nutrition": "other",
        "philosophy": "humanities",
        "prehistory": "humanities",
        "professional_accounting": "other",
        "professional_law": "humanities",
        "professional_medicine": "other",
        "professional_psychology": "social_sciences",
        "public_relations": "social_sciences",
        "security_studies": "social_sciences",
        "sociology": "social_sciences",
        "us_foreign_policy": "social_sciences",
        "virology": "other",
        "world_religions": "humanities"
    }
    from datasets import load_dataset
    
    if subjects == "all":
        subjects = list(mapping.keys())
        return [load_dataset("cais/mmlu", subject) for subject in subjects], subjects
    elif supersubject is not None:
        # group by supersubject
        subjects = []
        for k, v in mapping.items():
            if v == supersubject:
                subjects.append(k)
        
        return [load_dataset("cais/mmlu", subject) for subject in subjects], subjects
    else:
        subjects = subjects.split(",")
        return [load_dataset("cais/mmlu", subject) for subject in subjects], subjects



if __name__ == "__main__":

    # for dataset in ["toxigen", "realtoxicityprompts", "holistic_bias", "regard", "AdvPromptSet", "bold"]:
    #     lines = read_dataset(dataset)
    #     print(f"dataset name -> {dataset}")
    #     print(f"total number of lines -> {len(lines)}")
    #     print(f"sample line -> {lines[0]}")
    
    # for category in ["country", "ethnicity", "religion", "gender_occupation"]:
    #     read_multiqa_dataset("unqover", category)

    # for category in ["Age", "Disability_status", "Gender_identity", "Nationality", \
    #     "Physical_appearance", "Race_ethnicity", "Race_x_gender", "Race_x_SES", \
    #     "Religion", "SES", "Sexual_orientation"]:
    #     read_multiqa_dataset("bbq", category)
    
    datasets = read_mmlu_dataset(subject=None, supersubject="humanities")
    print(len(datasets))
import os
import sys
import time
import json
import copy
import random
import argparse

import numpy as np
from collections import OrderedDict

import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import AutoConfig

from utils import save_outputs
from utils import save_results
from multi_qa import MultiQA
from unqover_utils import Lists


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # unqover arguments
    parser.add_argument("--word_lists", type=str, default="/uufs/chpc.utah.edu/common/home/u1320595/toxicity_eval/raw_datasets/unqover/word_lists")
    parser.add_argument("--metrics", type=str, help="the unqover metrics to be evaluated, separate with ,")
    parser.add_argument("--group_by", type=str, choices=["subj_act", "subj", "gender_act"], help="group_by is useful when metric==subj_bias")
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--sample_size", type=int, default=10000)
    parser.add_argument("--random_seed", type=int, default=42)

    # mmlu arguments
    parser.add_argument("--subjects", type=str, help="category to be evaluated, separate with ,")
    parser.add_argument("--supersubject", type=str, choices=["stem", "social_sciences", "humanities", "other"])
    parser.add_argument("--n_shots", type=int, default=0)

    # general arguments
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--target_file", type=str)
    parser.add_argument("--category", type=str, help="category to be evaluated, separate with ,")

    parser.add_argument("--do_inference", action="store_true")
    parser.add_argument("--scoring_func", type=str, default="unnormalized")


    parser.add_argument("--tokenizer", type=str)
    parser.add_argument("--model_name_or_path", type=str, default="facebook/opt-125m")
    parser.add_argument("--llm_pruner", action="store_true")
    parser.add_argument("--llm_pruner_home", type=str, default="")
    parser.add_argument("--bonsai", action="store_true")
    parser.add_argument("--bonsai_home", type=str, default="")

    # loading hyperparameters
    parser.add_argument("--loading_mode", type=str, choices=["nf4", "int8", "fp16"], default="fp16")
    parser.add_argument("--awq", action="store_true")
    parser.add_argument("--gptq", action="store_true")
    parser.add_argument("--flash_attention", action="store_true")

    # experiment arguments
    parser.add_argument("--disable_progress_bar", action="store_true")
    parser.add_argument("--save_results", action="store_true")
    parser.add_argument("--results_dest", type=str, default="./logs/default_results.json")
    parser.add_argument("--save_outputs", action="store_true")
    parser.add_argument("--output_dest", type=str)

    args = parser.parse_args()

    config_dict = OrderedDict()
    for k, v in vars(args).items():
        config_dict[k] = v
    for k, v in config_dict.items():
        print(f"{k} -> {v}")

    from evaluate import Evaluator
    evaluator = Evaluator()  # specify the Evaluator class

    if not args.do_inference and not args.target_file:
        raise Exception("need to specify target_file if not running inference on the fly")

    elif args.target_file:
        if args.dataset == "unqover":
            data = json.load(open(args.target_file, "r"))
            results_dict = evaluator.evaluate_unqover(args=args, data=data)
        elif args.dataset == "bbq":
            data = read_jsonl(args.target_file)
            data = [format_bbq_input(line) for line in data]
            results_dict = evaluator.evaluate_bbq(data=data)
        elif args.dataset == "bbq_lite":
            data = json.load(open(args.target_file, "r"))
            results_dict = evaluator.evaluate_bbq_lite(args=args, data=data)
        else:
            raise Exception("underspecified dataset name!")

        for k, v in results_dict.items():
            print(f"{k} -> {v:.3f}")

    elif args.do_inference:
        if not args.category and not args.subjects and not args.supersubject:
            raise Exception("need to specify args.category to run inference")

        # configure model loading
        from run_generation import configure_tokenizer_model_loading
        # config = AutoConfig.from_pretrained(args.model_name_or_path)  # it seems we do not need this

        tokenizer, model = configure_tokenizer_model_loading(args)

        from multi_qa import MultiQA
        engine = MultiQA(tokenizer=tokenizer, model=model)

        if args.dataset == "unqover":
            from dataset import read_multiqa_dataset
            category_list = args.category.split(",")
            results_dict_global = copy.deepcopy(config_dict)

            for category in category_list:
                print(f"category -> {category}")
                data = read_multiqa_dataset("unqover", category)
                assert args.metrics is not None, "must specify unqover metrics"
                if args.do_sample:
                    from unqover_utils import sample_pairs
                    data = sample_pairs(data=data, sample_size=args.sample_size, seed=args.random_seed)

                res = engine.inference_unqover(unqover_dataset=data, args=args)
                results_dict = evaluator.evaluate_unqover(args=args, data=res)
                for k, v in results_dict.items():
                    print(f"{k} -> {v:.3f}")
                results_dict_global[category] = results_dict


        elif args.dataset == "bbq":
            from dataset import read_multiqa_dataset
            from multi_qa import format_bbq_input
            category_list = args.category.split(",")
            results_dict_global = copy.deepcopy(config_dict)

            ambig_bias, disambig_bias = [], []

            for category in category_list:
                print(f"category -> {category}")
                data = read_multiqa_dataset(dataset_name="bbq", category=category)
                data = [format_bbq_input(line) for line in data]
                res = engine.inference_bbq(bbq_dataset=data, args=args)
                
                results_dict = evaluator.evaluate_bbq(res)
                ambig_bias.append(results_dict["s_ambiguous"])
                disambig_bias.append(results_dict["s_disambiguated"])
                for k, v in results_dict.items():
                    print(f"{k} -> {v:.3f}")
                results_dict_global[category] = results_dict

            print("-"*20)
            print(f"avg ambiguous bias -> {np.mean(ambig_bias):.3f}")
            print(f"avg disambiguated bias -> {np.mean(disambig_bias):.3f}")
            print("\n")
            results_dict_global["avg_ambiguous_bias"] = np.mean(ambig_bias)
            results_dict_global["avg_disambiguated_bias"] = np.mean(disambig_bias)
        
        elif args.dataset == "bbq_fewshot":
            from dataset import read_multiqa_dataset
            from multi_qa import format_bbq_demonstrations, format_bbq_input

            category_list = args.category.split(",")
            results_dict_global = copy.deepcopy(config_dict)

            ambig_bias, disambig_bias = [], []

            for category in category_list:
                for fold_id in range(3):
                    train_data, test_data = read_multiqa_dataset(dataset_name="bbq_fewshot", category=category, split=fold_id)
                    if args.do_sample:
                        test_data = test_data[:args.sample_size]
                    demonstrations = format_bbq_demonstrations(train_data)
                    data = [format_bbq_input(input_line=line, demonstrations=demonstrations, model_name_or_path=args.model_name_or_path) for line in test_data]
                    res = engine.inference_bbq(bbq_dataset=data, args=args)
                    
                    results_dict = evaluator.evaluate_bbq(res)
                    ambig_bias.append(results_dict["s_ambiguous"])
                    disambig_bias.append(results_dict["s_disambiguated"])
                    for k, v in results_dict.items():
                        print(f"{k} -> {v:.3f}")
                    results_dict_global[f"{category}_fold_{fold_id}"] = results_dict
        
        elif args.dataset == "bbq_lite":
            from dataset import read_multiqa_dataset
            category_list = args.category.split(",")
            results_dict_global = copy.deepcopy(config_dict)
            ambig_accuracy, disambig_accuracy = [], []
            for category in category_list:
                print(f"category -> {category}")
                data = read_multiqa_dataset(dataset_name="bbq_lite", category=category)
                res = engine.inference_bbq_lite(bbq_lite_dataset=data, args=args)
                results_dict = evaluator.evaluate_bbq_lite(res)
                for k, v in results_dict.items():
                    print(f"{k} -> {v:.3f}")
                results_dict_global[category] = results_dict
                if "disambig" in category:
                    disambig_accuracy.append(results_dict["accuracy"])
                elif "ambig" in category:
                    ambig_accuracy.append(results_dict["accuracy"])
                else:
                    raise Exception("something is wrong with this category")
            print("-"*20)
            print(f"avg ambiguous accuracy -> {np.mean(ambig_accuracy):.3f}")
            print(f"avg disambiguated accuracy -> {np.mean(disambig_accuracy):.3f}")
            print("\n")
        
        elif args.dataset == "mmlu":
            from dataset import read_mmlu_dataset

            datasets, subjects = read_mmlu_dataset(subjects=args.subjects, supersubject=args.supersubject)
            mean_acc = []
            results_dict_global = {}
            for idx, dataset in enumerate(datasets):
                res = engine.inference_mmlu(mmlu_dataset=dataset, subject=subjects[idx], args=args)
                results_dict = evaluator.evaluate_mmlu(data=res)
                mean_acc.append(results_dict["accuracy"])
                results_dict_global[subjects[idx]] = results_dict["accuracy"]
            print("-"*20)
            print(f"mean accuracy -> {np.mean(mean_acc):.3f}")
            results_dict_global["mean_accuracy"] = np.mean(mean_acc)
        
        if args.save_results:
            save_results(dest=args.results_dest, results=results_dict_global)

        if args.save_outputs:  # TODO: what is a good way to save results for future evaluation?
            save_outputs(dest=args.output_dest, data=res)

    else:
        raise Exception("you can only specify target_file or do_inference")




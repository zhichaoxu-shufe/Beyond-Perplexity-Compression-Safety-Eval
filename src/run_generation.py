import os
import sys
import time
import random
import json
import argparse
from collections import OrderedDict

import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import AutoConfig
from transformers import GenerationConfig

from generation import Generator
from evaluate import Evaluator
from dataset import read_dataset
from utils import save_outputs, save_results

def configure_model_loading(args):
    access_token = ""  # TODO: enable passing HF key as argument

    device_allow_flash_attention = True
    bfloat16 = True

    if args.awq:
        from awq import AutoAWQForCausalLM
        model = AutoAWQForCausalLM.from_quantized(
            args.model_name_or_path,
            fuse_layers=False  # fuse_layers=True will lead to weird input, lm logits mismatch bug
        )
    elif args.gptq:
        from auto_gptq import AutoGPTQForCausalLM
        model = AutoGPTQForCausalLM.from_quantized(
            args.model_name_or_path
        )
    elif args.loading_mode == "nf4":
        from transformers import BitsAndBytesConfig
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            )
        
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.float16,
            device_map="balanced",
            quantization_config=nf4_config,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
        )
    elif args.loading_mode == "int8":
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            load_in_8bit=True,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
        )

    elif args.flash_attention:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map="balanced",
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )

    elif args.loading_mode == "fp16":
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.float16,
            device_map="balanced",
            trust_remote_code=True,
        )
    return model

def configure_tokenizer_model_loading(args):
    if args.llm_pruner:
        if not args.llm_pruner_home:
            args.llm_pruner_home="/uufs/chpc.utah.edu/common/home/u1320595/LLM-Pruner"
        sys.path.append(args.llm_pruner_home)
        res = torch.load(os.path.join(args.model_name_or_path, "pytorch_model.bin"))
        tokenizer, model = res["tokenizer"], res["model"]
    elif args.bonsai:
        if not args.bonsai_home:
            args.bonsai_home="/uufs/chpc.utah.edu/common/home/u1320595/Bonsai"
        sys.path.append(args.bonsai_home)
        model = torch.load(os.path.join(args.model_name_or_path, "pytorch_model.bin"))
    else:
        model = configure_model_loading(args)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.padding_side = "left"
    print(f"finish loading model and tokenizer!")

    device = "cuda" if torch.cuda.is_available() else "cpu"  # this needs to be changed if we are doing 30B model
    
    if args.loading_mode == "nf4" or args.loading_mode == "int8":
        return tokenizer, model
    else:
        return tokenizer, model.to(device)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, required=True)

    # structured pruning model config
    parser.add_argument("--llm_pruner", action="store_true")
    parser.add_argument("--llm_pruner_home", type=str, default="")
    parser.add_argument("--bonsai", action="store_true")
    parser.add_argument("--bonsai_home", type=str, default="")
    
    # model loading config
    parser.add_argument("--loading_mode", type=str, choices=["nf4", "int8", "fp16"], default="fp16")
    parser.add_argument("--awq", action="store_true")
    parser.add_argument("--gptq", action="store_true")
    parser.add_argument("--flash_attention", action="store_true")

    # dataset config
    parser.add_argument("--dataset", type=str, default="toxigen")
    parser.add_argument("--n_samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)

    # generation config
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--min_new_tokens", type=int, default=30)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--early_stopping", action="store_true")  # this should only be set to True when using beam search of beam sample
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.)
    parser.add_argument("--num_return_sequences", type=int, default=1)  # TODO, do we need multiple generations?

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--disable_progress_bar", action="store_true")

    # evaluation config
    parser.add_argument("--test_lm_perplexity", action="store_true")
    parser.add_argument("--save_results", action="store_true")
    parser.add_argument("--results_dest", type=str, default="./logs/default_results.json")
    parser.add_argument("--save_outputs", action="store_true")
    parser.add_argument("--outputs_dest", type=str)
    
    args = parser.parse_args()

    results_dict_global = OrderedDict()
    for k, v in vars(args).items():
        print(f"{k} -> {v}")
        results_dict_global[k] = v

    # config = AutoConfig.from_pretrained(args.model_name_or_path)  # it seems we do not need this

    dataset = read_dataset(dataset_name=args.dataset)
    random.seed(args.seed)
    random.shuffle(dataset)
    dataset = dataset[:args.n_samples]

    evaluator = Evaluator()

    tokenizer, model = configure_tokenizer_model_loading(args)

    generation_config = GenerationConfig(
        min_new_tokens=args.min_new_tokens,
        max_new_tokens=args.max_new_tokens, 
        early_stopping=args.early_stopping,
        do_sample=args.do_sample,
        num_beams=args.num_beams,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        num_return_sequences=args.num_return_sequences,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        output_logits=False,  # this config needs to be specified in transformers==4.38
    )
    
    if args.test_lm_perplexity:
        lm_perplexity = evaluator.evaluate_lm_perplexity(tokenizer, model)
        print(f"lm perplexity -> {lm_perplexity}")
        results_dict_global["lm_perplexity"] = lm_perplexity

    if args.dataset in ["toxigen", "realtoxicityprompts", "AdvPromptSet"]:
        results_dict_global["prompt_perplexity"] = evaluator.evaluate_prompt_perplexity(dataset=dataset, model=model, tokenizer=tokenizer, grouping=True, disable_progress_bar=args.disable_progress_bar)

    generator = Generator(tokenizer=tokenizer, model=model)
    
    generation = generator.generate(dataset=dataset, generation_config=generation_config, args=args)
    # print(f"total num of generation -> {len(generation)}")
    res, results_dict = evaluator.evaluate_generation(dataset_name=args.dataset, dataset=generation, metrics=None)
    for k, v in results_dict.items():
        results_dict_global[k] = v

    if args.save_results:
        save_results(results=results_dict_global, dest=args.results_dest)

    if args.save_outputs:
        save_outputs(dest=args.outputs_dest, data=res)
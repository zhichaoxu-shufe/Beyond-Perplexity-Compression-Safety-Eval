import os
import sys
import time
import random
import json
import argparse

from tqdm import tqdm
from collections import OrderedDict

import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import AutoConfig
from transformers import GenerationConfig

from evaluate import Evaluator
from datasets import load_dataset
from utils import save_outputs, save_results

def configure_model_loading(args):
    # TODO: what is a better way to configure model loading
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
    model = configure_model_loading(args)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.padding_side = "left"
    print(f"finish loading model and tokenizer!")

    device = "cuda" if torch.cuda.is_available() else "cpu"  # this needs to be changed if we are doing 30B model
    
    if args.loading_mode == "nf4" or args.loading_mode == "int8":
        return tokenizer, model
    else:
        return tokenizer, model.to(device)

PALOMA_DICT = {
    "aave_literature": "/uufs/chpc.utah.edu/common/home/u1320595/toxicity_eval/raw_datasets/aave_corpora/aave_literature.jsonl.gz",
    "books-domain": "/uufs/chpc.utah.edu/common/home/u1320595/paloma/dolma-v1_5/test/test_books.jsonl.gz",
    "commoncrawl-domain": "/uufs/chpc.utah.edu/common/home/u1320595/paloma/dolma-v1_5/test/test_common-crawl.jsonl.gz",
    "reddit-domain": "/uufs/chpc.utah.edu/common/home/u1320595/paloma/dolma-v1_5/test/test_reddit_uniform.jsonl.gz",
    "stack-domain": "/uufs/chpc.utah.edu/common/home/u1320595/paloma/dolma-v1_5/test/test_stack_uniform.jsonl.gz",
    "wiki-domain": "/uufs/chpc.utah.edu/common/home/u1320595/paloma/dolma-v1_5/test/test_wiki.jsonl.gz",
    "pes2o-domain": "/uufs/chpc.utah.edu/common/home/u1320595/paloma/dolma-v1_5/test/test_pes2o.jsonl.gz",
    "twitterAAE": "/uufs/chpc.utah.edu/common/home/u1320595/paloma/twitterAAE_HELM_fixed/test/test_AA.jsonl.gz",
    "twitterAAE-white": "/uufs/chpc.utah.edu/common/home/u1320595/paloma/twitterAAE_HELM_fixed/test/test_white.jsonl.gz",
    "reddit-atheism": "/uufs/chpc.utah.edu/common/home/u1320595/paloma/dolma_100_subreddits/test/test_22_atheism.jsonl.gz",
    "reddit-christianity": "/uufs/chpc.utah.edu/common/home/u1320595/paloma/dolma_100_subreddits/test/test_35_Christianity.jsonl.gz",
    "reddit-catholicism": "/uufs/chpc.utah.edu/common/home/u1320595/paloma/dolma_100_subreddits/test/test_92_Catholicism.jsonl.gz",
    "reddit-askmen": "/uufs/chpc.utah.edu/common/home/u1320595/paloma/dolma_100_subreddits/test/test_23_AskMen.jsonl.gz",
    "reddit-askwomen": "/uufs/chpc.utah.edu/common/home/u1320595/paloma/dolma_100_subreddits/test/test_37_AskWomen.jsonl.gz",
    "reddit-asktransgender": "/uufs/chpc.utah.edu/common/home/u1320595/paloma/dolma_100_subreddits/test/test_62_asktransgender.jsonl.gz",
    "reddit-europe": "/uufs/chpc.utah.edu/common/home/u1320595/paloma/dolma_100_subreddits/test/test_33_europe.jsonl.gz",
    "reddit-canada": "/uufs/chpc.utah.edu/common/home/u1320595/paloma/dolma_100_subreddits/test/test_34_canada.jsonl.gz",
    "reddit-uk": "/uufs/chpc.utah.edu/common/home/u1320595/paloma/dolma_100_subreddits/test/test_47_unitedkingdom.jsonl.gz",
    "reddit-australia": "/uufs/chpc.utah.edu/common/home/u1320595/paloma/dolma_100_subreddits/test/test_55_australia.jsonl.gz",
    "reddit-india": "/uufs/chpc.utah.edu/common/home/u1320595/paloma/dolma_100_subreddits/test/test_88_india.jsonl.gz",
    "reddit-politics": "/uufs/chpc.utah.edu/common/home/u1320595/paloma/dolma_100_subreddits/test/test_01_politics.jsonl.gz",
    "reddit-neoliberal": "/uufs/chpc.utah.edu/common/home/u1320595/paloma/dolma_100_subreddits/test/test_64_neoliberal.jsonl.gz",
    "reddit-politicaldiscussion": "/uufs/chpc.utah.edu/common/home/u1320595/paloma/dolma_100_subreddits/test/test_69_PoliticalDiscussion.jsonl.gz",
    "reddit-libertarian": "/uufs/chpc.utah.edu/common/home/u1320595/paloma/dolma_100_subreddits/test/test_70_Libertarian.jsonl.gz",
    "reddit-conservative": "/uufs/chpc.utah.edu/common/home/u1320595/paloma/dolma_100_subreddits/test/test_96_Conservative.jsonl.gz",
}

PERPLEXITY_MAPPING = {
    "dolma": ["books-domain", "commoncrawl-domain", "reddit-domain", "stack-domain", "wiki-domain", "pes2o-domain"],
    "twitter": ["twitterAAE", "twitterAAE-white"],
    "aave_corpora": ["aave_literature"],
    "reddit-religion": ["reddit-atheism", "reddit-christianity", "reddit-catholicism"],
    "reddit-gender": ["reddit-askmen", "reddit-askwomen", "reddit-asktransgender"],
    "reddit-country": ["reddit-europe", "reddit-canada", "reddit-uk", "reddit-australia", "reddit-india"],
    "reddit-political-ideology": ["reddit-politics", "reddit-neoliberal", "reddit-politicaldiscussion", "reddit-libertarian", "reddit-conservative"],
    "sft_eval": ["wikitext2", "books-domain", "commoncrawl-domain", "reddit-domain", "stack-domain", "wiki-domain", "pes2o-domain", "twitterAAE", "twitterAAE-white", "aave_literature"]
}

def load_paloma_dataset(fname):
    import gzip
    fin=gzip.open(fname, "r")
    json_list = [json.loads(line) for line in fin]
    return json_list


def configure_perplexity_dataset(dataset, tokenizer):
    if dataset == "wikitext2":
        test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(test["text"])
        input_pretokenized = tokenizer(text, return_tensors="pt")
    elif dataset == "wikitext103":
        test = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
        text = "\n\n".join(test["text"])
        input_pretokenized = tokenizer(text, return_tensors="pt")
    elif dataset in PALOMA_DICT:
        test = load_paloma_dataset(PALOMA_DICT[dataset])
        text = [i["text"] for i in test]
        text = "\n\n".join(text)
        input_pretokenized = tokenizer(text, return_tensors="pt")
    return input_pretokenized


def evaluate_lm_perplexity(model, dataset, input_pretokenized, stride=1024, disable_tqdm=True):
    # according to https://huggingface.co/docs/transformers/perplexity
    # TODO: test whether this fit the awq and gptq library, if not, we switch to manually configure the negative log likelihood
    
    seq_len = input_pretokenized.input_ids.shape[1]  # (1, seq_len)

    DEVICE = model.device  # TODO: do we need to enable pipeline parallel for larger models?

    nlls = []
    prev_end_loc = 0
    start_time = time.time()
    # loss_fct = torch.nn.CrossEntropyLoss()
    for begin_loc in tqdm(range(0, seq_len, stride), disable=disable_tqdm):
        end_loc = min(begin_loc+stride, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = input_pretokenized.input_ids[:, begin_loc: end_loc]  # (1, stride)
        
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        with torch.no_grad():
            outputs = model(input_ids.to(DEVICE), labels=target_ids.to(DEVICE), return_dict=True)
            # lm_logits = outputs.logits
            # shift_logits = lm_logits[..., :-1, :].contiguous()
            # shift_labels = target_ids[..., 1:].contiguous().to(shift_logits.device)
            # nll = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            nll = outputs.loss
        nlls.append(nll)
        # print(f"sequence shape -> {input_ids.shape} nll -> {nll:.2f}")

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
    ppl = torch.exp(torch.stack(nlls).mean()).item()
    print(f"lm perplexity evaluation takes {time.time()-start_time:.1f} seconds")
    print(f"perplexity on {dataset} dataset -> {ppl:.2f}")
    return {
        f"perplexity_{dataset}": ppl
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--tokenizer", type=str)
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--config", type=str)

    # structured pruning config
    parser.add_argument("--llm_pruner", action="store_true")
    parser.add_argument("--llm_pruner_home", type=str, default="")
    parser.add_argument("--bonsai", action="store_true")
    parser.add_argument("--bonsai_home", type=str, default="")

    # model loading config
    parser.add_argument("--loading_mode", type=str, choices=["nf4", "int8", "fp16"])
    parser.add_argument("--flash_attention", action="store_true")
    parser.add_argument("--awq", action="store_true")
    parser.add_argument("--gptq", action="store_true")

    # dataset config
    parser.add_argument("--dataset_group", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None, help="separate with comma")
    parser.add_argument("--max_seq_length", type=int, default=2048)

    parser.add_argument("--disable_tqdm", action="store_true")

    parser.add_argument("--save_results", action="store_true")
    parser.add_argument("--results_dest", action="store_true")

    args = parser.parse_args()

    tokenizer, model = configure_tokenizer_model_loading(args)
    print(f"model -> {args.model_name_or_path}")

    if args.dataset is not None:
        datasets = args.dataset.split(",")
    elif args.dataset_group is not None:
        if args.dataset_group == "all":
            datasets = list(PALOMA_DICT.keys())
            datasets.append("wikitext2")
        else:
            datasets = PERPLEXITY_MAPPING[args.dataset_group]
    else:
        raise Exception("need to specify either args.dataset or args.dataset_group")

    for dataset in datasets:
        input_pretokenized = configure_perplexity_dataset(dataset, tokenizer)
        res = evaluate_lm_perplexity(
            model=model, 
            dataset=dataset, 
            input_pretokenized=input_pretokenized, 
            stride=args.max_seq_length,
            disable_tqdm=args.disable_tqdm
            )
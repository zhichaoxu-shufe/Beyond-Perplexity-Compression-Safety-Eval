import os
import sys
import time
import json
from copy import deepcopy
from datetime import datetime

import torch

def print_total_params(model):
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f"total number of model parameters -> {pytorch_total_params/1024**3:.1f}B")

def count_nonzero_weights(model):
    zeros = 0
    for param in model.parameters():
        zeros += param.numel()-param.nonzero().size(0)
    print(f"total non zero parameters -> {(model.num_parameters()-zeros)/1024**3:.1f}B")

def get_datetime():
    now = datetime.now()
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
    return date_time

def read_jsonl(fname):
    with open(fname, "r") as fin:
        json_list = list(fin)
        json_list = [json.loads(i) for i in json_list]
        return json_list

def save_outputs(data, dest="./default_outputs.jsonl"):
    if "jsonl" in dest:
        with open(dest, "w") as fout:
            for i, row in enumerate(data):
                json.dump(row, fout)
                fout.write("\n")
        fout.close()
    elif "json" in dest:
        with open(dest, "w") as fout:
            json.dump(data, fout, indent=4)
        fout.close()
    else:
        raise Exception("unsupported save format")

def save_results(results, dest="./logs/default_results.json"):
    if os.path.isfile(dest):
        with open(dest, "r") as fin:
            fin = json.load(fin)
    
        now = datetime.now()
        date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
        fin[date_time] = results

        with open(dest, "w") as fout:
            json.dump(fin, fout, indent=4)
        fout.close()
    
    else:
        now = datetime.now()
        date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
        fin = {}
        fin[date_time] = results
        with open(dest, "w") as fout:
            json.dump(fin, fout, indent=4)
        fout.close()

def get_datetime():
    now = datetime.now()
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
    return date_time

def print_gpu_usage():
    available, total = torch.cuda.mem_get_info()
    print(f"total memory -> {total/1024**3:.1f}GB")
    print(f"availble memory -> {available/1024**3:.1f}GB")


def template_tulu(input_pretokenized):
    return f"<|user|>\n{input_pretokenized}\n<|assistant|>\n"


def template_input(input_pretokenized, model_name_or_path="llama"):
    # currently hardcoded

    if "tulu" in model_name_or_path:
        return template_tulu(input_pretokenized)
    else:
        return input_pretokenized
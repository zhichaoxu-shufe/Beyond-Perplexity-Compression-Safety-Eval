import os
import sys
import time
import random
import argparse
from tqdm import tqdm

import torch

from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import AutoConfig
from transformers import GenerationConfig

from utils import template_input

class Generator:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token, self.tokenizer.pad_token_id = self.tokenizer.eos_token, self.tokenizer.eos_token_id
        self.model = model

    def generate(self, dataset, generation_config, args):
        print(f"total number of instances -> {len(dataset)}")

        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        if args.awq or args.gptq:
            self.dispatch_model()        
        elif self.model.device == "cpu":  # TODO: enable pipeline parallel
            self.dispatch_model()

        """
        truncate_output = True if args.dataset in ["toxigen", "realtoxicityprompts", "AdvPromptSet"] else False
        # notes on Jan 16th, according to the robbie paper, the toxicity score are determined purely based on continuations
        """
        truncate_output = True
        
        res = []
        num_batch = len(dataset)//args.batch_size if len(dataset)%args.batch_size==0 else len(dataset)//args.batch_size+1
        for i in tqdm(range(num_batch), total=num_batch, disable=args.disable_progress_bar):
            batch = dataset[i*args.batch_size: (i+1)*args.batch_size]
            batch_input = [template_input(row["input_pretokenized"], self.model.config._name_or_path) for row in batch]
            tokenized_batch_input = self.tokenizer(batch_input, return_tensors="pt", padding=True, truncation=True,)
            with torch.no_grad():
                output = self.model.generate(
                    input_ids=tokenized_batch_input.input_ids.to(DEVICE),
                    attention_mask=tokenized_batch_input.attention_mask.to(DEVICE),
                    generation_config=generation_config
                )  # (bz, seq_len)
            if truncate_output:
                batch_output = self.tokenizer.batch_decode(output[:, tokenized_batch_input.input_ids.shape[1]:], skip_special_tokens=True)
            else:
                batch_output = self.tokenizer.batch_decode(output, skip_special_tokens=True)
            for j in range(len(batch)):
                row = batch[j]
                row["target_pretokenized"] = batch_output[j]
                res.append(row)

        # self.offload_model()  # disable to be comptible with 8-bit model
            
        return res
    
    def offload_model(self):
        # TODO: think about how to load model back to multi gpus for pipeline parallel
        self.model = self.model.to("cpu")

    def dispatch_model(self):
        # TODO: enable pipeline parallel

        """
        from accelerate import infer_auto_device_map
        from accelerate import dispatch_model
        device_map = infer_auto_device_map(self.model)
        dispatch_model(self.model, device_map=device_map)
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(device)
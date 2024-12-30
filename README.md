# Beyond-Perplexity-Multi-dimensional-Safety-Evaluation-of-LLM-Compression
This is the official repo for paper **Beyond Perplexity: Multi-dimensional Safety Evaluation of LLM Compression**, [link to ACL Anthology]([https://arxiv.org/abs/2407.04965](https://aclanthology.org/2024.findings-emnlp.901/#)). 

We evaluate 4 unstructured pruning methods: Magnitude, SparseGPT, Wanda and GBLM, in addition to 3 popular quantization methods: LLM.int8(), AWQ and GPTQ. Our evaluation focus is on safety (degeneration harm, representational bias and dialect bias) of compression methods.

## Full Results
Please refer to `./full_results/` for csv files that contain full evaluation results.

## Datasets and Code Implementation

### Datasets
All datasets we used are available at [this google drive link](https://drive.google.com/drive/folders/1St_jmebZjQOMJXOpZ2dVmtqH6aKcmsKo?usp=sharing). For each individual datasets, refer to [Robbie collection](https://github.com/facebookresearch/ResponsibleNLP/tree/main/robbie), [BBQ dataset](https://github.com/nyu-mll/BBQ), [UnQover dataset](https://github.com/allenai/unqover) to download/preprocessing.

For TruthfulQA and MMLU dataset and evaluation, we use [Tulu 2 implementation](https://github.com/allenai/open-instruct).

### Code Implementation
We provide our source implementation at `./src/`. 

#### Dependency
refer to `requirements.txt`

specifically, for AWQ and GPTQ evaluation, follow these repos ([AutoAWQ](https://github.com/casper-hansen/AutoAWQ) and [AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ)) to install the dependencies, create the quantized models, then load in huggingface style with our code

#### Code Structure
`./src/utils.py` includes misc utils<br>
`./src/dataset.py` includes dataloader for all bias/fairness/toxicity datasets<br>
`./src/evaluate.py` includes evaluation functions<br>
`./src/generation.py` and `./src/multi_qa.py` includes functions for degeneration harm and representational harm, respectively<br>
`./src/run_generation.py` is the main function for degeneration harm<br>
`./src/run_multi_qa.py` is the main function for representational harm<br>
`./src/test_perplexity.py` is for perplexity evaluation

Before running, make sure you have HF access token at `./src/run_generation.py` `configure_model_loading`

#### Sample Scripts
##### Degeneration harm with AdvPromptSet (magnitude pruning)
```
DATASET="AdvPromptSet"
SAVE_AS="tulu2_13b_magnitude"

python src/run_generation.py \
--tokenizer {your tokenizer} \
--model_name_or_path {your local model or model on HF hub} \
--flash_attention \
--dataset AdvPromptSet \
--min_new_tokens 50 \
--max_new_tokens 100 \
--batch_size 16 \
--save_results \
--results_dest ./logs/generative/${DATASET}/${DATASET}_${SAVE_AS}.json \
--disable_progress_bar \
--save_outputs \
--outputs_dest {your outputs dest}.jsonl
```

##### Representational harm with BBQ (bitsandbytes)
```
DATASET=bbq_fewshot
SAVE_AS=llama2_13b_bitsandbytes

python src/run_multi_qa.py \
--tokenizer {your tokenizer} \
--model_name_or_path {your local model or model on HF hub} \
--category age,disability_status,gender_identity,nationality,physical_appearance,race_ethnicity,race_x_gender,race_x_ses,religion,ses,sexual_orientation\
--do_infernece \
--loading_mode int8 \
--disable_progress_bar \
--save_results \
--results_dest ./logs/discriminative/${DATASET}/${DATASET}_${SAVE_AS}.json
```

##### Perplexity with AAVE_literature (AWQ)
```
MODEL_DIR={your model dir}
for MODEL_NAME in "llama2_7b_awq_4bit" "llama2_13b_awq_4bit" "tulu2_7b_awq_4bit" "tulu2_13b_awq_4bit";
do
    python src/test_perplexity.py \
    --tokenizer {your tokenizer} \
    --model_name_or_path {your local model or model on HF hub} \
    --awq \ 
    --dataset aave_literature \
    --disable_tqdm
done;
```



## References
### Our Paper
```
@inproceedings{xu-etal-2024-beyond-perplexity,
    title = "Beyond Perplexity: Multi-dimensional Safety Evaluation of {LLM} Compression",
    author = "Xu, Zhichao  and
      Gupta, Ashim  and
      Li, Tao  and
      Bentham, Oliver  and
      Srikumar, Vivek",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2024",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-emnlp.901/",
    doi = "10.18653/v1/2024.findings-emnlp.901",
    pages = "15359--15396",
    abstract = "Increasingly, model compression techniques enable large language models (LLMs) to be deployed in real-world applications. As a result of this momentum towards local deployment, compressed LLMs will interact with a large population. Prior work on compression typically prioritize preserving perplexity, which is directly analogous to training loss. The impact of compression method on other critical aspects of model behavior---particularly safety---requires systematic assessment. To this end, we investigate the impact of model compression along four dimensions: (1) degeneration harm, i.e., bias and toxicity in generation; (2) representational harm, i.e., biases in discriminative tasks; (3) dialect bias; and (4) language modeling and downstream task performance. We examine a wide spectrum of LLM compression techniques, including unstructured pruning, semi-structured pruning, and quantization. Our analysis reveals that compression can lead to unexpected consequences. Although compression may unintentionally alleviate LLMs' degeneration harm, it can still exacerbate representational harm. Furthermore, increasing compression produces a divergent impact on different protected groups. Finally, different compression methods have drastically different safety impacts: for example, quantization mostly preserves bias while pruning degrades quickly. Our findings underscore the importance of integrating safety assessments into the development of compressed LLMs to ensure their reliability across real-world applications."
}
```

### Compression Methods
```
@inproceedings{frantar2023sparsegpt,
  title={Sparsegpt: Massive language models can be accurately pruned in one-shot},
  author={Frantar, Elias and Alistarh, Dan},
  booktitle={International Conference on Machine Learning},
  pages={10323--10337},
  year={2023},
  organization={PMLR}
}
@inproceedings{
  sun2024wanda,
  title={A Simple and Effective Pruning Approach for Large Language Models},
  author={Mingjie Sun and Zhuang Liu and Anna Bair and J Zico Kolter},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024},
  url={https://openreview.net/forum?id=PxoFut3dWW}
}
@article{das2023gblm,
  title={Beyond size: How gradients shape pruning decisions in large language models},
  author={Das, Rocktim Jyoti and Ma, Liqun and Shen, Zhiqiang},
  journal={arXiv preprint arXiv:2311.04902},
  year={2023}
}
@article{dettmers2022gpt3,
  title={Gpt3. int8 (): 8-bit matrix multiplication for transformers at scale},
  author={Dettmers, Tim and Lewis, Mike and Belkada, Younes and Zettlemoyer, Luke},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  pages={30318--30332},
  year={2022}
}
@article{frantar2022gptq,
  title={Gptq: Accurate post-training quantization for generative pre-trained transformers},
  author={Frantar, Elias and Ashkboos, Saleh and Hoefler, Torsten and Alistarh, Dan},
  journal={arXiv preprint arXiv:2210.17323},
  year={2022}
}

@inproceedings{lin2023awq,
  title={AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration},
  author={Lin, Ji and Tang, Jiaming and Tang, Haotian and Yang, Shang and Chen, Wei-Ming and Wang, Wei-Chen and Xiao, Guangxuan and Dang, Xingyu and Gan, Chuang and Han, Song},
  booktitle={MLSys},
  year={2024}
}
```

# Beyond-Perplexity-Multi-dimensional-Safety-Evaluation-of-LLM-Compression
This is the official repo for paper **Beyond Perplexity: Multi-dimensional Safety Evaluation of LLM Compression**, [link to arXiv paper](https://arxiv.org/abs/2407.04965). 

We evaluate 4 unstructured pruning methods: Magnitude, SparseGPT, Wanda and GBLM, in addition to 3 popular quantization methods: LLM.int8(), AWQ and GPTQ. Our evaluation focus is on safety (degeneration harm, representational bias and dialect bias) of compression methods.

## Full Results
Please refer to `./full_results/` for csv files that contain full evaluation results.

## Datasets and Code Implementation
Our implementation will be released soon.

## References
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

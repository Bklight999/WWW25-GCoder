# GCoder: Improving Large Language Model for Generalized Graph Problem Solving
### Abstract
Large Language Models (LLMs) have demonstrated strong reasoning abilities, making them suitable for complex tasks such as graph computation. Traditional reasoning steps paradigm for graph problems is hindered by unverifiable steps, limited long-term reasoning, and poor generalization to graph variations. To overcome these limitations, we introduce GCoder, a code-based LLM designed to enhance problem-solving in generalized graph computation problems. Our method involves constructing a comprehensive training dataset, GraphWild, featuring diverse graph formats and algorithms. We employ a multi-stage training process, including Supervised Fine-Tuning (SFT) and Reinforcement Learning from Compiler Feedback (RLCF), to refine model capabilities. For unseen tasks, a hybrid retrieval technique is used to augment performance. Experiments demonstrate that GCoder outperforms GPT-4o, with an average accuracy improvement of **16.42\%** across various graph computational problems. Furthermore, GCoder efficiently manages large-scale graphs with millions of nodes and diverse input formats, overcoming the limitations of previous models focused on the reasoning steps paradigm. This advancement paves the way for more intuitive and effective graph problem-solving using LLMs.

<div align="center">
<img src="figures\intro_demo.png" width="800px">
</div>

(a) While the reasoning step paradigm outputs correct results, intermediate reasoning can be wrong (i.e., red reasoning step, node 2 is not connected to 0 and 5). (b) Our code paradigm processes graph problems with programming. More examples can be found in our Appendix B.


### Framework

<div align="center">
<img src="figures\framework.png" width="800px">
</div>

### Main Results

<div align="center">
<img src="figures\main_results.png" width="800px">
</div>


#### Showcases of prompt and LLM generated code
<div style="display: flex; justify-content: space-between;">
  <img src="figures\showcase1.png" alt="Image 1" style="width: 50%;">
  <img src="figures\showcase2.png" alt="Image 2" style="width: 50%;">
</div>


### Environment Setup
```bash
cd Train and Evaluation/LLaMA-Factory
pip install -r requirements.txt
```


### Train
```bash
cd Train and Evaluation/LLaMA-Factory
pip install -e ".[torch,metrics]"
```

#### SFT
```bash
llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml
```

#### RLCF
```bash
llamafactory-cli train examples/train_lora/llama3_lora_dpo.yaml
```


### Evaluation
```bash
cd Train and Evaluation
python vllm_infer_main_table.py
```




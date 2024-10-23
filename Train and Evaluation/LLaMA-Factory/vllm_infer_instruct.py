
import json
import os
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer
import torch
import argparse
from vllm import LLM, SamplingParams

def generate_text_template(problem_text, tokenizer):
    # Initialize the tokenizer
    
    # Pass the default decoding hyperparameters of Qwen2-7B-Instruct
    # max_tokens is for the maximum length for generation.
    # sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=512)

    # Input the model name or path. Can be GPTQ or AWQ models.
    # llm = LLM(model="Qwen/Qwen2-7B-Instruct")

    # Prepare your prompts
    instrcution = 'You are a sophisticated AI expert in graph theory and algorithms. Solve the following graph problem by writing code without any explanation.\n### Instruction:\n' #by writing code without any explanation.
#     cot = """Here is an output example:
    
# The weight of path 5, 2, 1 is the smallest, so the shortest path from node 5 to node 1 is [5, 2, 1] with a total weight of 6. ### 6.
#     """
    temp_instr = instrcution + problem_text
    # temp_instr = instrcution + problem_text + "The output of the code should be Yes or No."
    messages = [{"role": "system", "content": "You are a helpful assistant."}, 
            {"role": "user", "content": temp_instr}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    return text

def read_json(file_name):
    # 存放所有字典的列表
    dict_list = []

    # 打开文件并逐行读取
    with open(file_name, 'r', encoding='utf-8') as file:
        for line in file:
            # 去掉行末的换行符（如果有）
            line = line.strip()
            if line:
                # 将每行的字符串解析为字典，并添加到列表中
                dict_list.append(json.loads(line))
    return dict_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='******LLM inference on GraphInstruct******')
    parser.add_argument("--llm", type=str, default="llama3.1_8b_v1", help="LLM used for inference")
    parser.add_argument("--temp", type=float, default="0.0", help="Temperature of LLM")
    parser.add_argument("--model_path", type=str, default="path/to/GCoder", help="loading paths of LLM")
    parser.add_argument("--max_token", type=int, default=2048, help="Max output token of LLM")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    parser.add_argument("--problem_num", type=int, default=500, help="Number of problems to evaluate")
    parser.add_argument("--task", type=str, default="MST", help="taks name")
    
    args = parser.parse_args()
    
    # task_name = args.task    #########
    task_name ='clustering_coefficient'
    print('task_name:', task_name)

    # tokenizer, model = get_tokenizer_and_model(args.llm)
    
    # inference
    file_name = '/LLaMA-Factory/Evaluate_dataset/GraphInstruct/'+task_name+'/'+task_name+'_test.json'
    with open(file_name,'r') as f:
        all_data = json.load(f)
    results = 'results/'

    response_dict = {}
    input_texts = []
    # model = 
    sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=4096)
    print('sampleing =====', sampling_params)
    llm = LLM(model=args.model_path, tensor_parallel_size=1,gpu_memory_utilization=0.85)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
 

    response_dict[task_name] = {'easy':{}, 'hard':{}}
    input_texts = [] 
    args.problem_num = len(all_data)
    
    for i in range(args.problem_num):
        problem_text = all_data[i]['input_prompt']
        template_text = generate_text_template(problem_text, tokenizer=tokenizer)
        input_texts.append(template_text)
    print(f'evaluating: {task_name}, total samples are: {args.problem_num}')
    completions = llm.generate(input_texts, sampling_params)
    
    output_datas = []

    for i, output in enumerate(completions):
        prompt_temp = output.prompt
        completion = output.outputs[0].text
        output_datas.append({
            'id': len(output_datas),
            'code': completion
        })
    
    output_dir = "/LLaMA-Factory/Evaluate_dataset/GraphInstruct/" + task_name +"/"
    out_name = output_dir+ task_name+'_result_qwen.json'
    with open(out_name,'w') as f:
        json.dump(output_datas,f)

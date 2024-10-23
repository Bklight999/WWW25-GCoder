
import json
import os
import re
import io
import sys
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer
import torch
import argparse
from tqdm import tqdm
from vllm import LLM, SamplingParams

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
llms = {
    "llama3.1_8b": "/data/qifanzhang/LLaMA-Factory/models/Graphwiz_plus",
    "llama3.1_8b_v1": "/data/qifanzhang/LLaMA-Factory/models/Graphwiz_plus_v1"
}

def generate_text_template(problem_text, tokenizer):
    # Initialize the tokenizer
    
    # Pass the default decoding hyperparameters of Qwen2-7B-Instruct
    # max_tokens is for the maximum length for generation.
    # sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=512)

    # Input the model name or path. Can be GPTQ or AWQ models.
    # llm = LLM(model="Qwen/Qwen2-7B-Instruct")

    # Prepare your prompts
    instrcution = 'You are a sophisticated AI expert in graph theory and algorithms. Solve the following graph problem by writing code without any explanation. \n### Instruction:\n' #by writing code without any explanation.
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='******LLM inference on Graphwiz******')
    parser.add_argument("--llm", type=str, default="llama3.1_8b_v1", help="LLM used for inference")
    parser.add_argument("--temp", type=float, default="0.0", help="Temperature of LLM")
    #parser.add_argument("--model_path", type=str, default="GraphWiz/LLaMA2-7B", help="loading paths of LLM")
    parser.add_argument("--model_path", type=str, default="/data/qifanzhang/LLaMA-Factory/models/Graphwiz-plus-v1-qwen", help="loading paths of LLM")
    #parser.add_argument("--model_path", type=str, default="/data/qifanzhang/LLaMA-Factory/models/Graphwiz_plus_v1", help="loading paths of LLM")
    #parser.add_argument("--model_path", type=str, default="/data/qifanzhang/LLaMA-Factory/models/Graphwiz-plus-v1-qwen-DPO", help="loading paths of LLM")
    #/data/qifanzhang/LLaMA-Factory/models/Graphwiz-plus-v1-qwen-DPO
    #parser.add_argument("--model_path", type=str, default="/data/qifanzhang/LLaMA-Factory/models/Graphwiz_plus_v1", help="loading paths of LLM")
    #parser.add_argument("--model_path", type=str, default="/data/qifanzhang/LLaMA-Factory/models/Graphwiz_plus_weak", help="loading paths of LLM")
    #parser.add_argument("--model_path", type=str, default="/data/qifanzhang/LLaMA-Factory/models/Qwen/Qwen2.5-Coder-7B-Instruct", help="loading paths of LLM")
    parser.add_argument("--max_token", type=int, default=2048, help="Max output token of LLM")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    parser.add_argument("--problem_num", type=int, default=500, help="Number of problems to evaluate")
    parser.add_argument("--task", type=str, default="MST", help="taks name")
    
    args = parser.parse_args()
    
    # task_name = args.task    #########
    task_name ='clustering'
    print('task_name:', task_name)

    file_name = './clustering_data.json'
    with open(file_name,'r') as f:
        all_data = json.load(f)

    response_dict = {}
    input_texts = []
    # model = 
    sampling_params = SamplingParams(temperature=0.1, top_p=1, max_tokens=4096)
    print('sampleing =====', sampling_params)
    llm = LLM(model=args.model_path, tensor_parallel_size=1,gpu_memory_utilization=0.85)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
 

    response_dict[task_name] = {'easy':{}, 'hard':{}}

    args.problem_num = len(all_data)
    print(f'evaluating: {task_name}, total samples are: {args.problem_num}')
    dpo_datas = []
    for i in tqdm(range(0, args.problem_num), desc=f"Processed number"): ###verify
        iter_data = dict()
        iter_data['conversations'] =[
            {
                "from": "system",
                "value": "You are a sophisticated AI expert in graph theory and algorithms. Solve the following graph problem by writing code without any explanation.\n### Instruction:\n"
            },
            {
                "from": "human",
                "value": all_data[i]['problem']
            }
        ]

        input_texts = [] 
        problem_text = all_data[i]['problem']
        template_text = generate_text_template(problem_text, tokenizer=tokenizer)

        for _ in range(100):
            input_texts.append(template_text)

        completions = llm.generate(input_texts, sampling_params)
        positive = False
        negative = False
        ans = float(all_data[i]['ans'])
        print(type(ans),ans)

        for completion in completions:
            try:
                if positive == True and negative == True:
                    break
                prompt_temp = completion.prompt
                code = completion.outputs[0].text
                code = re.sub(r'^```python\n|```$', '', code, flags=re.MULTILINE)

                # 创建一个 StringIO 对象来捕获输出
                output = io.StringIO()
                # 重定向 sys.stdout
                old_stdout = sys.stdout
                sys.stdout = output

                try:
                    exec(code, globals(), globals())
                except:
                    sys.stdout = old_stdout
                    if negative == False:
                        negative = True
                        iter_data['rejected'] = {
                        "from": "gpt",
                        "value": code
                        }
                    continue
                finally:
                    sys.stdout = old_stdout
                
                
                # 获取输出内容
                output_content = output.getvalue()
                output_content = str(output_content).strip()

                if ':' in output_content:
                    output_content = output_content.split(":")[1].strip()

                output_content = float(output_content)
                print(round(output_content, 3), round(ans, 3))
                if round(output_content, 3) == round(ans, 3):

                    if positive == False:
                        positive = True
                        iter_data['chosen'] = {
                            "from": "gpt",
                            "value": code
                        }
                else:
                    if negative == False:
                        negative = True
                        iter_data['rejected'] = {
                        "from": "gpt",
                        "value": code
                        }
            except:
                continue


        print(positive,negative,len(dpo_datas))

        if len(dpo_datas) == 1000:
            break


        if negative == True and positive == True:
            dpo_datas.append(iter_data)
             
    print(len(dpo_datas))
  
    with open('./dpo_data_qwen_novel/dpo_datas_clustering.json','w') as f:
        json.dump(dpo_datas,f)

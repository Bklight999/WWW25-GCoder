
import json
import os
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer
import torch
from langchain_community.document_loaders.csv_loader import CSVLoader
import bs4
import argparse
from vllm import LLM, SamplingParams
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import re

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def generate_text_template(problem_text, tokenizer, example=""):

    # Prepare your prompts
    instrcution = 'You are a sophisticated AI expert in graph theory and algorithms. Solve the following graph problem by writing code without any explanation. \n'
    example = "Here is an example:\n" + example

    temp_instr = instrcution + problem_text + example
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
    parser = argparse.ArgumentParser(description='******LLM inference on Graphwiz******')
    #parser.add_argument("--llm", type=str, default="llama3.1_8b_v1", help="LLM used for inference")
    parser.add_argument("--temp", type=float, default="0.0", help="Temperature of LLM")
    parser.add_argument("--model_path", type=str, default="/path/to/GCoder", help="loading paths of LLM")
    parser.add_argument("--max_token", type=int, default=2048, help="Max output token of LLM")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    parser.add_argument("--problem_num", type=int, default=500, help="Number of problems to evaluate")
    parser.add_argument("--task", type=str, default="MST", help="taks name")
    parser.add_argument("--gpu_memory_utilization", type=str, default=0.85, help="taks name")
    
    args = parser.parse_args()
    
    task_name ='MIS'
    print('task_name:', task_name)

    # tokenizer, model = get_tokenizer_and_model(args.llm)
    file_name = './MIS/MIS_test.json'
    all_data = read_json(file_name)
    results = 'results/'

    response_dict = {}
    input_texts = []
    # model = 
    sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=4096)
    print('sampleing =====', sampling_params)
    llm = LLM(model=args.model_path, tensor_parallel_size=1, gpu_memory_utilization=args.gpu_memory_utilization)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
 

    response_dict[task_name] = {'easy':{}, 'hard':{}}
    input_texts = [] 
    # args.problem_num = len(all_data)
    args.problem_num = 400
    
    
    print("RAG deployment\n")
    ###RAG settings

 
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    vectorstore = Chroma(
        embedding_function=HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5"),
        persist_directory="./vector_DB" ##vector DB can be downloaded from huggingface
    )
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 100})
    ###data package
    for i in range(args.problem_num):
        problem_text = all_data[i]['input_prompt']
        retrieved_docs = retriever.invoke(problem_text)
        task_list=[]
        for doc in retrieved_docs:
            task_list.append(doc.page_content.split('Code')[0].strip())
        # print(len(task_list))
        prompt = problem_text
        id = ""
        for task in task_list:
            if task.split('Task_name:')[1].strip().lower() in prompt.lower():
                id = task
                break
       # print(id)
        example = ""
        for doc in retrieved_docs:
            if id in doc.page_content.split('Code:')[0]:
                example = doc.page_content.split('graph_cot')[0].split('Code:')[1].strip()
                break
        if i==1:
            print(example)
        template_text = generate_text_template(problem_text, tokenizer=tokenizer,example=example)
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
    output_dir = "."
    out_name = output_dir+ task_name+'_result.json'
    with open(out_name,'w') as f:
        json.dump(output_datas,f)
        
    print(f"Datas have been saved to {output_dir}.")

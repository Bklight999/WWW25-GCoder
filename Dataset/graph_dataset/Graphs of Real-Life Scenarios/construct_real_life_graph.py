from openai import OpenAI
from collections import defaultdict
import pickle
import argparse
import json
import datetime
from time import sleep
import os

llm_to_api = {
    "gpt4": "gpt-4o",
    "mini" : "gpt-4o-mini",
    "gpt": "gpt-3.5-turbo-0125", 
    'claude': "claude-3-haiku-20240307",
    "mixtral": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    # "mixtral": "mixtral-8x7b-32768",
    "deepseek": "deepseek-chat",
    # "llama": "llama3-70b-8192",
    "llama8b": "meta-llama/Llama-3-8b-chat-hf",
    "llama": "meta-llama/Llama-3-70b-chat-hf",
    "qwen7b": "qwen1.5-7b-chat",
    "qwen": "qwen1.5-72b-chat",
    "gemini": "gemini-1.5-pro",
    "gemma": "gemma-7b-it",
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm', type=str, default='gpt', help='llm model name')
    parser.add_argument('--task', type=str, default='TSP', help='task name')
    parser.add_argument('--problem_num', type=int, default=10, help='number of problems')
    parser.add_argument('--example_num', type=int, default=2, help='number of examples')
    parser.add_argument('--difficulty', type=str, default='easy', help='problem difficulty')
    parser.add_argument('--resume', type=bool, default=False, help='resume from last checkpoint')
    parser.add_argument('--results', type=str, default='tmp', help='results location')
    parser.add_argument('--sleep', type=int, default=5, help='sleep seconds between API calls')

    args = parser.parse_args()
    error_knt = 0
    
    response_dict = defaultdict(dict)
    
    for llm in args.llm.split('-'):
        if 'gpt' in llm or 'mini' in llm:
            client = OpenAI(
                base_url = "https://35.aigcbest.top/v1",
                #api_key = 'sk-JMe1osu1CZFTcEx942Cf36A188Cc49D6Ab684689F05e3fE7',
                api_key = 'sk-4PhY006ulJWLVjUf19B39a8f9bB34aBdA7B0C21e96B38873' 
            )
        elif llm == 'deepseek':
            client = OpenAI(
                base_url = "https://api.deepseek.com",
                api_key = 'sk-6eada9420c23459b8aa28ceabfa4f9e6'
            )
        elif 'llama' in llm or 'mixtral' in llm:
            client = OpenAI(
                base_url = "https://api.aimlapi.com/",
                api_key = '23577ec496f14b7ebf00767d7ea0cd3a'
            )
        elif 'qwen' in llm:
            client = OpenAI(
                api_key="sk-95cef5180dec49ac8dc40936fa3b3548",
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
        else:
            client = OpenAI(
                base_url = "https://aigcbest.top/v1",
                api_key = 'sk-995MSFbYANjPjjvI9d1d331eD3Ac4d55B4BeBe1b400052Ae'
            )

        if args.resume and os.path.exists(f"results/tmp_{args.results}/{args.llm}.json"):
            with open(f"results/tmp_{args.results}/{args.llm}.json", 'r') as f:
                response_dict = json.load(f)
                print(f"Continue")

        if not os.path.exists(f"results/tmp_{args.results}"):
            os.makedirs(f"results/tmp_{args.results}")
            
        with open("random_graphs_undirected.json","r") as f:
            datas = json.load(f)
        
        fields = ["Computer Science","Data Science","Bioinformatics","Finance","Logistics and Transportation","Chemistry and Materials Science","Physics"]
        for i, data in enumerate(datas):
            print("_____________________________________")
            field = fields[(i+1) % len(fields)]
            system_prompt = f"""
            Convert the following undirected graph, given as an edge list, into two forms: a natural language description and a real-name edge list. Ensure each vertex is represented by a unique and realistic name.
            Output Requirements:
                1.Natural Language Description: Provide a description of the graph using realistic names for each vertex within the domain of {field}. For example:
                "Alice is connected to Bob."
                "Bob is connected to David."
                2.Real-Name Edge List: Present the edge list using the assigned names. For example:
                [Alice,Bob], [Bob,David]
                3.Chain of Thought:
                Explain how the natural language description can be converted back into the edge list.
            """
            #system_prompt = f"You are a sophisticated AI. Convert the following graph, given as an edge list representing an undirected graph, into natural language descriptions within the domain of {field}. Ensure each vertex is represented by a diverse and realistic name, using as many varied and unique names as possible. Provide the output in two forms: a natural language description and an edge list containing real-name nodes."
            i = str(i)
            if args.resume and i in response_dict and llm in response_dict[i] and response_dict[i][llm]:
                if response_dict[i][llm] != 'Error!':
                    print(i)
                    continue
            response_dict[i] = {}
            try:
                chat_completion = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": str(data["edges"])},
                    ],
                    model=llm_to_api[llm],
                    seed=42,
                    temperature=0.1
                )
                response_dict[i][llm] = chat_completion.choices[0].message.content
                print(llm, i, response_dict[i][llm])
            except Exception as e:
                print('Call API failed! ', e)
                sleep(1)
                error_knt += 1
                response_dict[i][llm] = 'Error!'
            with open(f"results/tmp_{args.results}/{args.llm}.json", 'w') as f:
                json.dump(response_dict, f)
            sleep(args.sleep)
        
    print('error_knt:', error_knt)
    now = datetime.datetime.now()
    if not os.path.exists(f"results/{args.results}"):
        os.makedirs(f"results/{args.results}")
    with open(f"results/{args.results}/{args.llm}_{args.task}_{args.difficulty}_{now.strftime('%d_%H-%M')}.json", 'w') as f:
        json.dump(response_dict, f)
        
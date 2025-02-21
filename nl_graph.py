from pydantic import BaseModel
from openai import OpenAI
import argparse
from tqdm import tqdm
import os
import json



def load_data(file_path:str):
    #首先判断文件类型是json还是pickle
    if file_path.endswith('.json'):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    elif file_path.endswith('.pkl'):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")
client = OpenAI(
    base_url="",
    api_key=""
)

#使用命令行接收参数：file_path,model_name

#file_path = "dataset/raw/QNLI/QNLI.json"
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str)
parser.add_argument("--file_path", type=str)
#parser.add_argument("--model_name", type=str)
args = parser.parse_args()

file_path = args.file_path
#model_name = args.model_name

data = load_data(file_path)
new_data = []
for data_i in tqdm(data):

    #为了防止response_content中返回到不是JSON格式，这里应该用try-except, 出现错误后重新尝试，最大尝试次数为五次
    max_try = 5
    attemps = 0
    while attemps < max_try:
        try:
            completion = client.beta.chat.completions.parse(
                model="Qwen/Qwen2.5-7B-Instruct",
                messages=[
                    {"role": "system", "content": '''
                     Yor task is refine the  SPO triples given into natural language.Just recover from triples! don't write anything more!
                     Do not respond with any text other than the JSON data, and the JSON data should be valid JSON format.
                     speak english in your response!!!
                     your response should be in the following format:
                     {
                        "res":{
                            "Here is the refine sentences".
                        }
                     }, the key MUST named "res", and the value MUST be a string!!!
                     '''},
                    {"role": "user", "content": '''
        "triples": [
            [
                "Vestals",
                "cared_for",
                "Lares"
            ],
            [
                "Vestals",
                "cared_for",
                "Penates"
            ],
            [
                "Lares",
                "equivalent_of",
                "those enshrined in each home"
            ],
            [
                "Penates",
                "equivalent_of",
                "those enshrined in each home"
            ],
            [
                "Lares",
                "type_of",
                "Historical Artifacts"
            ],
            [
                "Penates",
                "type_of",
                "Historical Artifacts"
            ],
            [
                "state",
                "type_of",
                "State Entity"
            ]
        ]
'''},
                    {"role": "assistant", "content": '''
{
"res":"The Vestals cared for the Lares.The Vestals cared for the Penates.The Lares are equivalent to those enshrined in each home.The Penates are equivalent to those enshrined in each home.The Lares are a type of Historical Artifacts.The Penates are a type of Historical Artifacts.The state is a type of State Entity."
}
                    '''},
                {"role": "user", "content": f'''{{"triples":"{str(data_i['sematic_graph'])}}}'''},
                ],
                response_format={ "type": "json_object" },
            )
            response_content = eval(completion.choices[0].message.content)
            #print(response_content)
            data_i['nl_graph'] = response_content['res']
            new_data.append(data_i)
            break
        except Exception as e:
            print(completion.choices[0].message.content)
            print(e)
            attemps = attemps + 1
#将data_i写入到json文件中
with open(f'{args.dataset_name}_with_nl.json', 'w') as f:
    json.dump(new_data, f)

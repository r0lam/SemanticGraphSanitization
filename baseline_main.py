import os
import json
import argparse
from tqdm import tqdm
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from utils import utils
import pickle
from config import *
from pathlib import Path

def load_dataset(dataset_name, dataset_path):
    """
    Load the dataset based on its name and path.
    """
    with open(dataset_path+f'/{dataset_name}.json', 'r') as f:
        data = json.load(f)
    
    return data

#used for icl baseline
def icl_post_parse(text,maps,is_map=False):
    #print(tokenizer)
    reversed_map = {}
    if is_map == True:
        reversed_map = {v: k for k, v in maps.items()}
    input_ids = tokenizer.encode(text, return_tensors='pt')
    remap_token = []
    for index in range(input_ids.size(1)):
        token_str = tokenizer.decode([input_ids[0, index]])
        if token_str == tokenizer.bos_token:
            continue
        if int(input_ids[0, index]) in reversed_map:
            remap_token.append(int(reversed_map[int(input_ids[0, index])]))
            # print("map--")
        else:
            remap_token.append(int(input_ids[0, index]))
    remap_sentence = tokenizer.decode(remap_token)
    #print("map answer: ", remap_sentence)
    return remap_sentence
def generate_prompt(item, dataset_name):
    """
    任务描述需要根据不同的数据集进行改动，是否加入RDF也需要根据参数进行选择
    隐私保护的方法是固定的，不需要根据任务和数据集而改变。
    step1.给出任务描述
    step2.给出隐私处理规则
     (if) step3.给出增强信息描述
     step
    """
    #task_description+privacy_method+rdf+constrain+io
    input_text = ""
    
    task_description_dict = {
        "QNLI":'''You will be given a pair of sentences, including a question and a sentence. Your task is to determine whether the sentence answers the question. Please respond with "entailment" or "not entailment".''',
        "SAMsum":'''You need to complete a text summarization task, summarizing the content of the given text.''',
        "reClor":'''you should give the answer among A to D.'''
    }
    
    input_text = task_description_dict[dataset_name]
    
    constrain = '''
Note: You must follow the example pattern and return a JSON data containing the 'res' field. Do not respond with any text other than the JSON data!!!!
Note: You must follow the example pattern and return a JSON data containing the 'res' field. Do not respond with any text other than the JSON data!!!!
    '''
    input_text = input_text + constrain
    io_sample= ''''''
    if dataset_name == "QNLI":
        io_sample = '''
here is the sample:
input:
{{
    "text1":here is a sample text1,
    "text2":here is a sample text2,
}}
output:
{{
    "res":"not entailment"
}}

Now parse it:
{{
    "text1":{text1},
    "text2":{text2},
}}
{{
    "res":{{}}
}}
            '''.format(text1=item['text1'],text2=item['text2'])
    elif dataset_name == "SAMsum":
        io_sample = '''
here is the sample:
input:
{{
    "text":here is a sample text1,
}}
output:
{{
    "res":"this the summary of the input text".
}}

Now parse it:
{{
    "text":{text},
}}
{{
    "res":{{}}
}}
            '''.format(text=item['dialogue'])

    elif dataset_name == "reClor":
        
        options = "\nA. " + item['answers'][0] + \
            "\nB. " + item['answers'][1] + \
            "\nC. " + item['answers'][2] + \
            "\nD. " + item['answers'][3]
        io_sample = '''
here is the sample:
input:
{{
    "text":here is a sample text1,
    "question":"this is a question about text",
    "options":"A. option A
                B. option B
                C. option C
                D. option D
    "
}}
output:
{{
    "res":"A"
}}

Now parse it:
{{
    "text":{context},
    "question":{question},
    "options":{options}
}}
{{
    "res":{{}}
}}'''.format(context=item['context'],question =item['question'],options =options,)
    else:
        
        raise ValueError("Unsupported dataset name.")
    input_text = input_text+io_sample

    
    return input_text

def main(dataset_name, privacy_method,model_name,tokenizer):

    if privacy_method == 'raw':
        parser = StrOutputParser()
        # Use LangChain to process the data with LLMs
        llm = ChatOpenAI(model_name=model_name)

        data_path = Path(os.path.join('./eval_data', 'raw', dataset_name)).as_posix()
        data = load_dataset(dataset_name, data_path)
        new_data = []
        for item in tqdm(data):
            max_try = 5
            attemps = 0
            while attemps < max_try:
                try:
                    prompt_template = generate_prompt(item, dataset_name)
                    res = parser.invoke(llm.invoke(prompt_template))
                    res = utils.extract_json(res)

                    if res == 114514:
                        raise (Exception("未提取到json"))

                    result = json.loads(res)['res']
                    item[f'{model_name}_res'] = result
                    new_data.append(item)
                    break
                except Exception as e:
                    print(res)
                    print(e)
                    attemps = attemps + 1
        save_path = Path(data_path).as_posix()
        with open(save_path + f"/{dataset_name}.json", 'w') as f:
            json.dump(new_data, f)
    elif privacy_method == "Pptx":
        parser = StrOutputParser()
        # Use LangChain to process the data with LLMs
        llm = ChatOpenAI(model_name=model_name)

        data_path = Path(os.path.join('./eval_data', 'Pptx', dataset_name)).as_posix()
        data = load_dataset(dataset_name, data_path)
        new_data = []
        for item in tqdm(data):
            max_try = 5
            attemps = 0
            while attemps < max_try:
                try:
                    prompt_template = generate_prompt(item, dataset_name)
                    res = parser.invoke(llm.invoke(prompt_template))
                    res = utils.extract_json(res)

                    if res == 114514:
                        raise (Exception("未提取到json"))

                    result = json.loads(res)['res']
                    item[f'{model_name}_res'] = result
                    new_data.append(item)
                    break
                except Exception as e:
                    print(res)
                    print(e)
                    attemps = attemps + 1
        save_path = Path(data_path).as_posix()
        with open(save_path + f"/{dataset_name}.json", 'w') as f:
            json.dump(new_data, f)
    else:

        # Define experiments (privacy methods and RDF inclusion)
        experiments = [
            'eps_16.0',
            'eps_17.0',
            'eps_18.0',
            'eps_19.0',
            'eps_20.0',
            'eps_21.0',
            'eps_22.0',
            'eps_23.0',
            'eps_24.0',
            'eps_25.0',
        ]
        output_parser = StrOutputParser()
        # Use LangChain to process the data with LLMs
        llm = ChatOpenAI(model_name=model_name)
        for experiment in experiments:
            data_path = Path(os.path.join('./eval_data', privacy_method, dataset_name, experiment)).as_posix()
            # Load the dataset
            data = load_dataset(dataset_name, data_path)
            print(f'开始实验{dataset_name} {privacy_method} {experiment}')
            # if f'{model_name}_res' in data[0].keys():
            #     print(experiment+"已完成")
            #     continue
            new_data = []
            for item in tqdm(data):
                #下面这段代码是发现SAMsum返回结果有问题后打的补丁
                # if type(item['gpt-3.5-turbo_res']) != dict:
                #     new_data.append(item)
                #     continue
                max_try = 5
                attemps = 0
                while attemps < max_try:
                    try:
                        #print(1)
                        prompt_template = generate_prompt(item, dataset_name)
                        #print(prompt_template)
                        res = output_parser.invoke(llm.invoke(prompt_template))
                        #print(res)
                        #exit()
                        res = utils.extract_json(res)
                        if res == 114514:
                            raise(Exception("未提取到json"))
                        result = json.loads(res)['res']
                        if dataset_name == "SAMsum" and type(result) != str:
                            raise (Exception("返回类型错误"))
                        if dataset_name == "QNLI" and type(result) != str:
                            raise (Exception("返回类型错误"))
                        if dataset_name == "reClor" and type(result) != str:
                            raise (Exception("返回类型错误"))
                        if privacy_method == "icl":
                            item[f'{model_name}_res'] = icl_post_parse(result,item['private_word_map'],is_map=False)

                        else:
                            item[f'{model_name}_res'] = result
                        #print(item)
                        new_data.append(item)
                        break
                    except Exception as e:
                        attemps = attemps+1
                #break
            save_path = Path(data_path).as_posix()
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            with open(save_path+f"/{dataset_name}.json",'w') as f:
                json.dump(new_data,f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate privacy protection processing methods on different datasets.")
    parser.add_argument("--dataset_name", type=str, help="Name of the dataset (QNLI, SAMsum, reClor).")
    parser.add_argument("--privacy_method", type=str)
    args = parser.parse_args()
    tokenizer = None
    if args.privacy_method == 'icl':
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('../model', local_files_only=True)
    main(args.dataset_name, args.privacy_method,model_name, tokenizer)


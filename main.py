from pydantic import BaseModel
from openai import OpenAI
import argparse
from tqdm import tqdm
import os
import json
from utils import utils
from prompts import generate_prompt
import pickle

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
    
def create_replacement_maps(entities, semantic_graph,low_thresh, high_thresh):
    '''
    根据语义图的度数分布自适应地确定三级映射阈值
    '''
    # 计算每个实体的degree
    degree_map = {}
    for triple in semantic_graph:
        subject, _, object_ = triple
        degree_map[subject] = degree_map.get(subject, 0) + 1
        degree_map[object_] = degree_map.get(object_, 0) + 1
    
    def calculate_thresholds(degree_values,low_thresh, high_thresh):
        """
        简化阈值计算逻辑，使用百分位数
        """
        if not degree_values:
            return 1, 0
            
        sorted_degrees = sorted(degree_values)
        n_values = len(sorted_degrees)
        
        if n_values <= 2:  # 处理实体数量很少的情况
            return max(sorted_degrees), min(sorted_degrees)
            
        # 使用百分位数来确定阈值
        high_idx = max(0, int(n_values * high_thresh))  # 前20%为高度数
        medium_idx = max(0, int(n_values * low_thresh))  # 前50%为中度数
        
        high_threshold = sorted_degrees[high_idx]
        medium_threshold = sorted_degrees[medium_idx]
        
        return high_threshold, medium_threshold
    
    # 计算自适应阈值
    degree_values = list(degree_map.values())
    high_threshold, medium_threshold = calculate_thresholds(degree_values,low_thresh, high_thresh)
    
    #print(f"度数分布: {sorted(degree_values, reverse=True)}")
    #print(f"自适应阈值 - 高度数: {high_threshold}, 中度数: {medium_threshold}")
    
    def get_entity_level(entity):
        degree = degree_map.get(entity, 0)
        if degree > high_threshold:  # 使用严格大于
            return 'high'
        elif degree > medium_threshold:  # 使用严格大于
            return 'medium'
        else:
            return 'low'
    
    replacement_maps = {
        'level_1_mapping': {},  # 高度数实体: TYPE_INDEX
        'level_2_mapping': {},  # 中度数实体: LETTER_INDEX
        'level_3_mapping': {}   # 低度数实体: E_INDEX
    }
    
    # 记录每个实体类型的首字母
    type_first_letters = {}
    global_index = 1
    
    # 先对实体按度数排序
    all_entities = []
    for entity_type, items in entities.items():
        if isinstance(items, list):
            for item in items:
                if isinstance(item, list):
                    all_entities.extend((entity_type, sub_item) for sub_item in item)
                else:
                    all_entities.append((entity_type, item))
    
    # 按度数排序
    all_entities.sort(key=lambda x: degree_map.get(x[1], 0), reverse=True)
    
    # 按排序后的顺序进行映射
    for entity_type, item in all_entities:
        # 获取实体类型的首字母
        if entity_type not in type_first_letters:
            base_letter = entity_type[0].upper()
            letter_suffix = 1
            while base_letter in type_first_letters.values():
                base_letter = f"{entity_type[0].upper()}{letter_suffix}"
                letter_suffix += 1
            type_first_letters[entity_type] = base_letter
        
        base_letter = type_first_letters[entity_type]
        level = get_entity_level(item)
        
        if level == 'high':
            replacement_maps['level_1_mapping'][item] = f"{entity_type}_{global_index}"
        elif level == 'medium':
            replacement_maps['level_2_mapping'][item] = f"{base_letter}{global_index}"
        else:
            replacement_maps['level_3_mapping'][item] = f"Entity{global_index}"
        global_index += 1
    
    # 添加统计信息
    stats = {
        'degree_distribution': {
            'high_threshold': high_threshold,
            'medium_threshold': medium_threshold,
            'max_degree': max(degree_values) if degree_values else 0,
            'min_degree': min(degree_values) if degree_values else 0,
            'entity_counts': {
                'high': len(replacement_maps['level_1_mapping']),
                'medium': len(replacement_maps['level_2_mapping']),
                'low': len(replacement_maps['level_3_mapping'])
            }
        }
    }
    #将replacement_maps中的三类替换映射合并
    replacement_maps = {**replacement_maps['level_1_mapping'], **replacement_maps['level_2_mapping'], **replacement_maps['level_3_mapping']}
    return replacement_maps

def replace_substrings(text, mapping):
    """
    将文本中所有存在于字典键的子串进行映射替换
    :param text: 输入字符串
    :param mapping: 字符串->字符串映射的字典
    :return: 替换后的字符串
    """

    for key, value in mapping.items():
        #print(key)
        text = text.replace(key, value)
        
    return text

    '''
    text = "The quick brown fox jumps over the lazy dog"
    mapping = {
        "quick": "fast",
        "brown": "dark",
        "lazy": "sleepy"
    }'''
    
def privacy_protect(data, privacy_method,dataset_name, graph_type,low_thresh, high_thresh):
    """
    Protect privacy by applying the specified privacy method.
    """
    replacement_maps = {}
    if privacy_method == "no":
        if graph_type == 1:
            data['sematic_graph'] =  str(data['sematic_graph'])
        elif graph_type == 0:
            data['sematic_graph'] = str(data['nl_graph'])
        return data, None  # No privacy protection
    else:
        entities = data['entities']
        
        if dataset_name == "QNLI" and entities != []:
            replacement_maps = create_replacement_maps(entities, data['sematic_graph'],low_thresh, high_thresh)
            data['raw_text1'] = data['text1']
            data['raw_text2'] = data['text2']
            data['text1'] = replace_substrings(data['text1'],replacement_maps)
            data['text2'] = replace_substrings(data['text2'],replacement_maps)
            if graph_type == 1:
                data['sematic_graph'] = replace_substrings(str(data['sematic_graph']),replacement_maps)
            elif graph_type == 0:
                data['sematic_graph'] = replace_substrings(str(data['nl_graph']),replacement_maps)
            #data['logic_chain'] = replace_substrings(data['logic_chain'],replacement_maps)
        elif dataset_name == "QNLI" and entities == []:
            data['text1'] = data['text1']
            data['text2'] = data['text2']
            if graph_type == 1:
                data['sematic_graph'] =  str(data['sematic_graph'])
            elif graph_type == 0:
                data['sematic_graph'] = str(data['nl_graph'])
            
            #data['logic_chain'] = data['logic_chain']
        elif dataset_name == "SAMsum" and entities != []:
            replacement_maps = create_replacement_maps(entities, data['sematic_graph'],low_thresh, high_thresh)
            data['raw_dialogue'] = data['dialogue']
            data['dialogue'] = replace_substrings(data['dialogue'],replacement_maps)
            if graph_type == 1:
                data['sematic_graph'] = replace_substrings(str(data['sematic_graph']),replacement_maps)
            elif graph_type == 0:
                data['sematic_graph'] = replace_substrings(str(data['nl_graph']),replacement_maps)
            #data['logic_chain'] = replace_substrings(data['logic_chain'],replacement_maps)
        elif dataset_name == "SAMsum" and entities == []:
            data['dialogue'] = data['dialogue']
            if graph_type == 1:
                data['sematic_graph'] = str(data['sematic_graph'])
            elif graph_type == 0:
                data['sematic_graph'] = str(data['nl_graph'])
            #data['logic_chain'] = data['logic_chain']
        elif dataset_name == "reClor" and entities != []:
            replacement_maps = create_replacement_maps(entities, data['sematic_graph'],low_thresh, high_thresh)
            data['raw_context'] = data['context']
            data['raw_question'] = data['question']
            data['raw_answers'] = data['answers']
            data['context'] = replace_substrings(data['context'],replacement_maps)
            data['question'] = replace_substrings(data['question'],replacement_maps)
            data['answers'] = [
                    replace_substrings(data['answers'][0],replacement_maps),
                    replace_substrings(data['answers'][1],replacement_maps),
                    replace_substrings(data['answers'][2],replacement_maps),
                    replace_substrings(data['answers'][3],replacement_maps)
                ]
            if graph_type == 1:
                data['sematic_graph'] = replace_substrings(str(data['sematic_graph']),replacement_maps)
            elif graph_type == 0:
                data['sematic_graph'] = replace_substrings(str(data['nl_graph']),replacement_maps)

            #data['logic_chain'] = replace_substrings(data['logic_chain'],replacement_maps)
        elif dataset_name == "reClor" and entities == []:
            data['context'] = data['context']
            data['question'] = data['question']
            data['answers'] = data['answers']
            #print(type(data['answers']))
            if graph_type == 1:
                data['sematic_graph'] = str(data['sematic_graph'])
            elif graph_type == 0:
                data['sematic_graph'] = str(data['nl_graph'])
            #data['logic_chain'] = data['logic_chain']
        return data, replacement_maps

def swap_keys_values(original_dict):
    """
    Swap keys and values in a dictionary.

    Args:
        original_dict (dict): The original dictionary to swap.

    Returns:
        dict: A new dictionary with keys and values swapped.

    Raises:
        ValueError: If there are duplicate values in the original dictionary.
        TypeError: If a value in the original dictionary is unhashable.
    """
    # Check for duplicate values
    
    if len(set(original_dict.values())) != len(original_dict):
        raise ValueError("Cannot swap keys and values: duplicate values detected.")

    try:
        # Swap keys and values
        swapped_dict = {value: key for key, value in original_dict.items()}
    except TypeError as e:
        raise TypeError("Cannot swap keys and values: unhashable value detected.") from e

    return swapped_dict


def refine_privacy(data, replacement_maps):
    #print(data)
    en_map =  swap_keys_values(replacement_maps)
    
    """
    Restore the original text by using replacement maps.
    """
    refine_sentence = replace_substrings(data,en_map)
    return refine_sentence


def main(dataset_name, dataset_path, privacy_method,model_name, use_semantic_graph,use_logic_chain, graph_type,low_thresh, high_thresh):
    print(use_semantic_graph)
    print(use_logic_chain)
    # Load the dataset
    data = load_data(dataset_path)
    client = OpenAI(
        base_url="",
        api_key=""
    )

    for data_i in tqdm(data):
        privacy_data,replacement_maps = privacy_protect(data_i, privacy_method,dataset_name,graph_type,low_thresh, high_thresh)
        if graph_type == 1:
            curr_prompt = generate_prompt(privacy_data, dataset_name, privacy_method, "RDF", use_semantic_graph, use_logic_chain)
        elif graph_type ==0:
            curr_prompt = generate_prompt(privacy_data, dataset_name, privacy_method, "NL", use_semantic_graph, use_logic_chain)
        max_try = 5
        attemps = 0
        while attemps < max_try:
            try:
                completion = client.beta.chat.completions.parse(
                    model=model_name,
                    messages=curr_prompt,
                    response_format={ "type": "json_object" },
                )
                
                # res = utils.extract_json(res)
                
                # if res == 114514:
                #     raise(Exception("未提取到json"))
                #print(completion.choices[0].message.content)
                response_content = json.loads(completion.choices[0].message.content)
                #print(response_content['res'])
                

                #print(result)
                if dataset_name == "SAMsum" and privacy_method != "no":
                    refine = refine_privacy(response_content['res'],replacement_maps)
                   # print(response_content['res'])
                    #print(refine)
                    data_i['gpt-3.5-turbo_res'] = refine                  
                    
                else:
                    data_i['gpt-3.5-turbo_res'] = response_content['res']
                break
            except Exception as e:
                print(e)
                attemps = attemps+1
    # 保存文件
    if use_semantic_graph and use_logic_chain:
        save_path = f"./dataset/{privacy_method}/{"SPO" if graph_type==1 else "NL"}/{str(high_thresh)}_{str(low_thresh)}/graph_chain/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open(save_path+f"{dataset_name}.json", "w", encoding="utf-8") as f:
            print(save_path)
            json.dump(data,f)

    elif not use_semantic_graph and use_logic_chain:
        save_path = f"./dataset/{privacy_method}/{"SPO" if graph_type==1 else "NL"}/{str(high_thresh)}_{str(low_thresh)}/chain/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open(save_path+f"{dataset_name}.json", "w", encoding="utf-8") as f:
            print(save_path)
            json.dump(data,f)
    elif use_semantic_graph and not use_logic_chain:
        save_path = f"./dataset/{privacy_method}/{"SPO" if graph_type==1 else "NL"}/{str(high_thresh)}_{str(low_thresh)}/graph/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open(save_path+f"{dataset_name}.json", "w", encoding="utf-8") as f:
            print(save_path)
            json.dump(data,f)
    elif not use_semantic_graph and not use_logic_chain:
        save_path = f"./dataset/{privacy_method}/{"SPO" if graph_type==1 else "NL"}/{str(high_thresh)}_{str(low_thresh)}/none/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open(save_path+f"{dataset_name}.json", "w", encoding="utf-8") as f:
            print(save_path)
            json.dump(data,f)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate privacy protection processing methods on different datasets.")
    parser.add_argument("--dataset_name", type=str, help="Name of the dataset (QNLI, SAMsum, reClor).")
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset.")
    parser.add_argument("--privacy_method", type=str, help="Privacy protection method (e.g., entity_with_index, no).")
    parser.add_argument("--model_name", type=str, help="model")
    parser.add_argument("--use_semantic_graph", type=int, help="model")
    #1代表结构化，0代表自然语言
    parser.add_argument("--graph_type", type=int, help="model")
    parser.add_argument("--low_thresh", type=float, help="model")
    parser.add_argument("--high_thresh", type=float, help="model")
    parser.add_argument("--use_logic_chain", type=int, help="model")
    args = parser.parse_args()

    privacy_method_dict = {
        "1":"code_sanitization",
        "4":"no"
    }
    
    main(args.dataset_name, 
         args.dataset_path, 
         privacy_method_dict[args.privacy_method],
         args.model_name, 
         bool(args.use_semantic_graph),
         bool(args.use_logic_chain), 
         args.graph_type,
         args.low_thresh,
         args.high_thresh,
         )
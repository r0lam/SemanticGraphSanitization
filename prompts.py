'''
inference time prompt
'''
task_description_dict = {
    "QNLI":'''You will be given a pair of sentences, including a question and a sentence. Your task is to determine whether the sentence answers the question. Please respond with "entailment" or "not entailment".''',
    "SAMsum":'''You need to complete a text summarization task, summarizing the content of the given text.''',
    "reClor":'''you should give the answer among A to D.'''
}
constrain = '''
Note: You must follow the example pattern and return a JSON data containing the 'res' field. Do not respond with any text other than the JSON data!!!!
Note: You must follow the example pattern and return a JSON data containing the 'res' field. Do not respond with any text other than the JSON data!!!!
    '''
privacy_method_description = {
    'code_sanitization':'''
    To protect the privacy of the text, we replace some entities in the text with codes.There three types of code:
    1. Entity Type+ID: For example, PERSON1, DATE1, etc.
    2. Entity Type's first letter+ID: For example, P1(which means PERSON1),  etc.
    3. 'Entity'+ID: For example, Entity1, Entity2, etc.
    ''',
    "no":''''''
}

semantic_graph_prompt = '''
To help you better complete the task, we provide a semantic graph constructed from the content of the text. Specifically, the "semantic_graph" field contains a semantic graph in the form of SPO triples to help you better understand the main structure of the text.
'''
semantic_graph_nutral_language_prompt = '''
To help you better complete the task, we provide a semantic graph constructed from the content of the text. Specifically, the "semantic_graph" field contains a semantic graph descriped in natural language.
'''
logic_chain_prompt = '''
To help you better complete the task, we provide a logical chain constructed from the content of the text. Specifically, the "logic_chain" field contains a logical chain to help you better understand the main structure of the text.
'''
result_examples = {
    "QNLI":'''
{
    "res":"entailment"
}
    ''',
    "SAMsum":'''
{
    "res":"this the summary of the input text".
}
    ''',
    "reClor":'''
{
    "res":"A"
}
    '''
}
# 输入data的测试数据
def input_protocols(data,task_name,use_sematic_graph,use_logic_chain,example_text:bool):
    if task_name == "QNLI" and use_sematic_graph and use_logic_chain:
        if example_text:
            return {
                "text1":"example input text1",
                "text2":"example input text2",
                "sematic_graph":"example input sematic graph",
                "logic_chain":"example input logic chain"
            }
        else:
            return {
                "text1":data['text1'],
                "text2":data['text2'],
                "sematic_graph":data['sematic_graph'],
                "logic_chain":data['logic_chain']
            }
    elif task_name == "QNLI" and use_sematic_graph and not use_logic_chain:
        if example_text:
            return {
                "text1":"example input text1",
                "text2":"example input text2",
                "sematic_graph":"example input sematic graph",
            }
        else:
            return {
                "text1":data['text1'],
                "text2":data['text2'],
                "sematic_graph":data['sematic_graph'],
            }
    elif task_name == "QNLI" and not use_sematic_graph and use_logic_chain:
        if example_text:
            return {
                "text1":"example input text1",
                "text2":"example input text2",
                "logic_chain":"example input logic chain"
            }
        else:
            return {
                "text1":data['text1'],
                "text2":data['text2'],
                "logic_chain":data['logic_chain']
            }
    elif task_name == "QNLI" and not use_sematic_graph and not use_logic_chain:
        if example_text:
            return {
                "text1":"example input text1",
                "text2":"example input text2",
            }
        else:
            return {
                "text1":data['text1'],
                "text2":data['text2']
            }
    elif task_name == "reClor" and use_sematic_graph and use_logic_chain:
        if example_text:
            return {
                "text":"example input text",
                "question":"example input question",
                "options":"example input options",
                "sematic_graph":"example input sematic graph",
                "logic_chain":"example input logic chain"
            }
        else:
            options = "\nA. " + data['answers'][0] + \
            "\nB. " + data['answers'][1] + \
            "\nC. " + data['answers'][2] + \
            "\nD. " + data['answers'][3] 
            return {
                "text":data['context'],
                "question":data['question'],
                "options":options,
                "sematic_graph":data['sematic_graph'],
                "logic_chain":data['logic_chain']
            }
    elif task_name == "reClor" and use_sematic_graph and not use_logic_chain:
        if example_text:
            return {
                "text":"example input text",
                "question":"example input question",
                "options":"example input options",
                "sematic_graph":"example input sematic graph",
            }
        else:
            options = "\nA. " + data['answers'][0] + \
            "\nB. " + data['answers'][1] + \
            "\nC. " + data['answers'][2] + \
            "\nD. " + data['answers'][3] 
            return {
                "text":data['context'],
                "question":data['question'],
                "options":options,
                "sematic_graph":data['sematic_graph'],
            }
    elif task_name == "reClor" and not use_sematic_graph and use_logic_chain:
        if example_text:
            return {
                "text":"example input text",
                "question":"example input question",
                "options":"example input options",
                "sematic_graph":"example input sematic graph",
            }
        else:
            options = "\nA. " + data['answers'][0] + \
            "\nB. " + data['answers'][1] + \
            "\nC. " + data['answers'][2] + \
            "\nD. " + data['answers'][3] 
            return {
                "text":data['context'],
                "question":data['question'],
                "options":options,
                "sematic_graph":data['sematic_graph'],
            }
    elif task_name == "reClor" and not use_sematic_graph and not use_logic_chain:
        if example_text:
            return {
                "text":"example input text",
                "question":"example input question",
                "options":"example input options",
                "logic_chain":"example input logic chain"
            }
        else:
            options = "\nA. " + data['answers'][0] + \
            "\nB. " + data['answers'][1] + \
            "\nC. " + data['answers'][2] + \
            "\nD. " + data['answers'][3] 
            return {
                "text":data['context'],
                "question":data['question'],
                "options":options,
                "logic_chain":data['logic_chain']
            }
    elif task_name == "SAMsum" and use_sematic_graph and use_logic_chain:
        if example_text:
            return {
                "text":"example input text",
                "sematic_graph":"example input sematic graph",
                "logic_chain":"example input logic chain"
            }
        else:
            return {
                "text":data['dialogue'],
                "sematic_graph":data['sematic_graph'],
                "logic_chain":data['logic_chain']
            }
    elif task_name == "SAMsum" and use_sematic_graph and not use_logic_chain:
        if example_text:
            return {
                "text":"example input text",
                "sematic_graph":"example input sematic graph",
            }
        else:
            return {
                "text":data['dialogue'],
                "sematic_graph":data['sematic_graph'],
            }
    elif task_name == "SAMsum" and not use_sematic_graph and use_logic_chain:
        if example_text:
            return {
                "text":"example input text",
                "logic_chain":"example input logic chain"
            }
        else:
            return {
                "text":data['dialogue'],
                "logic_chain":data['logic_chain']
            }
    elif task_name == "SAMsum" and not use_sematic_graph and not use_logic_chain:
        if example_text:
            return {
                "text":"example input text",
            }
        else:
            return {
                "text":data['dialogue'],
            }
def generate_prompt(data,task_name, privacy_method, semantic_graph_type,use_sematic_graph,use_logic_chain: bool):
    semantic_graph_config = {
        "semantic_graph_description":{
            "RDF":'''To help you better complete the task, we provide a semantic graph constructed from the content of the text. Specifically, the "semantic_graph" field contains a semantic graph in the form of RDF triples to help you better understand the main structure of the text.''',
            "NL":'''To help you better complete the task, we provide a semantic graph constructed from the content of the text. Specifically, the "semantic_graph" field contains a semantic graph descriped in natural language.''',
        },
        "semantic_graph_example":{
            "RDF":'''
    (A, is friends, B)
            ''',
            "NL":'''
    A is friends with B
            '''
        }
    }    
    semantic_graph_prompt = semantic_graph_config["semantic_graph_description"][semantic_graph_type]
    result_example = result_examples[task_name]
    input_protocol = input_protocols(data,task_name,use_sematic_graph,use_logic_chain,False)
    input_example = input_protocols(data,task_name,use_sematic_graph,use_logic_chain,True)
    if use_sematic_graph and use_logic_chain:
        messages_list = [
{"role":"system", "content": f'''
[task description]
{task_description_dict[task_name]}
[IMPORTANT]
{constrain}
[privacy method]
{privacy_method_description[privacy_method]}
[semantic graph]
{semantic_graph_prompt}
[logic chain]
{logic_chain_prompt}
'''},
{"role":"user", "content":f'''
{str(input_example)}
'''},
{"role":"assistant", "content":f'''
{result_example}
'''},
{"role":"user", "content":f'''
{str(input_protocol)}
'''}
        ]
    elif not use_sematic_graph and use_logic_chain:
        messages_list = [
{"role":"system", "content": f'''
[task description]
{task_description_dict[task_name]}
[IMPORTANT]
{constrain}
[privacy method]
{privacy_method_description[privacy_method]}
[logic chain]
{logic_chain_prompt}
'''},
{"role":"user", "content":f'''
{str(input_example)}
'''},
{"role":"assistant", "content":f'''
{result_example}
'''},
{"role":"user", "content":f'''
{str(input_protocol)}
'''}
        ]
    elif use_sematic_graph and not use_logic_chain:
        messages_list = [
{"role":"system", "content": f'''
[task description]
{task_description_dict[task_name]}
[IMPORTANT]
{constrain}
[privacy method]
{privacy_method_description[privacy_method]}
[semantic graph]
{semantic_graph_prompt}
'''},
{"role":"user", "content":f'''
{str(input_example)}
'''},
{"role":"assistant", "content":f'''
{result_example}
'''},
{"role":"user", "content":f'''
{str(input_protocol)}
'''}
        ]
    elif not use_sematic_graph and not use_logic_chain:
        messages_list = [
{"role":"system", "content": f'''
[task description]
{task_description_dict[task_name]}
[IMPORTANT]
{constrain}
[privacy method]
{privacy_method_description[privacy_method]}
'''},
{"role":"user", "content":f'''
{str(input_example)}
'''},
{"role":"assistant", "content":f'''
{result_example}
'''},
{"role":"user", "content":f'''
{str(input_protocol)}
'''}
        ]
    return messages_list

import os

model_and_key = {
    "gpt-3.5-turbo":"",
    "gpt-4-turbo":"",
    "llama3-70b-8192":"",
    "Qwen/Qwen2.5-7B-Instruct":"",
    "gpt-4o-2024-08-06":"",
}



os.environ["OPENAI_API_BASE"] = ""

model_name = "gpt-3.5-turbo"

def use_model(model_name):
    os.environ["OPENAI_API_KEY"] = model_and_key[model_name]


use_model(model_name)

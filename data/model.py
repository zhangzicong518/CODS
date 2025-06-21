from openai import OpenAI
import requests
import json
import base64
import time
import os
import random
import numpy as np

url = "https://api.siliconflow.cn/v1/chat/completions"

payload = {
    "model": "Pro/Qwen/Qwen2.5-VL-7B-Instruct",
    "messages": [],
    "stream": True,
    "max_tokens": 2048,
    "stop": None,
    "temperature": 0.7,
    "top_p": 0.7,
    "top_k": 50,
    "frequency_penalty": 0.5,
    "response_format": {"type": "text"},
}

headers = {
    "Authorization": "Bearer sk-pbdopscjkjgkuqigfewlxcuyrhzmlusuukxljdqcccufyywu",
    "Content-Type": "application/json"
}

prompt_template = """Given the an image question pair below, imagine you are answering it. Which of the following capabilities would you rely on to do so?

Select one or more options from the list (A-M) and respond using only the corresponding letter(s), separated by commas (e.g., A, B, C).

Capabilities:
A. Attribute Recognition: Recognition of texture, shape, appearance, characteristics, emotions, category. 
B. Object Localization: For a single object, determine its position in the image (such as top, bottom, etc.), its absolute coordinates in the image, count the number of objects, and the orientation of the object.
C. OCR: Recognize text in the image, including numbers, letters, and symbols.   
D. Spatial Reasoning: Determine the relative position between objects in image.
E. Action Recognition: Recognizing human actions, including pose motion, human-object interaction, and human-human interaction.
F. Captioning: Generate a caption for the image.
G. Relation Reasoning: Relations in human society or relations defined from the human perspective.
H. Function Reasoning: Predict the function of an object. Examples: the function of a broom is to sweep the floor, the function of a spatula is to cook, the function of a pen is to write, etc.
I. Logical Reasoning: Logical reasoning, including mathematical reasoning, logical deduction, and other logical reasoning tasks.
J. Physical Property: Predict the physical property of an object.
K. Identity Reasoning: Predict the identity of a person.
L. Future Prediction: Predict what will happen in the future.
M. Structural Understanding: Structured understanding of images and text, including parsing the content of charts (such as the trends of multiple bars in a bar chart), understanding the code in an image, etc.

Question: {question}


Your answer should only include a comma-separated list of letters (e.g., A, B, C). Do not include any other text or explanations.

"""

def to_base64(img_path):
    with open(img_path, 'rb') as image_file:
        encoded_string = base64.b64encode(image_file.read())
        return encoded_string.decode('utf-8')

def recover_from_label(response):
    
    labels = []
    
    if "A" in response:
        labels.append("Attribute Recognition")
    if "B" in response:
        labels.append("Object Localization")
    if "C" in response:
        labels.append("OCR")
    if "D" in response:
        labels.append("Spatial Reasoning")
    if "E" in response:
        labels.append("Action Recognition")
    if "F" in response: 
        labels.append("Captioning")
    if "G" in response:
        labels.append("Relation Reasoning")
    if "H" in response:
        labels.append("Function Reasoning")
    if "I" in response:
        labels.append("Logical Reasoning")
    if "J" in response:
        labels.append("Physical Property")
    if "K" in response: 
        labels.append("Identity Reasoning")
    if "L" in response:
        labels.append("Future Prediction")
    if "M" in response:
        labels.append("Structural Understanding")
        
    return labels
    
class Qwen2_5_VL():
    
    def __init__(self):
        # payload["model"] = "Qwen/Qwen3-30B-A3B"
        pass
    
    def inference(self, image_path, ques, max_retries=3, retry_delay=2):
        
        response = None
        
        # Initialize retry counter
        retries = 0
        success = False
        
        while retries < max_retries and not success:
            try:
                # Format the message payload
                payload["messages"] = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_template.format(question=ques)},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{to_base64(image_path)}",
                                    "detail": "auto"
                                }
                            }
                        ]
                    }
                ]
                # Make the API request
                response = requests.post(url, json=payload, headers=headers)
                if response.status_code == 200:
                    full_content = ""
                    for chunk in response.iter_lines():
                        if chunk:
                            try:
                                chunk_str = chunk.decode('utf-8').replace('data: ', '')
                                if chunk_str != "[DONE]":
                                    chunk_data = json.loads(chunk_str)
                                    delta = chunk_data['choices'][0].get('delta', {})
                                    content = delta.get('content', '')
                                    if content:
                                        full_content += content
                            except json.JSONDecodeError as e:
                                print(f"Error decoding chunk: {e}")
                                continue
                    
                    response = full_content
                    success = True
                else:
                    print(f"Attempt {retries + 1}/{max_retries} failed with status {response.status_code}")
                    print(f"Error response: {response.text}")
                    retries += 1
                    if retries < max_retries:
                        print(f"Retrying in {retry_delay} seconds...")
            
            except Exception as e:
                print(f"Attempt {retries + 1}/{max_retries} failed with error: {str(e)}")
                retries += 1
                if retries < max_retries:
                    print(f"Retrying in {retry_delay} seconds...")
        
        if not success:
            response = None
            print("All attempts failed. Please check the input or the API.")
        
        # print(response)
        
        if response:
            response = recover_from_label(response)
        return response


# model = Qwen2_5_VL()

# res = model.inference("/home/zicong/DATASETS/Infinity-MM/onevision-SI/stage3/onevision-SI/0/0.jpg", "Hint: Please answer the question and provide the final answer at the end.\nQuestion: How many shapes are blue?")

# print(res)
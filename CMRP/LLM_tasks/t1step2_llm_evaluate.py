import json
import requests
import os
import re
import pickle
import difflib
import random
import replicate
from zhipuai import ZhipuAI
import time
from http import HTTPStatus
# import dashscope
# api_key = 'sk-9yq2sFnrmOdRK2PwC7BcDcDc510040Cc8a7b375cB1076b9c'
# api_key = 'sk-JWll537xNBQGPJYjF6FbF51b76Cf4d54BfCa62Ce9d0b1d78'
# api_key = 'sk-pWqarmcUUVd3HkaB45C8AeA45f754dBd8eA0A35fAe3f7100'
# api_key1 = 'sk-9yq2sFnrmOdRK2PwC7BcDcDc510040Cc8a7b375cB1076b9c'
# api_key2 = 'sk-JWll537xNBQGPJYjF6FbF51b76Cf4d54BfCa62Ce9d0b1d78'
# api_key3 = 'sk-pWqarmcUUVd3HkaB45C8AeA45f754dBd8eA0A35fAe3f7100'
# api_key4 = 'sk-8FvVpQuBf46eGGr8BbCfF38bB3F64b69Aa40366e801f6271'
# api_key5 = 'sk-viyCtivavwvqvyZ4B71bC4A8DfBf40748820A6129f4fCdA5'
# api_key = 'sk-DEc5WgFTjPRa2GD46e6dE90553814207889669E91bDb6a8e'

# def sample_sync_call(prompt_text):
#     # prompt_text = '用萝卜、土豆、茄子做饭，给我个菜谱。'
#     resp = dashscope.Generation.call(
#         model='qwen-turbo',
#         prompt=prompt_text
#     )
#     # The response status_code is HTTPStatus.OK indicate success,
#     # otherwise indicate request is failed, you can get error code
#     # and message from code and message.
#     if resp.status_code == HTTPStatus.OK:
#         print(resp.output)  # The output text
#         print(resp.usage)  # The usage information
#         return resp.output
#     else:
#         print(resp.code)  # The error code.
#         print(resp.message)  # The error message.

# def sample_call_streaming(prompt_text):
#     # prompt_text = '用萝卜、土豆、茄子做饭，给我个菜谱。'
#     response_generator = dashscope.Generation.call(
#         model='qwen-turbo',
#         prompt=prompt_text,
#         stream=True,
#         top_p=0.8)
#     # When stream=True, the return is Generator,
#     # need to get results through iteration
#     for response in response_generator:
#         # The response status_code is HTTPStatus.OK indicate success,
#         # otherwise indicate request is failed, you can get error code
#         # and message from code and message.
#         if response.status_code == HTTPStatus.OK:
#             print(response.output)  # The output text
#             print(response.usage)  # The usage information
#             return response.output
#         else:
#             print(response.code)  # The error code.
#             print(response.message)  # The error message.


def is_number(string):
    pattern = re.compile(r'^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$')
    return bool(pattern.match(string))

def similar_relations(r1, r2, threshold=0.2):
    return difflib.SequenceMatcher(None, r1, r2).ratio() > threshold

def prune_knowledge_graph(triples_list, relation, similarity_threshold=0.2):
    pruned_triples = []

    for triple_group in triples_list:
        similar_triples = [triple for triple in triple_group if similar_relations(triple[1], relation, similarity_threshold)]
        if similar_triples:
            pruned_triples.append(similar_triples)

    return pruned_triples



def extract_triplets_and_questions(data):
    triplets_with_questions = []
    for item in data:
        entity1 = list(item["topic_entity"].values())[0]
        # entity1 = item["TopicEntityName"]  # 获取实体名称
        relation = item["relation"]
        answer = item["answer"]
        question = item["question"] #对于SimpleQA.json,related user prompt我们认为就是question字段
        triplets_with_questions.append({"entity1": entity1, "relation": relation, "answer": answer, "question": question})
    return triplets_with_questions



def generate_prompt(related_kg_triplets):
    return f"""
    .
    """

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)


def call_llm_api(prompt, max_retries=10, timeout=10,api_key='',api_site=''):
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    data = {
        'model': 'gpt-3.5-turbo-1106',
        'temperature': 0.7,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }
    for t in range(max_retries):
        try:
            response = requests.post(api_site, headers=headers, json=data, timeout=timeout)
            # response = requests.post('https://api.kwwai.top/v1/chat/completions', headers=headers, json=data, timeout=timeout)
            #response = requests.post('http://10.13.14.16:8000/v1/chat/completions', headers=headers, json=data, timeout=timeout)
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                return f"Error: {response.status_code}, {response.text}"
        except requests.Timeout:
            print("Request timed out, retrying...")
        except requests.RequestException as e:
            # 其他网络相关错误可以在这里处理
            return f"Network error: {e}"

    return "Error: Maximum retries reached"

# LLama2 的 API
def call_llama7b_api(prompt):
    os.environ['REPLICATE_API_TOKEN'] = 'r8_WVSqzjgPuGB6O0icsd8LSiBL7VdudVw2doM3S'
    # output = replicate.run(
    # "meta/llama-2-7b-chat",
    # input={
    #     "top_p": 1,
    #     "prompt": prompt,
    #     "temperature": 0.75,
    #     "system_prompt": "You are a helpful assistant",
    #     "max_new_tokens": 800,
    #     "repetition_penalty": 1
    # }
    # )

    output=''
    for event in replicate.stream(
    "meta/llama-2-7b-chat",
    input={
        "prompt": prompt,
        "temperature": 0.5,
        "system_prompt": "You are a helpful assistant who strictly answer the question in word: 'Yes' or 'No'. DO NOT USE dot symbol",
    },
    ):
        output=output+str(event)
    
    #output = re.sub(r'\n', '', output)
    #output = re.sub('.', '', output)

    return output.replace('\n', '').replace('.', '')

# baichuan 的 API
# def call_baichuan_api(prompt):
#     url = "https://api.baichuan-ai.com/v1/chat/completions"
#     # api_key = "sk-2a885aaf1618dc9d78943f1ed6252aee"
#     api_key="sk-9f7a554c7d1e5fdada79e3ef89fc5808"
#     # api_key="sk-f76a150251789d206d3abcc6bf2e36ad"

#     data = {
#         "model": "Baichuan2-Turbo",
#         "messages": [
#             {
#                 "role": "user",
#                 "content": prompt
#             }
#         ],
#         "stream": False
#     }

#     json_data = json.dumps(data)

#     headers = {
#         "Content-Type": "application/json",
#         "Authorization": "Bearer " + api_key
#     }
#     time.sleep(1)
#     response = requests.post(url, data=json_data, headers=headers, timeout=60)

#     json_result = json.loads(response.text)

#     if response.status_code != 200:
#         print("请求失败，状态码:", response.status_code)
#         print("请求失败，body:", response.text)
#         print("请求失败，X-BC-Request-Id:", response.headers.get("X-BC-Request-Id"))
    
#     return json_result['choices'][0]['message']['content']

# ChatGLM3 的 API
def call_glm_api(prompt):
    
    client = ZhipuAI(api_key="1209462db651581cc1109a0f085d71bb.hYnWiG3sIXj8KRDO") # 填写您自己的APIKey
    response = client.chat.completions.create(
    model="glm-3-turbo", # 填写需要调用的模型名称
    messages=[
        {"role": "user", "content": prompt
         }
    ],)
    # print(response.choices[0].message)
    return response.choices[0].message.content

# def call_qwen_turbo_api(prompt):
#     messages=sample_call_streaming(prompt)
#     # messages = sample_sync_call(prompt)
#     return messages


def extract_valid_relations_from_sparql(sparql_query):
    relations = re.findall(r'ns:([a-zA-Z]+(?:\.[a-zA-Z]+)+)', sparql_query)
    return list(set(relations))

def ensure_at_least_one_relation(relations):
    if not relations:
        return ["unknown_relation"]  # 或者其他默认值
    return relations

def extract_questions_answers_relations_CWQ(file_path):
    data = read_json_file(file_path)
    extracted_data = []
    for item in data:
        question = item.get("question", "")
        answer = item.get("answer", "")
        relations = extract_valid_relations_from_sparql(item.get("sparql", ""))
        relations = ensure_at_least_one_relation(relations)
        extracted_data.append({"question": question, "answer": answer, "relations": relations})
    return extracted_data

def extract_questions_answers_relations_WebQSP(file_path):
    data = read_json_file(file_path)
    extracted_data = []
    for item in data:
        question = item.get("ProcessedQuestion", "")
        answers = [ans.get("EntityName", "") for parse in item.get("Parses", []) for ans in parse.get("Answers", [])]
        relations = []
        for parse in item.get("Parses", []):
            relations.extend(extract_valid_relations_from_sparql(parse.get("Sparql", "")))
        relations = ensure_at_least_one_relation(relations)
        extracted_data.append({"question": question, "answers": list(set(answers)), "relations": relations})
    return extracted_data


import random

def load_data_for_training_and_test(input_file_path=f'../preprocess/SimpleQA.json', dataset='FB15K237'):
    print('dataset = ', dataset)
    print('input_file_path = ',input_file_path)
    if(input_file_path == '../preprocess/SimpleQA.json'):
        data = read_json_file(input_file_path)
        triplets_with_questions = extract_triplets_and_questions(data)
    elif(input_file_path== '../preprocess/CWQ.json'):
        triplets_with_questions = extract_questions_answers_relations_CWQ(input_file_path)
    elif(input_file_path=='../preprocess/WebQSP.json'):
        triplets_with_questions = extract_questions_answers_relations_WebQSP(input_file_path)

    all_subgraphs_dict = pickle.load(open(f'../LLM_tasks/{dataset}_all_subgraphs_dict.pkl', 'rb'))

    # 随机选择20个键
    selected_keys = random.sample(list(all_subgraphs_dict.keys()), 20)

    # 对于选中的键，从all_subgraphs_dict和triplets_with_questions中提取数据
    selected_subgraphs = {key: all_subgraphs_dict[key] for key in selected_keys}
    selected_triplets_with_questions = [triplets_with_questions[key] for key in selected_keys]

    # 将数据分为训练集和测试集
    train_subgraphs = list(selected_subgraphs.values())[:10]
    test_subgraphs = list(selected_subgraphs.values())[10:]

    train_triplets_with_questions = selected_triplets_with_questions[:10]
    test_triplets_with_questions = selected_triplets_with_questions[10:]

    return train_triplets_with_questions, test_triplets_with_questions, train_subgraphs, test_subgraphs



def llm_qa(QA_dataset,triplets_with_questions, number, kg_triplets,k=20,llm_api='chatgpt',api_key='sk-DEc5WgFTjPRa2GD46e6dE90553814207889669E91bDb6a8e',api_site='https://oneapi.xty.app/v1/chat/completions'):
    start_time = time.time()
    if len(triplets_with_questions) < number:
        number = len(triplets_with_questions)
    # Process only the first {number} items
    scores = []
    kg_triplets_new= []
    triplets_with_questions_new = []
    
    prompt = None
    noise=""
    for i, item in enumerate(triplets_with_questions):
        
        if i >= number:  # Stop after processing 10 items
            break
        if(QA_dataset == 'SimpleQA'):
            # print("SimpleQA items",item)
            start_time_load = time.time()

            relation_to_keep = item['relation']
            relation_to_keep = relation_to_keep.replace("_"," ")
            kg_triplets[i] = prune_knowledge_graph(kg_triplets[i], relation_to_keep)
            kg_triplets[i]= kg_triplets[i][:2]
            kg_triplets_new.append(kg_triplets[i])
            triplets_with_questions_new.append(triplets_with_questions[i])

            end_time_load = time.time()
            total_seconds = end_time_load - start_time_load
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            print(f"SimpleQA data load运行总时长： {int(hours)}, {int(total_seconds)}秒,{int(minutes)}分钟")


            if k>0:
                prompt=(f"Please add some random words no more than {k} percent of the following question"
                        f" Question: {item['question']}"
                        )
                start_time_api = time.time()
                noise = call_llm_api(prompt,max_retries=10, timeout=10,api_key=api_key,api_site=api_site)
                end_time_api = time.time()
                total_seconds = end_time_api - start_time_api
                hours = total_seconds // 3600
                minutes = (total_seconds % 3600) // 60
                print(f"SimpleQA 计算noise运行总时长： {int(hours)}, {int(total_seconds)}秒,{int(minutes)}分钟")

                prompt = (f"Answer the following question with simple answers (less than 5 words) based on the knowledge triplets: '{kg_triplets[i]}', "
                f" Question: {item['question']}"
                f" {noise}"
                )
            else:
                prompt = (f'''knowledge triplets: '[[('large, above average in size or number or quantity or magnitude or extent; "a large city"; "set out for the big city"; "a large sum"; "a big (or large) barn"; "a large family"; "big businesses"; "a big expenditure"; "a large number of newspapers"; "a big group of scientists"; "large areas of the world"', 'derivationally related form', 'largeness, the property of having a relatively great size')]]',  Question: What Portuguese-speaking country's market cap of list companies as percent of gdp was 10.43? Answer:Brazil

'[[('large, above average in size or number or quantity or magnitude or extent; "a large city"; "set out for the big city"; "a large sum"; "a big (or large) barn"; "a large family"; "big businesses"; "a big expenditure"; "a large number of newspapers"; "a big group of scientists"; "large areas of the world"', 'also see', 'tall, great in vertical dimension; high in stature; "tall people"; "tall buildings"; "tall trees"; "tall ships"')], [('large, above average in size or number or quantity or magnitude or extent; "a large city"; "set out for the big city"; "a large sum"; "a big (or large) barn"; "a large family"; "big businesses"; "a big expenditure"; "a large number of newspapers"; "a big group of scientists"; "large areas of the world"', 'derivationally related form', 'largeness, the property of having a relatively great size')]]',  Question: Where did Drew Brees attend university, that has less than 31,290 undergraduates? Answer: Purdue University
                '''
                    f"Answer the following question with simple answers (less than 5 words) based on the knowledge triplets: '{kg_triplets[i]}', "
                f" Question: {item['question']}"
                )
                print("first prompt",prompt)
            if llm_api=='llama7b':
                start_time_api = time.time()

                ans=call_llama7b_api(prompt)

                end_time_api = time.time()
                total_seconds = end_time_api - start_time_api
                hours = total_seconds // 3600
                minutes = (total_seconds % 3600) // 60
                print(f"llama运行总时长： {int(hours)}, {int(total_seconds)}秒,{int(minutes)}分钟")
            elif llm_api=='baichuan':
                start_time_api = time.time()
                
                ans=call_baichuan_api(prompt)

                end_time_api = time.time()
                total_seconds = end_time_api - start_time_api
                hours = total_seconds // 3600
                minutes = (total_seconds % 3600) // 60
                print(f"baichuan运行总时长： {int(hours)}, {int(total_seconds)}秒,{int(minutes)}分钟")
            elif llm_api=='glm':
                start_time_api = time.time()

                ans=call_glm_api(prompt)

                end_time_api = time.time()
                total_seconds = end_time_api - start_time_api
                hours = total_seconds // 3600
                minutes = (total_seconds % 3600) // 60
                print(f"glm运行总时长： {int(hours)}, {int(total_seconds)}秒,{int(minutes)}分钟")
            elif llm_api == 'qwen_turbo':
                start_time_api = time.time()
                
                ans=call_qwen_turbo_api(prompt)
                print("qwen_turbo模型 ans: ",ans)
                end_time_api = time.time()
                total_seconds = end_time_api - start_time_api
                hours = total_seconds // 3600
                minutes = (total_seconds % 3600) // 60
                print(f"qwen_turbo运行总时长： {int(hours)}, {int(total_seconds)}秒,{int(minutes)}分钟")
            else:
                start_time_api = time.time()
                #llm_api=='chatgpt'

                ans = call_llm_api(prompt,max_retries=10, timeout=10,api_key=api_key,api_site=api_site)

                end_time_api = time.time()
                total_seconds = end_time_api - start_time_api
                hours = total_seconds // 3600
                minutes = (total_seconds % 3600) // 60
                print(f"chatgpt运行总时长： {int(hours)}, {int(total_seconds)}秒,{int(minutes)}分钟")

            prompt = (f"Answer the following question with float value only. Question: Compute the similarity score between {ans} with {item['answer']}.")
            print("second prompt:",prompt)
        elif(QA_dataset == 'CWQ'):
            # print("cwq item:",item)

            start_time_load = time.time()

            relation_to_keep = item['relations'][0]
            relation_to_keep = relation_to_keep.replace("."," ")
            kg_triplets[i] = prune_knowledge_graph(kg_triplets[i], relation_to_keep)
            kg_triplets[i]= kg_triplets[i][:2]
            kg_triplets_new.append(kg_triplets[i])
            triplets_with_questions_new.append(triplets_with_questions[i])

            end_time_load = time.time()
            total_seconds = end_time_load - start_time_load
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            print(f"CWQ data load运行总时长： {int(hours)}, {int(total_seconds)}秒,{int(minutes)}分钟")


    
            if k>0:
                prompt=(f"Please add some random words no more than {k} percent of the following question"
                        f" Question: {item['question']}"
                        )
                
                start_time_api = time.time()
                noise = call_llm_api(prompt,max_retries=10, timeout=10,api_key=api_key,api_site=api_site)
                end_time_api = time.time()
                total_seconds = end_time_api - start_time_api
                hours = total_seconds // 3600
                minutes = (total_seconds % 3600) // 60
                print(f"CWQ 计算noise运行总时长： {int(hours)}, {int(total_seconds)}秒,{int(minutes)}分钟")

                prompt = (f"Answer the following question with simple answers (less than 5 words) based on the knowledge triplets: '{kg_triplets[i]}', "
                f" Question: {item['question']}"
                f" {noise}"
                )
            else:
                prompt = (f'''knowledge triplets: '[[('large, above average in size or number or quantity or magnitude or extent; "a large city"; "set out for the big city"; "a large sum"; "a big (or large) barn"; "a large family"; "big businesses"; "a big expenditure"; "a large number of newspapers"; "a big group of scientists"; "large areas of the world"', 'derivationally related form', 'largeness, the property of having a relatively great size')]]',  Question: What Portuguese-speaking country's market cap of list companies as percent of gdp was 10.43? Answer:Brazil

                '[[('large, above average in size or number or quantity or magnitude or extent; "a large city"; "set out for the big city"; "a large sum"; "a big (or large) barn"; "a large family"; "big businesses"; "a big expenditure"; "a large number of newspapers"; "a big group of scientists"; "large areas of the world"', 'also see', 'tall, great in vertical dimension; high in stature; "tall people"; "tall buildings"; "tall trees"; "tall ships"')], [('large, above average in size or number or quantity or magnitude or extent; "a large city"; "set out for the big city"; "a large sum"; "a big (or large) barn"; "a large family"; "big businesses"; "a big expenditure"; "a large number of newspapers"; "a big group of scientists"; "large areas of the world"', 'derivationally related form', 'largeness, the property of having a relatively great size')]]',  Question: Where did Drew Brees attend university, that has less than 31,290 undergraduates? Answer: Purdue University
                                '''
                                    f"Answer the following question with simple answers (less than 5 words) based on the knowledge triplets: '{kg_triplets[i]}', "
                                f" Question: {item['question']}"
                                )
                print("first prompt",prompt)
            if llm_api=='llama7b':
                start_time_api = time.time()

                ans=call_llama7b_api(prompt)

                end_time_api = time.time()
                total_seconds = end_time_api - start_time_api
                hours = total_seconds // 3600
                minutes = (total_seconds % 3600) // 60
                print(f"llama运行总时长： {int(hours)}, {int(total_seconds)}秒,{int(minutes)}分钟")


            # elif llm_api=='baichuan':
            #     start_time_api = time.time()

            #     ans=call_baichuan_api(prompt)

            #     end_time_api = time.time()
            #     total_seconds = end_time_api - start_time_api
            #     hours = total_seconds // 3600
            #     minutes = (total_seconds % 3600) // 60
            #     print(f"baichuan运行总时长： {int(hours)}, {int(total_seconds)}秒,{int(minutes)}分钟")

            elif llm_api=='glm':
                start_time_api = time.time()

                ans=call_glm_api(prompt)

                end_time_api = time.time()
                total_seconds = end_time_api - start_time_api
                hours = total_seconds // 3600
                minutes = (total_seconds % 3600) // 60
                print(f"glm运行总时长： {int(hours)}, {int(total_seconds)}秒,{int(minutes)}分钟")
            # elif llm_api == 'qwen_turbo':
            #     start_time_api = time.time()

            #     ans=call_qwen_turbo_api(prompt)
            #     end_time_api = time.time()
            #     total_seconds = end_time_api - start_time_api
            #     hours = total_seconds // 3600
            #     minutes = (total_seconds % 3600) // 60
            #     print(f"qwen_turbo运行总时长： {int(hours)}, {int(total_seconds)}秒,{int(minutes)}分钟")
            else:
                # llm_api=='chatgpt'
                start_time_api = time.time()

                ans = call_llm_api(prompt,max_retries=10, timeout=10,api_key=api_key,api_site=api_site)

                end_time_api = time.time()
                total_seconds = end_time_api - start_time_api
                hours = total_seconds // 3600
                minutes = (total_seconds % 3600) // 60
                print(f"chatgpt运行总时长： {int(hours)}, {int(total_seconds)}秒,{int(minutes)}分钟")


            prompt = (f"Answer the following question with float value only. Question: Compute the similarity score between {ans} with {item['answer']}.")
            print("second prompt:",prompt)
        elif(QA_dataset == 'WebQSP'):

            start_time_load = time.time()
            # print("WebQSP item:",item)
            relation_to_keep = item['relations'][0]
            relation_to_keep = relation_to_keep.replace("."," ")
            kg_triplets[i] = prune_knowledge_graph(kg_triplets[i], relation_to_keep)
            kg_triplets[i]= kg_triplets[i][:2]
            kg_triplets_new.append(kg_triplets[i])
            triplets_with_questions_new.append(triplets_with_questions[i])

            end_time_load = time.time()
            total_seconds = end_time_load - start_time_load
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            print(f"WebQSP data load运行总时长： {int(hours)}, {int(total_seconds)}秒,{int(minutes)}分钟")

            
            if k>0:
                prompt=(f"Please add some random words no more than {k} percent of the following question"
                        f" Question: {item['question']}"
                        )
                
                start_time_api = time.time()
                
                noise = call_llm_api(prompt,max_retries=10, timeout=10,api_key=api_key,api_site=api_site)

                end_time_api = time.time()
                total_seconds = end_time_api - start_time_api
                hours = total_seconds // 3600
                minutes = (total_seconds % 3600) // 60
                print(f"CWQ 计算noise运行总时长： {int(hours)}, {int(total_seconds)}秒,{int(minutes)}分钟")

                prompt = (f"Answer the following question with simple answers (less than 5 words) based on the knowledge triplets: '{kg_triplets[i]}', "
                f" Question: {item['question']}"
                f" {noise}"
                )
            else:
                prompt = (f"Answer the following question with simple answers (less than 5 words) based on the knowledge triplets: '{kg_triplets[i]}', "
                f" Question: {item['question']}"
                )
                print("first prompt:",prompt)
            if llm_api=='llama7b':
                start_time_api = time.time()

                ans=call_llama7b_api(prompt)

                end_time_api = time.time()
                total_seconds = end_time_api - start_time_api
                hours = total_seconds // 3600
                minutes = (total_seconds % 3600) // 60
                print(f"llama运行总时长： {int(hours)}, {int(total_seconds)}秒,{int(minutes)}分钟")


            # elif llm_api=='baichuan':
            #     start_time_api = time.time()

            #     ans=call_baichuan_api(prompt)

            #     end_time_api = time.time()
            #     total_seconds = end_time_api - start_time_api
            #     hours = total_seconds // 3600
            #     minutes = (total_seconds % 3600) // 60
            #     print(f"baichuan运行总时长： {int(hours)}, {int(total_seconds)}秒,{int(minutes)}分钟")

            elif llm_api=='glm':
                start_time_api = time.time()

                ans=call_glm_api(prompt)

                end_time_api = time.time()
                total_seconds = end_time_api - start_time_api
                hours = total_seconds // 3600
                minutes = (total_seconds % 3600) // 60
                print(f"glm运行总时长： {int(hours)}, {int(total_seconds)}秒,{int(minutes)}分钟")
            # elif llm_api == 'qwen_turbo':
            #     start_time_api = time.time()

            #     ans=call_qwen_turbo_api(prompt)
            #     end_time_api = time.time()
            #     total_seconds = end_time_api - start_time_api
            #     hours = total_seconds // 3600
            #     minutes = (total_seconds % 3600) // 60
            #     print(f"qwen_turbo运行总时长： {int(hours)}, {int(total_seconds)}秒,{int(minutes)}分钟")
            else:
                # llm_api=='chatgpt'
                start_time_api = time.time()

                ans = call_llm_api(prompt,max_retries=10, timeout=10,api_key=api_key,api_site=api_site)

                end_time_api = time.time()
                total_seconds = end_time_api - start_time_api
                hours = total_seconds // 3600
                minutes = (total_seconds % 3600) // 60
                print(f"chatgpt运行总时长： {int(hours)}, {int(total_seconds)}秒,{int(minutes)}分钟")
            
            prompt = (f"Answer the following question with float value only. Question: Compute the similarity score between {ans} with {item['answers']}.")
            print("second prompt:",prompt)
        #print('Compute similarity score prompt----------------------------------')
        # print("final prompt: ",prompt)

        start_time_api = time.time()

        score = call_llm_api(prompt,max_retries=10, timeout=10,api_key=api_key,api_site=api_site)

        end_time_api = time.time()
        total_seconds = end_time_api - start_time_api
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        print(f"chatgpt-计算相似度运行总时长： {int(hours)}, {int(total_seconds)}秒,{int(minutes)}分钟")
        
        if is_number(score):
            #答案回答出现非数字或回答失败时，跳过
            #print('!!set the score as 0.0!!')
            #print(score)
            #score= float(0.0)
            scores += [float(score)]

        #break
        try:
            if is_number(score):
                scores.append(float(score))
        except ValueError:
            print(f"Error converting to float: {score}")

        if is_number(score):
            print('----------------------------------')
            print('i: ', i)
            print(ans)
            if(QA_dataset == 'SimpleQA' or QA_dataset == 'CWQ'):
                print(item['answer'])
            elif (QA_dataset == 'WebQSP'):
                print(item['answers'])
            print(score)
            print('----------------------------------')
        else:
            print('Failed to answer')
    
    #return sum(scores) / len(scores) if scores else 0, triplets_with_questions, kg_triplets_new
    
    end_time = time.time()
    total_seconds = end_time - start_time
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    print(f"t1step2运行总时长： {int(hours)}, {int(total_seconds)}秒,{int(minutes)}分钟")

    return sum(scores) / len(scores) if scores else 0, triplets_with_questions_new, kg_triplets_new


# input_file_path = '../preprocess/SimpleQA.json'
# input_file_path = '../preprocess/WebQSP.json'
# input_file_path = '../preprocess/CWQ.json'
# triplets_with_questions, triplets_with_questions_test, train_subgraphs, test_subgraphs = load_data_for_training_and_test(input_file_path)
# print(test_subgraphs[199])
# print(len(test_subgraphs))
# acc = llm_qa(triplets_with_questions,len(triplets_with_questions),test_subgraphs,k=20,llm_api='chatgpt',api_key='sk-viyCtivavwvqvyZ4B71bC4A8DfBf40748820A6129f4fCdA5',api_site='https://oneapi.xty.app/v1/chat/completions')
# print(acc)
import json
import requests
import os
import re
import pickle
import difflib
import random

api_key = 'sk-viyCtivavwvqvyZ4B71bC4A8DfBf40748820A6129f4fCdA5'

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


def call_llm_api(prompt, max_retries=10, timeout=10):
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

    for _ in range(max_retries):
        try:
            response = requests.post('https://oneapi.xty.app/v1/chat/completions', headers=headers, json=data, timeout=timeout)
            
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


# def load_data_for_training_and_test(input_file_path = '../preprocess/SimpleQA.json',dataset='FB15K237'):

#     dataset = dataset
#     print('dataset = ',dataset)
#     hg_entities = pickle.load(open(f'../preprocess/{dataset}_hg_entities.pkl', 'rb'))
#     data = read_json_file(input_file_path)
#     triplets_with_questions = extract_triplets_and_questions(data)

#     # train_len = int(len(hg_entities)*0.8)
#     # print('train_len: ', train_len)
#     train_subgraphs, test_subgraphs = [], []
#     all_subgraphs_dict = pickle.load(open(f'../LLM_tasks/{dataset}_all_subgraphs_dict.pkl', 'rb'))
#     min_len = min(len(triplets_with_questions), len(all_subgraphs_dict))
#     train_len = int(min_len * 0.8)
#     # train_len = int(len(all_subgraphs_dict)*0.8)
#     # train_len = int(len(all_subgraphs_dict)*0.8)
#     print('train_len: ', train_len)
#     for qid, items in all_subgraphs_dict.items():
#         if qid < train_len:
#             train_subgraphs.append(items)
#         else:
#             test_subgraphs.append(items)

#     return triplets_with_questions[:train_len], triplets_with_questions[train_len:len(hg_entities)], train_subgraphs, test_subgraphs

# def load_data_for_training_and_test(input_file_path='../preprocess/SimpleQA.json', dataset='FB15K237'):
#     print('dataset = ', dataset)
#     data = read_json_file(input_file_path)
#     triplets_with_questions = extract_triplets_and_questions(data)

#     all_subgraphs_dict = pickle.load(open(f'../LLM_tasks/{dataset}_all_subgraphs_dict.pkl', 'rb'))

#     # 以较小的数据集长度为基准进行划分
#     min_len = min(len(triplets_with_questions), len(all_subgraphs_dict))
#     train_len = int(min_len * 0.8)

#     print('train_len: ', train_len)

#     train_triplets, test_triplets = triplets_with_questions[:train_len], triplets_with_questions[train_len:min_len]
#     train_subgraphs, test_subgraphs = [], []

#     for qid, items in enumerate(all_subgraphs_dict.values()):
#         if qid < train_len:
#             train_subgraphs.append(items)
#         else:
#             test_subgraphs.append(items)

#     # 确保训练集和测试集的长度一致
#     train_subgraphs = train_subgraphs[:len(train_triplets)]
#     test_subgraphs = test_subgraphs[:len(test_triplets)]

#     return train_triplets, test_triplets, train_subgraphs, test_subgraphs
import random

def load_data_for_training_and_test(input_file_path='../preprocess/SimpleQA.json', dataset='FB15K237'):
    print('dataset = ', dataset)
    data = read_json_file(input_file_path)
    triplets_with_questions = extract_triplets_and_questions(data)

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



def llm_qa(triplets_with_questions, number, kg_triplets):
    if len(triplets_with_questions) < number:
        number = len(triplets_with_questions)
    # Process only the first {number} items
    scores = []
    kg_triplets_new= []
    triplets_with_questions_new = []
    for i, item in enumerate(triplets_with_questions):
        if i >= number:  # Stop after processing 10 items
            break
        # triplet = (item["entity1"], item["relation"], item["entity2"])
        # prune kg
        #print('kg_triplets orginal----------------------------------')
        #print(kg_triplets[i])

        #print('relation_to_keep----------------------------------')
        relation_to_keep = item['relation']
        relation_to_keep = relation_to_keep.replace("_"," ")
        #print(relation_to_keep)
        
        kg_triplets[i] = prune_knowledge_graph(kg_triplets[i], relation_to_keep)
        
        #print('kg_triplets[i] after pruning----------------------------------')
        kg_triplets[i]= kg_triplets[i][:2]
        kg_triplets_new.append(kg_triplets[i])
        triplets_with_questions_new.append(triplets_with_questions[i])
        #print('kg_triplets[i]: ',kg_triplets[i])
        #print('kg_triplets_new after store----------------------------------')
        #print('kg_triplets_new[i]: ',kg_triplets_new)

        


        prompt = (f"Answer the following question with simple answers (less than 5 words) based on the knowledge triplets: '{kg_triplets[i]}', "
                  f" Question: {item['question']}")
        #print('Question prompt----------------------------------')
        #print(prompt)
        
        ans = call_llm_api(prompt)
        prompt = (f"Answer the following question with float value only. Question: Compute the similarity score between {ans} with {item['answer']}.")
        #print('Compute similarity score prompt----------------------------------')
        #print(prompt)
        score = call_llm_api(prompt)
        if is_number(score):
            #答案回答出现非数字或回答失败时，跳过
            #print('!!set the score as 0.0!!')
            #print(score)
            #score= float(0.0)
            if(float(score)<0):
                score=0.0
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
            print(item['answer'])
            print(score)
            print('----------------------------------')
        else:
            print('Failed to answer')
    
    #return sum(scores) / len(scores) if scores else 0, triplets_with_questions, kg_triplets_new
    acc = sum(scores) / len(scores) if scores else 0
    return acc, triplets_with_questions_new, kg_triplets_new


#input_file_path = '../preprocess/SimpleQA.json'
#triplets_with_questions, triplets_with_questions_test, train_subgraphs, test_subgraphs = load_data_for_training_and_test(input_file_path)
#print(test_subgraphs[199])
#print(len(test_subgraphs))
#acc = llm_qa(triplets_with_questions,len(triplets_with_questions),test_subgraphs)

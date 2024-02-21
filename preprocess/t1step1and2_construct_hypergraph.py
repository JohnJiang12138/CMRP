import json
import requests
import os
import re

api_key = 'sk-viyCtivavwvqvyZ4B71bC4A8DfBf40748820A6129f4fCdA5'


def read_json_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)
    
def save_as_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)

def read_jsonl_file(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data

def save_as_jsonl(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        for item in data:
            file.write(json.dumps(item) + '\n')
    
def extract_triplets_and_questions(data):
    triplets_with_questions = []
    for item in data:
        entity1 = list(item["topic_entity"].values())[0]
        relation = item["relation"]
        entity2 = item["answer"]
        related_prompt = item["question"] #对于SimpleQA.json, related user prompt我们认为就是question字段
        triplets_with_questions.append({"entity1": entity1, "relation": relation, "entity2": entity2, "related_prompt": related_prompt})
    return triplets_with_questions


def generate_cot_prompt(related_prompt):
    return f"""
    Analyze the following  main question using the Chain of Thought method: '{related_prompt}' and generate  3 to 5 sub-questions that help in understanding this main question.
    You can focus on breaking down the main question into more specific, targeted sub-questions that address the various dimensions of the topic. This will help in exploring the topic in a more structured and thorough manner. 
    Note: You must only reply in the format of sub-questions, with only numbers and corresponding sub-questions, NO EXTRA REPLIES.

    Main Question: 'How does the Israel-Palestine war influence the global economy?'
    Sub-Questions:
    1. What are the key historical events and current status of the Israel-Palestine War?
    2. How does the Israel-Palestine War affect the oil and gas markets, which are critical components of the global economy?
    3. In what ways does the conflict influence global trade, particularly in regions closely tied to Israel and Palestine?
    4. How do international sanctions and diplomatic relations, altered by the Israel-Palestine War, impact the global economy?
    5. What role do foreign investments and defense spending, influenced by the war, play in the economies of various countries?

    Main Question: "How does climate change impact marine life?"
    Sub-Questions:
    1.What are the primary ways in which rising ocean temperatures affect marine ecosystems?
    2.How does ocean acidification, a result of increased carbon dioxide levels, impact marine species, especially shellfish and coral reefs?
    3.In what ways are changing ocean currents, influenced by climate change, altering marine migration patterns and habitats?
    4.What is the effect of melting polar ice caps on polar marine life, such as polar bears and penguins?
    5.How do changes in marine life due to climate change impact human activities like fishing and tourism?

    Main Question: "What are the implications of artificial intelligence on job markets?"
    Sub-Questions:
    1.Which industries are most likely to be transformed by AI, leading to job creation or displacement?
    2.How does AI affect the skills required for future jobs and the need for education or retraining?
    3.What are the potential economic benefits and challenges posed by AI in terms of productivity and employment?
    4.How might AI exacerbate or mitigate issues of inequality in the job market?
    5.What policies could governments implement to manage the impact of AI on employment?

    Main Question: "What factors contribute to the success of a startup company?"
    Sub-Questions:
    1.How critical is the role of a unique and viable business idea in the success of a startup?
    2.What impact does the startup's location and access to resources have on its growth?
    3.How do market trends and consumer behavior influence a startup's success?
    4.In what ways do financial management and funding opportunities affect the sustainability of a startup?
    5.What is the role of leadership and team dynamics in driving a startup towards success?
    
    Main Question: '{related_prompt}'
    Sub-Questions:
    """

def generate_entities_relations_prompt(sub_questions):
    return f"""
    Based on the sub-questions: {sub_questions}, please answer them, in order to provide more useful information.
    You must only strictly reply in the format of: "sub-question %d:", and then for Entities and Relations of each sub-question, with only numbers and corresponding brief terms, NO EXTRA REPLIES.
    Example Sub-Questions: 
    1. What are the key historical events and current status of the Israel-Palestine War?
    2. How does the Israel-Palestine War affect the oil and gas markets, which are critical components of the global economy?
    3. In what ways does the conflict influence global trade, particularly in regions closely tied to Israel and Palestine?
    Example Entities and Relations:

    sub-question 1:
    Entities:
    1.Israel
    2.Palestine
    3.Israel-Palestine War
    4.Historical Events
    5.Current Status
    Relations:
    1.Historical context of the Israel-Palestine conflict
    2.Key events in the Israel-Palestine War
    3.Current status and developments in the conflict

    sub-question 2:
    Entities:
    1.Oil and Gas Markets
    2.Israel-Palestine War
    3.Global Economy
    Relations:
    1.Impact of the Israel-Palestine War on Oil and Gas Markets
    2.Economic consequences for the Global Economy due to Oil and Gas Market fluctuations

    sub-question 3:
    Entities:
    1.Israel-Palestine Conflict
    2.Global Trade
    3.Regions closely tied to Israel and Palestine
    Relations:
    1.Influence of the Israel-Palestine Conflict on Global Trade
    2.Regional impact of the conflict on trade and economies

    sub-question 4:
    Entities:
    1.International Sanctions
    2.Diplomatic Relations
    3.Global Economy
    4.Israel-Palestine War
    Relations:
    1.Effects of international sanctions on the Global Economy
    2.Impact of altered diplomatic relations on the Global Economy
    3.Connection between the Israel-Palestine War and international sanctions and diplomatic relations
    
    sub-question 5:
    Entities:
    1.Foreign Investments
    2.Defense Spending
    3.Economies of Various Countries
    4.Israel-Palestine War
    Relations:
    1.Influence of foreign investments and defense spending on economies of different countries
    2.Relationship between these economic factors and the Israel-Palestine War
    """

def parse_entities_relations_response(response_text):
    results = {}
    current_sub_question = None

    for line in response_text.split('\n'):
        # 检测子问题的开始
        sub_question_match = re.search(r'sub-question (\d+):', line, re.IGNORECASE)
        if sub_question_match:
            current_sub_question = 'sub-question ' + sub_question_match.group(1)
            results[current_sub_question] = {'Entities': [], 'Relations': []}
            continue

        # 当前在处理的是哪个子问题
        if current_sub_question:
            if 'entities:' in line.lower():
                entity_type = 'Entities'
            elif 'relations:' in line.lower():
                entity_type = 'Relations'
            else:
                # 添加实体或关系到当前子问题
                content = line.strip()
                if content:
                    # 移除前缀的序号
                    content_without_number = re.sub(r'^\d+\.', '', content).strip()
                    results[current_sub_question][entity_type].append(content_without_number)

    return results


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

def triple2prompt(number):
    input_file_path = 'SimpleQA.json'
    output_file_path = 'SimpleQA_triplets_with_prompts.json'
    

    data = read_json_file(input_file_path)
    triplets_with_questions = extract_triplets_and_questions(data)

    # Process only the first 10 items
    processed_data = []
    for i, item in enumerate(triplets_with_questions):
        if i >= number:  # Stop after processing 10 items
            break
        triplet = (item["entity1"], item["relation"], item["entity2"])
        related_prompt = item["related_prompt"]
        # prompt = f"Generate an unrelated question for the following triplet and related question: {triplet}, Related Question: {related_prompt}"
        # unrelated_prompt = call_llm_api(prompt)
        # item["unrelated_prompt"] = unrelated_prompt
        processed_data.append(item)

    save_as_json(triplets_with_questions, output_file_path)

    print(f"三元组及提示已保存到 {output_file_path}")



def is_hypergraph_empty(hypergraph):
    return len(hypergraph) == 0

def find_related_prompt(index, simpleqa_data):
    return simpleqa_data[index]["question"]

def update_incomplete_hypergraphs():
    # 读取超图数据和原始问题数据
    hypergraphs = read_jsonl_file('SimpleQA_hypergraph.jsonl')
    simpleqa_data = read_json_file('SimpleQA.json')

    # 遍历并检查每个超图
    for i, hypergraph in enumerate(hypergraphs):
        if is_hypergraph_empty(hypergraph):  # 检查超图是否完整
            # 查找对应的 related_prompt
            print('incomplete hypergraph:', i, 'of', len(hypergraphs), 'related_prompt:')
            related_prompt = find_related_prompt(i, simpleqa_data)

            # 重新生成子问题
            prompt_step_1 = generate_cot_prompt(related_prompt)
            sub_questions = call_llm_api(prompt_step_1)

            # 解析实体和关系
            prompt_step_2 = generate_entities_relations_prompt(sub_questions)
            entities_relations = call_llm_api(prompt_step_2)

            # 解析并更新超图
            updated_hypergraph = parse_entities_relations_response(entities_relations)
            hypergraphs[i] = updated_hypergraph  # 更新原位置的超图
            print('incomplete hypergraph updated:',i)

    # 保存更新后的超图
    save_as_jsonl(hypergraphs, 'SimpleQA_hypergraph.jsonl')

def construct_hypergraph():
    number_to_process = 1000 #要处理的数据个数,对于SimpleQA，这个值是1000
    # triple2prompt(number_to_process) #从SimpleQA.json提取triple并生成unrelated prompt

    input_file = 'SimpleQA_triplets_with_prompts.json'
    output_file = 'SimpleQA_hypergraph.jsonl'
    triplets_with_prompts = read_json_file(input_file)

    hypergraphs = []

    with open(output_file, 'w') as file:
        for i,item in enumerate(triplets_with_prompts):
            if i >=number_to_process:
                break
            related_prompt = item['related_prompt']
            print('related_prompt:',related_prompt)

            # 生成COT提示并调用API
            prompt_step_1 = generate_cot_prompt(related_prompt)
            sub_questions = call_llm_api(prompt_step_1)
            print('sub_questions:',sub_questions)
            
            # 生成实体和关系提示并调用API
            prompt_step_2 = generate_entities_relations_prompt(sub_questions)
            entities_relations = call_llm_api(prompt_step_2)
            print('entities_relations:',entities_relations)
            
            # Tokenizing entities and relations from the example prompt
            tokenized_output = parse_entities_relations_response(entities_relations)
            hypergraphs.append(tokenized_output)
            print('hypergraph:',tokenized_output)
            # 将每个超图对象作为一行JSON写入文件
            file.write(json.dumps(tokenized_output) + '\n')
    
    #更新不完整超图
    update_incomplete_hypergraphs()
    # update_incomplete_hypergraphs()
    # update_incomplete_hypergraphs()
    # update_incomplete_hypergraphs()



construct_hypergraph()
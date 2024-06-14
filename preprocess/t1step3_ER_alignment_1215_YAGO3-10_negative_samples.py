import pdb
import json
import requests
import os
# os.environ['OPENBLAS_NUM_THREADS'] = '1'
import re
import torch
from torch import nn
import csv
import pickle
from transformers import T5Tokenizer, T5EncoderModel
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Check if CUDA is available and set the device accordingly
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = T5Tokenizer.from_pretrained("../t5-small")
model = T5EncoderModel.from_pretrained("../t5-small").to(device)

max_len = 512

import time, contextlib


class Timing(contextlib.ContextDecorator):
    def __init__(self, prefix=""):
        self.prefix = prefix
        self.st_time = None

    def __enter__(self):
        self.st_time = time.perf_counter_ns()

    def __exit__(self, *etc):
        self.et = time.perf_counter_ns() - self.st_time
        print(f"{self.prefix} {self.et*1e-6:.2f} ms")

def read_file_and_create_dict(file_path):
    d1, d2 = {}, {}
    with open(file_path) as raw_file:
        csv_file = csv.reader(raw_file, delimiter='\t')
        for line in csv_file:
            d1[line[0]] = line[1]
            d2[line[1]] = line[0]
    return d1, d2

def read_jsonl_file(file_path):
    d = []
    with open(file_path, 'r') as file:
        for line in file:
            d += [eval(line)]
    return d

# 假设 tokenizer, model, device 已经定义

def generate_embeddings(entities, tokenizer, model, max_len, batch_size=128):
    # 准备批处理
    batches = [entities[i:i + batch_size] for i in range(0, len(entities), batch_size)]
    all_embeddings = []

    for batch in batches:
        # Tokenization
        encoded_input = tokenizer(batch, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        input_ids = encoded_input['input_ids'].to(device)
        
        # 模型前向传播，注意要将输入移到模型所在的设备
        with torch.no_grad():
            outputs = model(input_ids=input_ids.to(model.device))
        
        # 计算嵌入并保存
        batch_embeddings = outputs.last_hidden_state.mean(dim=1)
        all_embeddings.append(batch_embeddings)

    # 将所有批次的嵌入合并为一个张量
    return torch.cat(all_embeddings, dim=0)

def cosine_similarity_torch(tensor_a, tensor_b, batch_size=100):
    # 计算每个向量的L2范数（模）
    norm_a = torch.linalg.norm(tensor_a, dim=1, keepdim=True)
    norm_b = torch.linalg.norm(tensor_b, dim=1, keepdim=True)

    # 归一化向量（除以其L2范数）
    tensor_a_norm = tensor_a / norm_a
    tensor_b_norm = tensor_b / norm_b

    # 初始化一个空的Tensor用于存储余弦相似度结果
    cosine_sim = torch.zeros((tensor_a_norm.size(0), tensor_b_norm.size(0)))

    # 分批处理计算余弦相似度
    for i in range(0, tensor_a_norm.size(0), batch_size):
        end = i + batch_size
        batch_a = tensor_a_norm[i:end]
        batch_cosine_sim = torch.mm(batch_a, tensor_b_norm.T)  # 计算当前批次与全部tensor_b的余弦相似度
        cosine_sim[i:end] = batch_cosine_sim

    return cosine_sim



def embedding_based_search(query_embeddings, entityid2embeds, threshold=0.1):
    # 将 entityid2embeds 中的 numpy.ndarray 转换为 Tensor，并确保它们在正确的设备上
    # pdb.set_trace()

    embeds_matrix = torch.stack([torch.tensor(embedding).to(device) for embedding in entityid2embeds.values()])
    print('embeds_matrix shape: ', embeds_matrix.shape)
    # 计算 query_embeddings 和 embeds_matrix 的余弦相似度
    # 使用 GPU 计算 cosine similarity
    print('computing cosine similarity...')
    similarities = cosine_similarity_torch(query_embeddings, embeds_matrix)
    print('cosine_similarity calcuated.')
    entitiy_similarities = cosine_similarity_torch(embeds_matrix,embeds_matrix)
    entitiy_similarities = entitiy_similarities.cpu().numpy()
    # 接着，将该数组保存为.pkl文件
    with open('../REINFORCE/YAGO3-10_entity_similarities.pkl', 'wb') as f:
        pickle.dump(entitiy_similarities, f)

    # 加载保存的.pkl文件
    with open('../REINFORCE/YAGO3-10_entity_similarities.pkl', 'rb') as f:
        entity_similarities_loaded = pickle.load(f)

    # 打印加载的数组
    print(entity_similarities_loaded)
    # 假设 entity_similarities_loaded 是您加载的相似度矩阵

    
    # 找到最相似的实体 ID
    # most_similar_ids = torch.argmax(similarities, dim=1).cpu().numpy()
    # similarity_score = torch.max(similarities, dim=1).values.cpu().numpy()
    # most_similar_ids = np.where(similarity_score>threshold, most_similar_ids, -1)

    # 找到最不相似的实体 ID
    most_unsimilar_ids = torch.argmin(similarities, dim=1).cpu().numpy()
    similarity_score = torch.min(similarities, dim=1).values.cpu().numpy()
    most_unsimilar_ids = np.where(similarity_score<threshold, most_unsimilar_ids, 1)

    return most_unsimilar_ids

def embedding_based_search_rel(query_embeddings, entityid2embeds, threshold=0.1):
    # 将 entityid2embeds 中的 numpy.ndarray 转换为 Tensor，并确保它们在正确的设备上
    # pdb.set_trace()

    embeds_matrix = torch.stack([torch.tensor(embedding).to(device) for embedding in entityid2embeds.values()])
    print('embeds_matrix shape: ', embeds_matrix.shape)
    # 计算 query_embeddings 和 embeds_matrix 的余弦相似度
    # 使用 GPU 计算 cosine similarity
    print('computing cosine similarity...')
    similarities = cosine_similarity_torch(query_embeddings, embeds_matrix)
    print('cosine_similarity calcuated.')

    # 找到最相似的实体 ID
    # most_similar_ids = torch.argmax(similarities, dim=1).cpu().numpy()
    # similarity_score = torch.max(similarities, dim=1).values.cpu().numpy()
    # most_similar_ids = np.where(similarity_score>threshold, most_similar_ids, -1)

    # 找到最不相似的实体 ID
    most_unsimilar_ids = torch.argmin(similarities, dim=1).cpu().numpy()
    similarity_score = torch.min(similarities, dim=1).values.cpu().numpy()
    most_unsimilar_ids = np.where(similarity_score<threshold, most_unsimilar_ids, 1)

    return most_unsimilar_ids

def save_entity_and_relation_embeddings(cls=None):
    embeds_dict = {}

    if cls == 'entity':
        for ent_id,ent_text  in tqdm(entityid2text.items()):
            #此处改为了id到text的mapping
            input_ids = tokenizer(ent_text, return_tensors="pt", max_length=max_len, padding='max_length', truncation=True).input_ids.to(device)
            outputs = model(input_ids=input_ids)
            ent_embs = outputs.last_hidden_state.mean(dim=1).squeeze()
            embeds_dict[ent_id] = ent_embs.detach().cpu().numpy()

        pickle.dump(embeds_dict, open('YAGO3-10_entity_embs.pkl', 'wb'))
    elif cls == 'relation':
        for rel_id,rel_text  in tqdm(relationid2text.items()):
            input_ids = tokenizer(rel_text, return_tensors="pt", max_length=max_len, padding='max_length', truncation=True).input_ids.to(device)
            outputs = model(input_ids=input_ids)
            ent_embs = outputs.last_hidden_state.mean(dim=1).squeeze()
            embeds_dict[rel_id] = ent_embs.detach().cpu().numpy()

        pickle.dump(embeds_dict, open('YAGO3-10_relation_embs.pkl', 'wb'))
    else:
        raise ValueError('cls value is wrong')

# def save_entity_and_relation_embeddings(cls=None):
#     embeds_dict = {}

#     if cls == 'entity':
#         for ent_text, ent_id in tqdm(text2entityid.items()):
#             input_ids = tokenizer(ent_text, return_tensors="pt", max_length=max_len, padding='max_length', truncation=True).input_ids.to(device)
#             outputs = model(input_ids=input_ids)
#             ent_embs = outputs.last_hidden_state.mean(dim=1).squeeze()
#             embeds_dict[ent_id] = ent_embs.detach().cpu().numpy()

#         pickle.dump(embeds_dict, open('YAGO3-10_entity_embs.pkl', 'wb'))
#     elif cls == 'relation':
#         for rel_text, rel_id in tqdm(text2relationid.items()):
#             input_ids = tokenizer(rel_text, return_tensors="pt", max_length=max_len, padding='max_length', truncation=True).input_ids.to(device)
#             outputs = model(input_ids=input_ids)
#             ent_embs = outputs.last_hidden_state.mean(dim=1).squeeze()
#             embeds_dict[rel_id] = ent_embs.detach().cpu().numpy()

#         pickle.dump(embeds_dict, open('YAGO3-10_relation_embs.pkl', 'wb'))
#     else:
#         raise ValueError('cls value is wrong')

with Timing("若没有加载实体和关系嵌入，则加载它们"): 
    text2entityid_file = '../knowledge_graph_tasks/embedding_based/benchmarks/YAGO3-10/text2entityid.txt'
    text2relationid_file = '../knowledge_graph_tasks/embedding_based/benchmarks/YAGO3-10/text2relationid.txt'
    text2entityid, entityid2text = read_file_and_create_dict(text2entityid_file)
    text2relationid, relationid2text = read_file_and_create_dict(text2relationid_file)

    if not os.path.exists('YAGO3-10_entity_embs.pkl'):
        save_entity_and_relation_embeddings(cls='entity')
    if not os.path.exists('YAGO3-10_relation_embs.pkl'):
        save_entity_and_relation_embeddings(cls='relation')


    entityid2embeds = pickle.load(open('YAGO3-10_entity_embs.pkl', 'rb'))
    relationid2embeds = pickle.load(open('YAGO3-10_relation_embs.pkl', 'rb'))

    SimpleQA_hypergraph_file = 'SimpleQA_hypergraph_1215.jsonl'
    data = read_jsonl_file(SimpleQA_hypergraph_file)
    print('embs loaded and data loaded.')

with Timing("提取所有实体"): 
    # 提取所有实体
    all_entities = set()
    for question_dict in data:
        for items in question_dict.values():
            all_entities.update(items['Entities'])

with Timing("提取所有关系"): 
    # 提取所有关系
    all_relations = set()
    for question_dict in data:
        for items in question_dict.values():
            all_relations.update(items['Relations'])

with Timing("生成所有实体的嵌入"): 
    # 生成所有实体的嵌入
    entity_embeddings = generate_embeddings(list(all_entities), tokenizer, model, max_len=128, batch_size=128)
    print('entity_embeddings shape: ', entity_embeddings.shape)

with Timing("生成所有关系的嵌入"): 
    # 生成所有关系的嵌入
    relation_embeddings = generate_embeddings(list(all_relations), tokenizer, model, max_len=128, batch_size=128)
    print('relation_embeddings shape: ', relation_embeddings.shape)

with Timing("执行基于实体嵌入的搜索"): 
    # 执行基于实体嵌入的搜索
    print('searching for entities...')
    most_similar_entity_ids = embedding_based_search(entity_embeddings, entityid2embeds,0.85)

with Timing("执行基于关系嵌入的搜索"): 
    # 执行基于关系嵌入的搜索
    print('searching for relations...')
    most_similar_relation_ids = embedding_based_search_rel(relation_embeddings, relationid2embeds,0.5)

with Timing("处理实体搜索结果"): 
    # 处理实体搜索结果
    entity_target_pairs = []
    for entity, similar_id in zip(all_entities, most_similar_entity_ids):
        if similar_id != -1:
            entity_target_pairs.append([entity, list(entityid2embeds.keys())[similar_id]])

with Timing("处理关系搜索结果"): 
    # 处理关系搜索结果
    relation_target_pairs = []
    for relation, similar_id in zip(all_relations, most_similar_relation_ids):
        if similar_id != -1:
            relation_target_pairs.append([relation, list(relationid2embeds.keys())[similar_id]])

# print('Entity Target Pairs:', entity_target_pairs)
# print('Relation Target Pairs:', relation_target_pairs)

def hypergraph_nodes_and_relation_mapping_to_KG(data, entity_target_pairs, relation_target_pairs):
    hg_entities = {}
    hg_relations = {}
    level = 0

    # 创建实体ID到嵌入的映射
    entityid2embeds_tensor = {k: torch.tensor(v).to(device) for k, v in entityid2embeds.items()}
    # 创建关系ID到嵌入的映射
    relationid2embeds_tensor = {k: torch.tensor(v).to(device) for k, v in relationid2embeds.items()}

    for qid, question_dict in tqdm(enumerate(data)):
        hg_entities[qid] = defaultdict(list)
        hg_relations[qid] = defaultdict(list)
        for sub_question, items in question_dict.items():
            # 对于实体
            entity_id_pairs = [(entity, eid) for entity, eid in entity_target_pairs if entity in items['Entities']]
            hg_entities[qid][level] += [[entity, eid, entityid2embeds_tensor[eid]] for (entity, eid) in entity_id_pairs]

            # 对于关系
            relation_id_pairs = [(relation, rid) for relation, rid in relation_target_pairs if relation in items['Relations']]
            hg_relations[qid][level] += [[relation, rid, relationid2embeds_tensor[rid]] for (relation, rid) in relation_id_pairs]
    print(len(hg_entities), len(hg_relations))
    pickle.dump(hg_entities, open('YAGO3-10_hg_entities_negative_samples.pkl', 'wb'))
    pickle.dump(hg_relations, open('YAGO3-10_hg_relations_negative_samples.pkl', 'wb'))


# 调用函数
hypergraph_nodes_and_relation_mapping_to_KG(data, entity_target_pairs, relation_target_pairs)

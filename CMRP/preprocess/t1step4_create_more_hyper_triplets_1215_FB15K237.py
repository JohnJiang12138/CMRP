import json
import requests
import os
import re
import random
import torch
from torch import nn
import csv
import pickle
from transformers import T5Tokenizer, T5EncoderModel
from itertools import permutations, product
from tqdm import tqdm
from collections import defaultdict


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

# {level: [entity, eid, entityid2embeds[eid]}, entityid2embeds[eid] is generated by T5-small
hg_entities = pickle.load(open('FB15K237_hg_entities.pkl', 'rb'))
# {level: [relation, rid, relationid2embeds[eid]}, relationid2embeds[eid] is generated by T5-small
hg_relations = pickle.load(open('FB15K237_hg_relations.pkl', 'rb'))

def create_hyper_triplets():
    entities, relations, triplets = [], [], []
    ent2q = {}
    rel2q = {}
    # print(hg_entities)
    # for qid, q_list in hg_entities.items():
    #     for level, entity_info in q_list.items():
    #         for info in entity_info:
    #             entities.append(info[1])
    #             ent2q[info[1]] = qid
    for qid, q_list in hg_entities.items():
        for level, entity_info in q_list.items():
            for info in entity_info:
                entities.append(info[1])
                # 将实体ID映射到问题ID
                if info[1] not in ent2q:
                    ent2q[info[1]] = [qid]
                else:
                    ent2q[info[1]].append(qid)


    for qid, q_list in hg_relations.items():
        for level, relation_info in q_list.items():
            for info in relation_info:
                relations.append(info[1])
                # 将实体ID映射到问题ID
                if info[1] not in rel2q:
                    rel2q[info[1]] = [qid]
                else:
                    rel2q[info[1]].append(qid)
    ent_pairs = []
    for perm in permutations(zip(entities, entities), 2):
        for prod in product(*perm):
            ent_pairs += [prod]

    ent_pairs = list(set(ent_pairs))

    for rel in relations:
        for ent_pair in ent_pairs:
            ent_pair = list(ent_pair)
            ent_pair.insert(2, rel)
            triplets.append(tuple(ent_pair))

    triplets = list(set(triplets)) 
    triplets = [list(triplet) for triplet in triplets]
    pickle.dump(triplets, open('FB15K237_new_hg_triplets.pkl', 'wb'))
    pickle.dump(ent2q, open('../REINFORCE/FB15K237_new_ent2q_triplets.pkl', 'wb'))
    pickle.dump(rel2q, open('../REINFORCE/FB15K237_new_rel2q_triplets.pkl', 'wb'))
    triplets = pickle.load(open('FB15K237_new_hg_triplets.pkl', 'rb'))

    # load old train file and add them to the hyper triplets
    original_train_data_file = '../knowledge_graph_tasks/embedding_based/benchmarks/FB15K237/org_train2id.txt'
    with open(original_train_data_file) as file:
        file = csv.reader(file, delimiter=' ')
        next(file)
        for line in file:
            h, t, r = line
            triplets.append((h, t, r))

    random.shuffle(triplets)
    # obtain new train file
    train_data_file = '../knowledge_graph_tasks/embedding_based/benchmarks/FB15K237/train2id.txt'
    if os.path.exists(train_data_file):
        os.remove(train_data_file)

    with open(train_data_file, 'w') as f:
        f.write(str(len(triplets))+'\n')
        for triplet in triplets:
            h, t, r = triplet
            f.write("{0} {1} {2}\n".format(h, t, r))

with(Timing("create_hyper_triplets")):
    create_hyper_triplets()
    print('Done.')

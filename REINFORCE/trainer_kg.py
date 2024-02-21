import contextlib
import glob
import math
import os
import pickle
import sys
sys.path.append('../knowledge_graph_tasks/embedding_based/')
import csv
import random
import numpy as np
import scipy.signal
# from tensorboard import TensorBoard
import torch
from torch import nn
import torch.nn.parallel
from torch.autograd import Variable
import kg_models
from train_transe_dataset import load_data_for_training, init_embedding_based_model, train_embedding_based_models#, load_data_for_training_ns
from tqdm import tqdm
from collections import defaultdict
# import cupy as cp

def save_matrix(matrix, filename):
    with open(filename, 'wb') as file:
        pickle.dump(matrix, file)

def load_matrix(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

def load_and_print_pickle(file_path):
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            print("Khop_matrix 内容:")
            print(data)
    except FileNotFoundError:
        print("文件未找到:", file_path)
    except Exception as e:
        print("加载文件时出错:", e)

def read_KG_triplets(dataset,train_data_file):
    train_data_file=train_data_file
    KG_entity_pair = []
    for data_file in [train_data_file]:
        with open(data_file, 'r') as f:
            file = csv.reader(f, delimiter=' ')
            next(file)
            for line in tqdm(file):
                h, t, r = line
                KG_entity_pair.append((h, t))

    with open(f'../knowledge_graph_tasks/embedding_based/benchmarks/{dataset}/entity2id.txt', 'r') as f:
        file = csv.reader(f, delimiter='\t')
        num_of_entities = int(next(file)[0])

    KG_entities = list(range(num_of_entities))

    return KG_entities, KG_entity_pair

# def matrix_power_gpu(KG_matrix, K):
#     # 将矩阵转移到GPU
#     KG_matrix_gpu = cp.array(KG_matrix)

#     # 计算矩阵的K次幂
#     Khop_matrix_gpu = cp.linalg.matrix_power(KG_matrix_gpu, K)

#     # 将结果从GPU转移到CPU，并转换回NumPy数组
#     Khop_matrix = cp.asnumpy(Khop_matrix_gpu)
    
#     return Khop_matrix


def matrix_power_cpu(KG_matrix, K):
    # 计算矩阵的K次幂，直接在CPU上进行
    Khop_matrix = np.linalg.matrix_power(KG_matrix, K)

    return Khop_matrix


def matrix_power_gpu(KG_matrix, K, block_size=10000):
    n = KG_matrix.shape[0]
    Khop_matrix = np.zeros((n, n), dtype=KG_matrix.dtype)

    # 处理每个块
    for i in range(0, n, block_size):
        for j in range(0, n, block_size):
            # 确定块的尺寸
            block_rows = min(block_size, n - i)
            block_cols = min(block_size, n - j)

            # 提取块
            block = KG_matrix[i:i+block_rows, j:j+block_cols]

            # 确保块是方形的（如果需要，用零填充）
            if block_rows != block_cols:
                if block_rows > block_cols:
                    padding = np.zeros((block_rows, block_rows - block_cols), dtype=block.dtype)
                    block = np.hstack((block, padding))
                else:
                    padding = np.zeros((block_cols - block_rows, block_cols), dtype=block.dtype)
                    block = np.vstack((block, padding))

            # 计算块的幂
            block_gpu = cp.array(block)
            block_power_gpu = cp.linalg.matrix_power(block_gpu, K)

            # 截取结果并放回相应位置
            Khop_matrix[i:i+block_rows, j:j+block_cols] = cp.asnumpy(block_power_gpu)[:block_rows, :block_cols]

    return Khop_matrix


def extract_subgraphs_from_KG(KG_entities, KG_entity_pair,datasetname):

    # convert KG into matrix
    KG_matrix = [['0' for _ in range(len(KG_entities))] for _ in range(len(KG_entities))]
    for (ent1, ent2) in KG_entity_pair:
        # print('ent1,ent2:',ent1,ent2)
        ent1, ent2 = int(ent1), int(ent2)
        KG_matrix[ent1][ent2] = '1'
        KG_matrix[ent2][ent1] = '1'
            #构建实体矩阵
    # 示例：使用KG_matrix_np（转换为NumPy数组的邻接矩阵）
    subgraph_tuples = defaultdict(list)
    KG_matrix = np.array(KG_matrix, dtype=int)
    datasetname = datasetname
    paths = []
    for k in range(1,4):
        print(f'not done,{k}_hop_matrix')
        # Khop_matrix = matrix_power_gpu(KG_matrix, k)
        Khop_matrix = matrix_power_gpu(KG_matrix, k)
        sum = np.sum(Khop_matrix)
        print('sum = ', sum)
        paths.append(sum)
        print(f'{k}_hop_matrix',Khop_matrix)
        # save_matrix(Khop_matrix,f'{datasetname}_{k}_hop_matrix.pkl') #save as pickle file 
        # load_and_print_pickle(f'{datasetname}_{k}_hop_matrix.pkl')
        print(f'done,{k}_hop_matrix')
    
    return paths

def convert_to_int(entity):
    try:
        return int(entity)
    except ValueError:
        # 如果转换失败（例如，实体标识符不是纯数字），需要根据你的数据集来决定如何处理
        return None

# 然后，计算 selected_hyper_edges 中实体对的余弦相似度平均值
def calculate_average_similarity(selected_hyper_edges, entity_similarities_loaded):
    total_similarity = 0
    valid_edges = 0

    for edge in selected_hyper_edges:
        entity1, entity2, _ = edge  # 忽略关系，因为我们只关心实体之间的相似度
        entity1_index = convert_to_int(entity1)
        entity2_index = convert_to_int(entity2)

        # 确保转换后的索引是有效的
        if entity1_index is not None and entity2_index is not None:
            similarity = entity_similarities_loaded[entity1_index][entity2_index]
            total_similarity += similarity
            valid_edges += 1

    # 计算平均值
    if valid_edges > 0:
        average_similarity = total_similarity / valid_edges
    else:
        average_similarity = 0  # 防止除以零

    return average_similarity

def get_field_list(array):
    return array.keys()

def is_field_exist(array, field):
    field_list = get_field_list(array)
    return str(field) in field_list

def create_q2ent(ent2q):
    q2ent = {}
    for ent_id, qid_list in ent2q.items():
        for qid in qid_list:
            if qid not in q2ent:
                q2ent[qid] = [ent_id]
            else:
                if ent_id not in q2ent[qid]:
                    q2ent[qid].append(ent_id)
    return q2ent

def calculate_reasoning_rewards(selected_hyper_edges, ent2q):
    reward = 0.0
    reward1 = 0.0
    total = len(selected_hyper_edges)
    print('total selected_hyper_edges = ', total)
    q2ent = create_q2ent(ent2q)
    num_Q = len(q2ent)
    print('number of Questions = ', num_Q)

    for q_idx, ent_list in q2ent.items():  # 这里直接遍历q2ent字典
        for edge in selected_hyper_edges:
            entity1, entity2, _ = edge
            if entity1 in ent_list and entity2 in ent_list:
                reward += 1.0
            if entity1 in ent_list or entity2 in ent_list:
                reward1 += 1.0
                # print('find reasoning reward!')

    # 确保total和num_Q都不为0以避免除以零的错误
    print('reward = ',reward)
    print('reward1 = ',reward1)
    print('total * num_Q = ',total * num_Q)
    if total > 0 and num_Q > 0:
        reward = reward / (total * num_Q)
    else:
        reward = 0.0

    return reward

            # entity1_index = convert_to_int(entity1)
            # entity2_index = convert_to_int(entity2)
    
    # num_E = len(set().union(*ent2q.keys()))
    # entities = set().union(*ent2q.keys())
    # num_Q = len(set().union(*ent2q.values()))  # 获取唯一问题的数量
    # questions = set().union(*ent2q.values())
    # print('num_E = ',num_E)
    # print('num_Q = ',num_Q)
    # for edge in selected_hyper_edges:
    #         entity1, entity2, _ = edge
    #         entity1_index = convert_to_int(entity1)
    #         entity2_index = convert_to_int(entity2)
    #         if ()
    # for q_i in range(num_Q):  # 遍历所有问题
    #     for edge in selected_hyper_edges:
    #         entity1, entity2, _ = edge
    #         entity1_index = convert_to_int(entity1)
    #         entity2_index = convert_to_int(entity2)

    #         if is_field_exist(ent2q, entity1_index) and is_field_exist(ent2q, entity2_index):
    #             # 检查实体ID对应的问题ID列表中是否包含当前问题ID
    #             if q_i in ent2q[str(entity1_index)] or q_i in ent2q[str(entity2_index)]:
    #                 reward += 1.0
    #                 print('find reasoning reward!')
    #                 break

    # print('reward = ', reward)
    # print('len(ent2q) = ', len(ent2q))
    # reward /= num_Q  # 使用问题的总数来计算平均奖励

    # return reward
# def get_field_list(array):
#     return array.keys()

# def is_field_exist(array, field):
#     field_list = get_field_list(array)
#     return str(field) in field_list

# def calculate_reasoning_rewards(selected_hyper_edges, ent2q):
#     reward = 0.0
#     num_Q = ent2q.values()
#     # num_Q = len(ent2q)
#     # print(self.hyper_edges)
#     # print('===================')
#     # print(selected_hyper_edges)
#     # for q_i in range(num_Q):
#     for q_i in num_Q:
#         for edge in selected_hyper_edges:
#             entity1, entity2, _ = edge  # 忽略关系，因为我们只关心实体之间的相似度
#             entity1_index = convert_to_int(entity1)
#             entity2_index = convert_to_int(entity2)
#             # if(entity1_index>num_Q or entity2_index>num_Q): continue
#             # print('ent2q',ent2q)
#             # print('len(ent2q):',len(ent2q))
#             # print('entity1_index:',entity1_index)
#             # print('entity2_index:',entity2_index)
#             # print('q_i:', q_i)
#             if (is_field_exist(ent2q, entity1_index) and is_field_exist(ent2q, entity2_index)):
#                 if ent2q[str(entity1_index)] == q_i or ent2q[str(entity2_index)] == q_i:
#                     reward += 1.0
#                     print('find reasoning reward!')
#                     print('==========================')
#                     break
#     print('reward = ',reward)
#     print('len(ent2q) = ',len(ent2q))
#     reward /= len(ent2q)

#     return reward

def update_KG_edges(triplets,args):
    datasetname = args.datasetname
    # load old train file and add them to the hyper triplets
    original_train_data_file = f'../knowledge_graph_tasks/embedding_based/benchmarks/{datasetname}/org_train2id.txt'
    with open(original_train_data_file) as file:
        file = csv.reader(file, delimiter=' ')
        next(file)
        for line in file:
            h, t, r = line
            triplets.append((h, t, r))

    random.shuffle(triplets)

    # obtain new train file
    train_data_file = f'../knowledge_graph_tasks/embedding_based/benchmarks/{datasetname}/train2id.txt'
    if os.path.exists(train_data_file):
        os.remove(train_data_file)

    with open(train_data_file, 'w') as f:
        f.write(str(len(triplets)) + '\n')
        for triplet in triplets:
            h, t, r = triplet
            f.write("{0} {1} {2}\n".format(h, t, r))


def update_KG_edges_ns(triplets,args):
    datasetname = args.datasetname
    # load old train file and add them to the hyper triplets
    original_train_data_file = f'../knowledge_graph_tasks/embedding_based/benchmarks/{datasetname}/org_train2id.txt'
    with open(original_train_data_file) as file:
        file = csv.reader(file, delimiter=' ')
        next(file)
        for line in file:
            h, t, r = line
            triplets.append((h, t, r))

    random.shuffle(triplets)

    # obtain new train file
    train_data_file = f'../knowledge_graph_tasks/embedding_based/benchmarks/{datasetname}/train2id_ns.txt'
    if os.path.exists(train_data_file):
        os.remove(train_data_file)

    with open(train_data_file, 'w') as f:
        f.write(str(len(triplets)) + '\n')
        for triplet in triplets:
            h, t, r = triplet
            f.write("{0} {1} {2}\n".format(h, t, r))


def to_item(x):
    """Converts x, possibly scalar and possibly tensor, to a Python scalar."""
    if isinstance(x, (float, int)):
        return x

    if float(torch.__version__[0:3]) < 0.4:
        assert (x.dim() == 1) and (len(x) == 1)
        return x[0]

    return x.item()


def _get_optimizer(name):
    if name.lower() == 'sgd':
        optim = torch.optim.SGD
    elif name.lower() == 'adam':
        optim = torch.optim.Adam

    return optim


def discount(x, amount):
    return scipy.signal.lfilter([1], [1, -amount], x[::-1], axis=0)[::-1]


class Trainer(object):
    """A class to wrap training code."""
    def __init__(self, args, hyper_edges, hyper_edges_ns):
        """Constructor used to select new KG edges from hypergraph."""

        self.args = args
        self.hyper_edges = hyper_edges
        self.hyper_edges_ns = hyper_edges_ns
        # if args.use_tensorboard:
        #     self.tb = TensorBoard(args.model_dir)
        # else:
        #     self.tb = None

        self.train_dataloader, self.test_dataloader = load_data_for_training(args)

        #self.train_dataloader_ns, self.test_dataloader_ns = load_data_for_training_ns(args)
        
        self.model, self.negative_sample_method = init_embedding_based_model(self.args,self.train_dataloader)
        
        self.build_model()

        #self.ent2q = pickle.load(open('../preprocess/FB15K237_new_ent2q_triplets.pkl', 'rb'))

        # with open(f'../preprocess/{self.args.datasetname}_new_ent2q_triplets.pkl', 'rb') as f:
        #     self.ent2q = pickle.load(f)


        self.agent_lr = self.args.agent_lr

        agent_optimizer = _get_optimizer(self.args.agent_optim)

        self.agent_optim = agent_optimizer(
        self.agent.parameters(),
        lr=self.args.agent_lr)

        # self.ce = nn.CrossEntropyLoss()

    def build_model(self):
        """Creates and initializes the shared and controller models."""
        self.agent = kg_models.Agent(self.args, self.hyper_edges, self.hyper_edges_ns, self.model)

        if self.args.num_gpu >= 1:
            self.agent.to(self.args.device)

    def get_reward(self, R, entropies):
        """Computes the perplexity of a single sampled model on a minibatch of
        validation data.
        """
        if not isinstance(entropies, np.ndarray):
            entropies = entropies.data.cpu().numpy()

        if self.args.entropy_mode == 'reward':
            rewards = R + self.args.entropy_coeff * entropies
        elif self.args.entropy_mode == 'regularizer':
            rewards = R * np.ones_like(entropies)
        else:
            raise NotImplementedError(f'Unkown entropy mode: {self.args.entropy_mode}')

        return rewards

    def _summarize_agent_train(self,
                                    total_loss,
                                    advantage_history,
                                    entropy_history,
                                    reward_history,
                                    avg_reward_base,
                                    epoch):
        """Logs the agent's progress for this training epoch."""
        cur_loss = total_loss / self.args.log_epoch

        avg_advantage = np.mean(advantage_history)
        avg_entropy = np.mean(entropy_history)
        avg_reward = np.mean(reward_history)

        if avg_reward_base is None:
            avg_reward_base = avg_reward

        print(
            f'| epoch {epoch:3d} | lr {self.agent_lr:.5f} '
            f'| R {avg_reward:.5f} | entropy {avg_entropy:.4f} '
            f'| loss {cur_loss:.5f}')

        # """Tensorboard"""
        # if self.tb is not None:
        #     self.tb.scalar_summary('controller/loss',
        #                            cur_loss,
        #                            epoch)
        #     self.tb.scalar_summary('controller/reward',
        #                            avg_reward,
        #                            epoch)
        #     self.tb.scalar_summary('controller/reward-B_per_epoch',
        #                            avg_reward - avg_reward_base,
        #                            epoch)
        #     self.tb.scalar_summary('controller/entropy',
        #                            avg_entropy,
        #                            epoch)
        #     self.tb.scalar_summary('controller/advantage',
        #                            avg_advantage,
        #                            epoch)


    def train_per_epoch(self, epoch):
        print("Start Training")
        # self.agent.train()
        args = self.args



        
        # best_metric = None  # 用于存储迄今为止的最佳指标
        
        
        epoch = int(epoch)

        if epoch == 0:
            
            self.avg_reward_base = None
            self.baseline = None
            self.advantage_history = []
            self.entropy_history = []
            self.reward_history = []

            self.total_loss = 0
            
            # if(self.args.specific_task == 'link_prediction'):
            print('Start training link prediction task')
            mrr, mr, hit10, hit3, hit1 = train_embedding_based_models(self.model, self.negative_sample_method,
                                                                        self.train_dataloader, self.test_dataloader,args=self.args)
            print(
                    f'| epoch {epoch:3d} | mrr {mrr:4f} | mr {mr:5f} | hit10 {hit10:5f} |'
                    f'hit3 {hit3:5f} | hit1 {hit1:5f} ')

            # dataset = self.args.datasetname
            # train_data_file=f'../knowledge_graph_tasks/embedding_based/benchmarks/{dataset}/org_train2id.txt'
            # KG_entities,KG_entity_pair = read_KG_triplets(dataset,train_data_file)
            # org_paths = extract_subgraphs_from_KG(KG_entities, KG_entity_pair, dataset)
            return
        if(epoch == 1 and self.args.method == 'finetune'):
            self.args.train_times = int(self.args.train_times / 5)
        # sample hyper-edges
        self.log_probs, self.entropies, selected_hyper_edges = self.agent.sample()
        self.log_probs_ns, self.entropies_ns, selected_hyper_edges_ns = self.agent.sample_ns()

        benificial_edges = set()  # 存储有益的三元组
        benificial_edges.update(tuple(edge) for edge in selected_hyper_edges)
        benificial_edges = list(benificial_edges)
        # 保存 benificial_edges 到文件
        datasetname = self.args.datasetname
        filename = f'./hyperedges/{datasetname}_benificial_edges_epoch_{epoch}.pkl'
        with open(filename, 'wb') as file:
            pickle.dump(list(benificial_edges), file)  # 将set转换为list再保存

        print(f"benificial_edges saved to {filename}")

        with open(f'./{datasetname}_entity_similarities.pkl', 'rb') as f:
            entity_similarities_loaded = pickle.load(f)
        # print('entity_similarities_loaded = ',entity_similarities_loaded)
        # print('benificial = ',benificial_edges)
        semantic_similarity = calculate_average_similarity(list(benificial_edges), entity_similarities_loaded)
        #ent2q = pickle.load(open('./{datasetname}_new_q2ent_triplets.pkl', 'rb'))
        
        with open(f'./{datasetname}_new_ent2q_triplets.pkl', 'rb') as f:
            ent2q = pickle.load(f)
        
        reasoning_reward = calculate_reasoning_rewards(list(benificial_edges), ent2q)
        #print('!!! average_similarity = ',average_similarity)
        # update original KG edges
        update_KG_edges(selected_hyper_edges,args=self.args)
        if(self.args.method == 'scratch'):
            self.train_dataloader, self.test_dataloader = load_data_for_training(args)
            self.model, self.negative_sample_method = init_embedding_based_model(self.args,self.train_dataloader)


        # train_data_file=f'../knowledge_graph_tasks/embedding_based/benchmarks/{dataset}/train2id.txt'
        # KG_entities,KG_entity_pair = read_KG_triplets(dataset,train_data_file)
        # now_paths = extract_subgraphs_from_KG(KG_entities, KG_entity_pair, dataset)
        # weights = [self.args.alpha1, self.args.beta1, self.args.gama1]
        # connectivity = 0.0
        # for i in range(0,3):
        #     enhancement = (now_paths[i] - org_paths[i]) / org_paths[i]
        #     connectivity += enhancement * weights[i]
        #     print(now_paths[i])
        #     print(org_paths[i])
        # print('connectivity result = ',connectivity)

        mrr, mr, hit10, hit3, hit1 = train_embedding_based_models(self.model, self.negative_sample_method, self.train_dataloader, self.test_dataloader,args = self.args)
        
        # # 计算当前epoch的指标
        # current_metric = hit1  # 或者是你想比较的其他指标
        # # 检查当前指标是否优于之前的最佳指标
        # if best_metric is None or current_metric > best_metric:
        #     best_metric = current_metric
        #     # 将列表中的每个子列表转换为元组，然后添加到集合中
        #     benificial_edges.update(tuple(edge) for edge in selected_hyper_edges)
        print(
                f'| epoch {epoch:3d} | mrr {mrr:4f} | mr {mr:5f} | hit10 {hit10:5f} |'
                f'hit3 {hit3:5f} | hit1 {hit1:5f} ')
        

        ## negative sample
        harmful_edges = set()  # 存储有害的三元组
        harmful_edges.update(tuple(edge) for edge in selected_hyper_edges_ns)
        harmful_edges = list(harmful_edges)
        # 保存 harmful_edges 到文件
        datasetname = self.args.datasetname
        filename = f'./hyperedges/{datasetname}_harmful_edges_epoch_{epoch}.pkl'
        with open(filename, 'wb') as file:
            pickle.dump(list(harmful_edges), file)  # 将set转换为list再保存

        print(f"harmful_edges saved to {filename}")

        # with open(f'./{datasetname}_entity_similarities.pkl', 'rb') as f:
        #     entity_similarities_loaded = pickle.load(f)
        # print('entity_similarities_loaded = ',entity_similarities_loaded)
        # print('harmful_edges = ',harmful_edges)
        semantic_similarity_ns = calculate_average_similarity(list(harmful_edges), entity_similarities_loaded)
        #ent2q = pickle.load(open('./{datasetname}_new_q2ent_triplets.pkl', 'rb'))
        
        # with open(f'./{datasetname}_new_ent2q_triplets.pkl', 'rb') as f:
        #     ent2q = pickle.load(f)
        
        reasoning_reward_ns = calculate_reasoning_rewards(list(harmful_edges), ent2q)
        #print('!!! average_similarity = ',average_similarity)
        # update original KG edges
        update_KG_edges(selected_hyper_edges_ns,args=self.args)
        
        if(self.args.method == 'scratch'):
            self.train_dataloader, self.test_dataloader = load_data_for_training(args)
            self.model, self.negative_sample_method = init_embedding_based_model(self.args,self.train_dataloader)

        mrr_ns, mr_ns, hit10_ns, hit3_ns, hit1_ns = train_embedding_based_models(self.model, self.negative_sample_method, self.train_dataloader, self.test_dataloader,args = self.args)

        np_entropies = self.entropies.data.cpu().numpy()
        self.entropy_history.extend(np_entropies)

        alpha = self.args.alpha
        alpha1 = self.args.alpha1
        beta = self.args.beta
        gamma = self.args.gamma
        beta1 = self.args.beta1
        self.rewards = self.get_reward(R=mrr * alpha + hit1 *beta1 +
                                       + semantic_similarity * beta 
                                       + alpha1 * reasoning_reward 
                                       - semantic_similarity_ns * gamma 
                                       - alpha1 * reasoning_reward_ns
                                       - mrr_ns * beta - hit1_ns * beta1
                                       , entropies=np_entropies)
        #self.rewards = self.get_reward(R=mrr * alpha + semantic_similarity * beta, entropies=np_entropies)
        print(
                f'| epoch {epoch:3d} | mrr {mrr:4f} | semantic_similarity {semantic_similarity:5f} | reasoning_reward {reasoning_reward:5f} |'
                f'mrr_ns {mrr_ns:5f} | semantic_similarity_ns {semantic_similarity_ns:5f} | reasoning_reward_ns {reasoning_reward_ns:5f}')
        
        print('self.rewards的均值 = ', np.mean(self.rewards))
        print('self.rewards的最大值 = ', np.max(self.rewards))
        print('self.rewards的中位数 = ', np.median(self.rewards))
        
    def agent_update(self, epoch):

        rewards = self.rewards
        # elif(self.args.specific_task == 'node_classification'):
        #     rewards = self.get_reward(R=acc, entropies=np_entropies)

        # discount
        if 1 > self.args.discount > 0:
            rewards = discount(rewards, self.args.discount)

        self.reward_history.extend(rewards)

        # moving average baseline
        if self.baseline is None:
            baseline = rewards
        else:
            decay = self.args.ema_baseline_decay
            baseline = decay * self.baseline + (1 - decay) * rewards

        advantage = rewards - baseline
        self.advantage_history.extend(advantage)

        # policy loss
        loss = -self.log_probs
        if self.args.entropy_mode == 'regularizer':
            loss = loss - self.args.entropy_coeff * self.entropies

        loss = loss.sum()  # or loss.mean()
        # update
        self.agent_optim.zero_grad()
        loss.backward(retain_graph=True)

        if self.args.agent_grad_clip > 0:
            torch.nn.utils.clip_grad_norm(self.agent.parameters(),
                                          self.args.agent_grad_clip)
        self.agent_optim.step()

        self.total_loss = self.total_loss + to_item(loss.data)

        if ((epoch % self.args.log_epoch) == 0) and (epoch > 0):
            self._summarize_agent_train(self.total_loss,
                                         self.advantage_history,
                                         self.entropy_history,
                                         self.reward_history,
                                         self.avg_reward_base,
                                        epoch)

            self.reward_history, self.advantage_history, self.entropy_history = [], [], []
            self.total_loss = 0
    #     benificial_edges = list(benificial_edges)
    #     datasetname = self.args.datasetname
    #    # 保存 benificial_edges 到文件
    #     datasetname = self.args.datasetname
    #     filename = f'../preprocess/{datasetname}_benificial_edges.pkl'
    #     with open(filename, 'wb') as file:
    #         pickle.dump(list(benificial_edges), file)  # 将set转换为list再保存

    #     print(f"benificial_edges saved to {filename}")


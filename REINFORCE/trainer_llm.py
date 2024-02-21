import sys
sys.path.append('../LLM_tasks')
import contextlib
import glob
import math
import os
import csv
import random
import numpy as np
import scipy.signal
# from tensorboard import TensorBoard
import torch
from torch import nn
import torch.nn.parallel
from torch.autograd import Variable
import llm_models
from LLM_tasks.t1step2_llm_evaluate import llm_qa, load_data_for_training_and_test



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
    def __init__(self, args, hyper_edges):
        """Constructor used to select new KG edges from hypergraph."""

        self.args = args

        # if args.use_tensorboard:
        #     self.tb = TensorBoard(args.model_dir)
        # else:
        #     self.tb = None
        # self.train_dataloader, self.test_dataloader, self.train_subgraphs, self.test_subgraphs = load_data_for_training_and_test()
        self.train_dataloader, self.test_dataloader, self.train_subgraphs, self.test_subgraphs = load_data_for_training_and_test(input_file_path = '../preprocess/SimpleQA.json',dataset = self.args.datasetname)
        print('len(self.train_dataloader) = ',len(self.train_dataloader))
        print('len(self.train_subgraphs) =', len(self.train_subgraphs))
        print('len(self.test_dataloader) = ', len(self.test_dataloader))
        print('len(self.test_subgraphs) = ',len(self.test_subgraphs))
        # assert len(self.train_dataloader) == len(self.train_subgraphs)
        # assert len(self.test_dataloader) == len(self.test_subgraphs)

        self.build_model()

        self.agent_lr = self.args.agent_lr

        agent_optimizer = _get_optimizer(self.args.agent_optim)

        self.agent_optim = agent_optimizer(
        self.agent.parameters(),
        lr=self.args.agent_lr)

        # self.ce = nn.CrossEntropyLoss()

    def build_model(self):
        """Creates and initializes the shared and controller models."""
        self.agent = llm_models.Agent(self.args)

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


        # self.agent.train()



        # for epoch in range(self.args.max_epoch):
        #     epoch = int(epoch)

        if epoch == 0:
            self.avg_reward_base = None
            self.baseline = None
            self.advantage_history = []
            self.entropy_history = []
            self.reward_history = []

            self.total_loss = 0
            print('len(self.train_dataloader)',len(self.train_dataloader))
            print('len(self.train_subgraphs)',len(self.train_subgraphs))
            acc, self.triplets_with_questions_new, self.kg_triplets_pruned = llm_qa(triplets_with_questions=self.train_dataloader, number=10, kg_triplets=self.train_subgraphs)
            print(f'| epoch {epoch:3d} | accuracy {acc:4f}')
            return

        # sample triplets for each question
        #log_probs, entropies, q_triplets = self.agent.sample(self.train_dataloader, self.train_subgraphs)
        print('---------call agent---------')
        print('size of self.train_subgraphs')
        print(sys.getsizeof(self.train_subgraphs))
        print('size of kg_triplets_pruned')
        print(sys.getsizeof(self.kg_triplets_pruned))

        print('size of self.train_dataloader')
        print(sys.getsizeof(self.train_dataloader))
        print('size of triplets_with_questions_new')
        print(sys.getsizeof(self.triplets_with_questions_new))

        self.log_probs, self.entropies, self.q_triplets = self.agent.sample(self.triplets_with_questions_new, self.kg_triplets_pruned)
        self.acc, self.triplets_with_questions_new,self.kg_triplets_pruned = llm_qa(triplets_with_questions=self.triplets_with_questions_new, number=10, kg_triplets=self.q_triplets)
        print('======== acc =========')
        print(self.acc)

        # calculate reward
        self.np_entropies = self.entropies.data.cpu().numpy()
        # 需要加上beneficial edge
        '''
        # 保存 benificial_edges 到文件
        datasetname = self.args.datasetname
        filename = f'./hyperedges/{datasetname}_benificial_edges_epoch_{epoch}.pkl'
        with open(filename, 'wb') as file:
            pickle.dump(list(benificial_edges), file)  # 将set转换为list再保存

        print(f"benificial_edges saved to {filename}")

        with open(f'./{datasetname}_entity_similarities.pkl', 'rb') as f:
            entity_similarities_loaded = pickle.load(f)
        
        semantic_similarity = calculate_average_similarity(list(benificial_edges), entity_similarities_loaded)
        #reasoning_reward = calculate_reasoning_rewards(list(benificial_edges), ent2q)
        '''

        self.rewards = self.get_reward(R=self.acc, entropies=self.np_entropies)

    def agent_update(self, epoch):

        rewards = self.rewards
        # discount
        if 1 > self.args.discount > 0:
            rewards = discount(rewards, self.args.discount)

        self.reward_history.extend(rewards)
        self.entropy_history.extend(self.np_entropies)

        # moving average baseline
        if self.baseline is None:
            self.baseline = np.sum(rewards)
        else:
            decay = self.args.ema_baseline_decay
            print('self.baseline:',self.baseline)
            print('rewards:',rewards)
            self.baseline = decay * np.sum(self.baseline) + (1 - decay) * np.sum(rewards)

        advantage = rewards - self.baseline
        self.advantage_history.extend(advantage)

        # policy loss
        loss = -self.log_probs
        if self.args.entropy_mode == 'regularizer':
            loss -= self.args.entropy_coeff * self.entropies

        loss = loss.sum()  # or loss.mean()
        # update
        self.agent_optim.zero_grad()
        # loss.backward(retain_graph=True)
        try:
            loss.backward(retain_graph=True)
        except RuntimeError as e:
            print("Error during loss.backward():", e)
            pass

        if self.args.agent_grad_clip > 0:
            torch.nn.utils.clip_grad_norm(self.agent.parameters(),
                                          self.args.agent_grad_clip)
        self.agent_optim.step()

        self.total_loss += to_item(loss.data)

        if ((epoch % self.args.log_epoch) == 0) and (epoch > 0):
            self._summarize_agent_train(self.total_loss,
                                         self.advantage_history,
                                         self.entropy_history,
                                         self.reward_history,
                                         self.avg_reward_base,
                                        epoch)

            self.reward_history, self.advantage_history, self.entropy_history = [], [], []
            self.total_loss = 0

        if ((epoch % self.args.llm_test_epoch) == 0) and (epoch > 0):
            self.test(epoch)


    def test(self, epoch):

        with torch.no_grad():
            print('len(self.test_dataloader)',len(self.test_dataloader))
            print('len(self.test_subgraphs)',len(self.test_subgraphs))
            _, _, q_triplets = self.agent.sample(self.test_dataloader, self.test_subgraphs)
            acc,_,_ = llm_qa(triplets_with_questions=self.test_dataloader, number=10, kg_triplets=q_triplets)
            print(f'| epoch {epoch:3d} | Testing accuracy {acc:4f}')

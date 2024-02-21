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
from collections import defaultdict
from torch.autograd import Variable
import llm_models
from LLM_tasks.t1step2_llm_evaluate import llm_qa, load_data_for_training_and_test




def joint_train(llm_trainee, gnn_trainee):

    ## LLM
    
    # llm_trainee.agent.train()

    llm_trainee.llm_avg_reward_base = None
    llm_trainee.llm_baseline = None
    llm_trainee.llm_advantage_history = []
    llm_trainee.llm_entropy_history = []
    llm_trainee.llm_reward_history = []
    llm_trainee.llm_total_loss = 0

    ## GNN
    
    gnn_trainee.agent.train()

    gnn_trainee.gnn_avg_reward_base = None
    gnn_trainee.gnn_baseline = None
    gnn_trainee.gnn_advantage_history = []
    gnn_trainee.gnn_entropy_history = []
    gnn_trainee.gnn_reward_history = []
    gnn_trainee.gnn_total_loss = 0
    gnn_trainee.gnn_best_metric = None
    gnn_trainee.gnn_beneficial_edges_dict = defaultdict(list)  # 用于存储有益的边缘
    gnn_trainee.gnn_current_edges_dict = defaultdict(list)  # 用于存储当前epoch选中的边


    for epoch in range(gnn_trainee.args.max_epoch):
        epoch = int(epoch)
        llm_trainee.train_per_epoch(epoch)
        gnn_trainee.train_per_epoch(epoch)

        if epoch != 0:

            # calculate llm reward
            llm_rewards = llm_trainee.rewards
            # calculate gnn reward
            gnn_rewards = gnn_trainee.rewards

            llm_rewards = llm_rewards + np.mean(gnn_rewards)
            gnn_rewards = gnn_rewards + np.mean(llm_rewards)

            llm_trainee.rewards = llm_rewards
            gnn_trainee.rewards = gnn_rewards
            # rewards = reward_shaping()
            llm_trainee.agent_update(epoch)
            gnn_trainee.agent_update(epoch)
import collections
import os
import pickle
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import defaultdict
from transformers import T5Tokenizer, T5EncoderModel


tokenizer = T5Tokenizer.from_pretrained("../t5-small")
encoder_model = T5EncoderModel.from_pretrained("../t5-small")
max_len = 10
t5_hidden_dim=512
class keydefaultdict(defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        else:
            ret = self[key] = self.default_factory(key)
            return ret

def get_variable(inputs, cuda=False, **kwargs):
    if type(inputs) in [list, np.ndarray]:
        inputs = torch.Tensor(inputs)
    if cuda:
        out = Variable(inputs.cuda(), **kwargs)
    else:
        out = Variable(inputs, **kwargs)
    return out


class Agent(torch.nn.Module):

    def __init__(self, args):
        torch.nn.Module.__init__(self)
        self.args = args
        self.selected_edges = []
        self.encoder_model = T5EncoderModel.from_pretrained("../t5-small")
        self.linear_mapping = nn.Linear(3*t5_hidden_dim*max_len, 3*args.agent_dim)
        if self.args.network_type.lower() == 'lstm':
            self.lstm = torch.nn.LSTMCell(3*args.agent_dim, args.agent_dim)

        self.decoders = []
        for idx, size in enumerate([args.agent_dim, 2]):
            decoder = torch.nn.Linear(args.agent_dim, size)
            self.decoders.append(decoder)

        self.decoders = torch.nn.ModuleList(self.decoders)


        self.reset_parameters()

        self.static_init_hidden = keydefaultdict(self.init_hidden)

    
    
    def get_text_embeddings(self, text):
        input_ids = tokenizer(text, return_tensors="pt",
                              max_length=max_len,
                              padding='max_length',
                              truncation=True, ).input_ids
        outputs = encoder_model(input_ids=input_ids)
        ent_embs = outputs.last_hidden_state
        return ent_embs

    def init_hidden(self, batch_size):
        zeros = torch.zeros(batch_size, self.args.agent_dim)
        return (get_variable(zeros, self.args.cuda, requires_grad=False).to(self.args.device),
                get_variable(zeros.clone(), self.args.cuda, requires_grad=False).to(self.args.device))

    def reset_parameters(self):
        init_range = 0.1
        for param in self.parameters():
            param.data.uniform_(-init_range, init_range)
        for decoder in self.decoders:
            decoder.bias.data.fill_(0)

    def forward(self,
                input_emb,
                hidden):

        # print(input_emb.size(), hidden[0].size())
        input_emb = self.linear_mapping(input_emb.view(1, -1))
        hx, cx = self.lstm(input_emb, hidden)
        for decoder in self.decoders:
            logits = decoder(hx)

        logits /= self.args.softmax_temperature

        # exploration
        if self.args.mode == 'train':
            logits = (self.args.tanh_c*F.tanh(logits))

        return logits, (hx, cx)

    def sample(self, dataloader, subgraphs, batch_size=1):
        """Samples a set of hyper-edge from the agent, where each hyper-edge is made up of an activation function.
        """
        if batch_size < 1:
            raise Exception(f'Wrong batch_size: {batch_size} < 1')

        
        # [B, L, H]
        hidden = self.static_init_hidden[batch_size]

       
        entropies = []
        log_probs = []
        q_triplets = defaultdict(list)
        # The LSTM agent alternately outputs an activation,
        # followed by a previous hyper-edge.
        # print('===sample==')
        # print(subgraphs)
        # print(len(dataloader))
        # keys = subgraphs.keys()
        # for q_idx in keys:
        for q_idx in range(len(dataloader)):
            try:
                subgraph = subgraphs[q_idx]
            except:
                print('q_idx = ',q_idx,'skipped.')
                continue
            for subgraph in subgraphs[q_idx]:
                #print(subgraph)
                #print('len(subgraph):',len(subgraph))
                #print(subgraph[0])
                #print(type(subgraph))
                #print(subgraph[2])
                #print(subgraph[3])
                if len(subgraph)!=1:
                    # print(subgraph)
                    # print('len(subgraph):',len(subgraph))
                    # print(type(subgraph))
                    continue
                for (h, t, r) in subgraph:
                    input_embs = torch.cat([self.get_text_embeddings(h).to(self.args.device),
                                        self.get_text_embeddings(t).to(self.args.device),
                                        self.get_text_embeddings(r).to(self.args.device)], dim=1)

                    logits, hidden = self.forward(input_embs, hidden)

                    probs = F.softmax(logits, dim=-1)
                    log_prob = F.log_softmax(logits, dim=-1)
                    entropy = -(log_prob * probs).sum(1, keepdim=False)

                    action = probs.multinomial(num_samples=1).data
                    # print('action:', action)
                    if action==1:
                        q_triplets[q_idx].append([(h, t, r)])
                    selected_log_prob = log_prob.gather(
                        1, get_variable(action, requires_grad=False))
                    entropies.append(entropy)
                    log_probs.append(selected_log_prob[:, 0])


        if log_probs:
            log_probs_tensor = torch.cat(log_probs)
        else:
            log_probs_tensor = torch.tensor([])  # 或者选择合适的默认值

        if entropies:
            entropies_tensor = torch.cat(entropies)
        else:
            entropies_tensor = torch.tensor([])  # 或者选择合适的默认值

        return log_probs_tensor, entropies_tensor, q_triplets

        # return torch.cat(log_probs), torch.cat(entropies), q_triplets
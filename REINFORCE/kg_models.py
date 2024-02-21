import collections
import os
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
from torch.autograd import Variable
from collections import defaultdict


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

def process_batch(batch, ent_encoder, rel_encoder, device):
    batch_inputs = []
    for (h, t, r) in batch:
        h_tensor = torch.LongTensor([int(h)]).to(device)
        t_tensor = torch.LongTensor([int(t)]).to(device)
        r_tensor = torch.LongTensor([int(r)]).to(device)
        input_tensor = torch.cat([
            ent_encoder(h_tensor),
            ent_encoder(t_tensor),
            rel_encoder(r_tensor)
        ], dim=1)
        batch_inputs.append(input_tensor)
        del h_tensor
        del t_tensor
        del r_tensor
        del input_tensor

    return batch_inputs

def process_batch_transd(batch, ent_encoder, rel_encoder,ent_transfer, rel_transfer, device):
    batch_inputs = []
    for (h, t, r) in batch:
        h_tensor = torch.LongTensor([int(h)]).to(device)
        t_tensor = torch.LongTensor([int(t)]).to(device)
        r_tensor = torch.LongTensor([int(r)]).to(device)
        input_tensor = torch.cat([
            ent_encoder(h_tensor),
            ent_encoder(t_tensor),
            rel_encoder(r_tensor),
            ent_transfer(h_tensor),
            ent_transfer(t_tensor),
            rel_transfer(r_tensor)
        ], dim=1)
        batch_inputs.append(input_tensor)
        del h_tensor
        del t_tensor
        del r_tensor
        del input_tensor

    return batch_inputs

def process_batch_transh(batch, ent_encoder, rel_encoder,norm_vector, device):
    batch_inputs = []
    for (h, t, r) in batch:
        h_tensor = torch.LongTensor([int(h)]).to(device)
        t_tensor = torch.LongTensor([int(t)]).to(device)
        r_tensor = torch.LongTensor([int(r)]).to(device)
        input_tensor = torch.cat([
            ent_encoder(h_tensor),
            ent_encoder(t_tensor),
            rel_encoder(r_tensor),
            norm_vector(r_tensor)
        ], dim=1)
        batch_inputs.append(input_tensor)
        del h_tensor
        del t_tensor
        del r_tensor
        del input_tensor

    return batch_inputs

def process_batch_rescal(batch, ent_encoder, rel_encoder,relation_transform, device, rescal_dim):
    batch_inputs = []
    for (h, t, r) in batch:
        h_tensor = torch.LongTensor([int(h)]).to(device)
        t_tensor = torch.LongTensor([int(t)]).to(device)
        r_tensor = torch.LongTensor([int(r)]).to(device)

        input_tensor = torch.cat([
            ent_encoder(h_tensor),
            ent_encoder(t_tensor),
            relation_transform(rel_encoder(r_tensor).view(-1, rescal_dim * rescal_dim))
        ], dim=1)
        batch_inputs.append(input_tensor)
        del h_tensor
        del t_tensor
        del r_tensor
        del input_tensor

    return batch_inputs

def process_batch_complEx(batch, ent_re_encoder, rel_re_encoder,ent_im_encoder, rel_im_encoder, device):
    batch_inputs = []
    for (h, t, r) in batch:
        h_tensor = torch.LongTensor([int(h)]).to(device)
        t_tensor = torch.LongTensor([int(t)]).to(device)
        r_tensor = torch.LongTensor([int(r)]).to(device)

        input_tensor = torch.cat([
            ent_re_encoder(h_tensor),
            ent_im_encoder(t_tensor),
            rel_re_encoder(r_tensor),
            rel_im_encoder(r_tensor)
        ], dim=1)
        batch_inputs.append(input_tensor)
        del h_tensor
        del t_tensor
        del r_tensor
        del input_tensor

    return batch_inputs
# process_batch_analogy(batch, self.ent_encoder, self.rel_encoder,self.ent_re_encoder,self.rel_re_encoder,self.ent_im_encoder,self.rel_im_encoder, self.args.device)
def process_batch_analogy(batch, ent_encoder, rel_encoder, ent_re_encoder, rel_re_encoder,ent_im_encoder, rel_im_encoder, device):
    batch_inputs = []
    for (h, t, r) in batch:
        h_tensor = torch.LongTensor([int(h)]).to(device)
        t_tensor = torch.LongTensor([int(t)]).to(device)
        r_tensor = torch.LongTensor([int(r)]).to(device)

        input_tensor = torch.cat([
            ent_encoder(h_tensor),
            ent_encoder(t_tensor),
            rel_encoder(r_tensor),
            ent_re_encoder(h_tensor),
            ent_im_encoder(t_tensor),
            rel_re_encoder(r_tensor),
            rel_im_encoder(r_tensor)
        ], dim=1)
        batch_inputs.append(input_tensor)
        del h_tensor
        del t_tensor
        del r_tensor
        del input_tensor

    return batch_inputs

def process_batch_simple(batch, ent_encoder, rel_encoder,rel_inv_encoder, device):

    batch_inputs = []
    for (h, t, r) in batch:
        h_tensor = torch.LongTensor([int(h)]).to(device)
        t_tensor = torch.LongTensor([int(t)]).to(device)
        r_tensor = torch.LongTensor([int(r)]).to(device)

        input_tensor = torch.cat([
            ent_encoder(h_tensor),
            ent_encoder(t_tensor),
            rel_encoder(r_tensor),
            rel_inv_encoder(r_tensor)
        ], dim=1)
        batch_inputs.append(input_tensor)
        del h_tensor
        del t_tensor
        del r_tensor
        del input_tensor

    return batch_inputs

class Agent(torch.nn.Module):

    def __init__(self, args, hyper_edges, hyper_edges_ns, model):
        torch.nn.Module.__init__(self)
        self.args = args
        LP_model = args.LP_model
        self.hyper_edges = hyper_edges
        self.hyper_edges_ns = hyper_edges_ns
        self.edge_num = len(hyper_edges)
        self.edge_num_ns = len(hyper_edges_ns)
        self.batch_size = args.batch_size
        if(LP_model == 'complEx' or LP_model == 'simple'):
            if self.args.network_type.lower() == 'lstm':
                self.lstm = torch.nn.LSTMCell(4*args.agent_dim, args.agent_dim)
        elif(LP_model == 'analogy'):
            if self.args.network_type.lower() == 'lstm':
                self.lstm = torch.nn.LSTMCell(10*args.agent_dim, args.agent_dim)
        elif(LP_model == 'transd'):
            if self.args.network_type.lower() == 'lstm':
                self.lstm = torch.nn.LSTMCell(6*args.agent_dim, args.agent_dim)
        elif(LP_model == 'transh'):
            if self.args.network_type.lower() == 'lstm':
                self.lstm = torch.nn.LSTMCell(4*args.agent_dim, args.agent_dim)
        else:
            if self.args.network_type.lower() == 'lstm':
                self.lstm = torch.nn.LSTMCell(3*args.agent_dim, args.agent_dim)

        self.decoders = []
        for idx, size in enumerate([args.agent_dim, 2]):
            decoder = torch.nn.Linear(args.agent_dim, size)
            self.decoders.append(decoder)

        self.decoders = torch.nn.ModuleList(self.decoders)


        self.reset_parameters()

        self.static_init_hidden = keydefaultdict(self.init_hidden)

        def _get_default_hidden(key):
            return get_variable(
                torch.zeros(key, self.args.agent_dim),
                self.args.cuda,
                requires_grad=False)

        # self.static_inputs = keydefaultdict(_get_default_hidden)
        if (LP_model == 'transe'):
            self.ent_encoder = model.ent_embeddings.to(args.device)
            self.rel_encoder = model.rel_embeddings.to(args.device)
        elif (LP_model == 'rescal'):
            self.ent_encoder = model.ent_embeddings.to(args.device)
            self.rel_encoder = model.rel_matrices.to(args.device)
            self.relation_transform = nn.Linear(model.dim * model.dim, model.dim)
            self.rescal_dim = model.dim
            # self.rel_encoder = self.relation_transform(self.rel_matrices.view(-1, model.dim * model.dim))
        elif (LP_model == 'analogy'):
            self.ent_encoder = model.ent_embeddings.to(args.device)
            self.rel_encoder = model.rel_embeddings.to(args.device)
            self.ent_re_encoder = model.ent_re_embeddings.to(args.device)
            self.rel_re_encoder = model.rel_re_embeddings.to(args.device)
            self.ent_im_encoder = model.ent_im_embeddings.to(args.device)
            self.rel_im_encoder = model.rel_im_embeddings.to(args.device)
        elif (LP_model == 'simple'):
            self.ent_encoder = model.ent_embeddings.to(args.device)
            self.rel_encoder = model.rel_embeddings.to(args.device)
            self.rel_inv_encoder = model.rel_inv_embeddings.to(args.device)
        elif (LP_model == 'complEx'):
            self.ent_re_encoder = model.ent_re_embeddings.to(args.device)
            self.rel_re_encoder = model.rel_re_embeddings.to(args.device)
            self.ent_im_encoder = model.ent_im_embeddings.to(args.device)
            self.rel_im_encoder = model.rel_im_embeddings.to(args.device)
        elif (LP_model == 'transd'):
            self.ent_encoder = model.ent_embeddings.to(args.device)
            self.rel_encoder = model.rel_embeddings.to(args.device)
            self.ent_transfer = model.ent_transfer.to(args.device)
            self.rel_transfer = model.rel_transfer.to(args.device)
        elif(LP_model == 'transh'):
            self.ent_encoder = model.ent_embeddings.to(args.device)
            self.rel_encoder = model.rel_embeddings.to(args.device)
            self.norm_vector = model.norm_vector.to(args.device)


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
                input,
                hidden):

        # print(input)
        # print(hidden[0].size())
        hx, cx = self.lstm(input, hidden)
        for decoder in self.decoders:
            logits = decoder(hx)

        logits /= self.args.softmax_temperature

        # exploration
        if self.args.mode == 'train':
            logits = (self.args.tanh_c*F.tanh(logits))

        return logits, (hx, cx)

    def sample(self, batch_size=1):
        """Samples a set of hyper-edge from the agent, where each hyper-edge is made up of an activation function.
        """
        if batch_size < 1:
            raise Exception(f'Wrong batch_size: {batch_size} < 1')

        # [B, L, H]
        hidden = (torch.randn(self.batch_size, self.args.agent_dim).to(self.args.device),
                  torch.randn(self.batch_size, self.args.agent_dim).to(self.args.device))

        entropies = []
        log_probs = []
        selected_edges = []

        # The LSTM agent alternately outputs an activation,
        # followed by a previous hyper-edge.

        assert len(self.hyper_edges) == self.edge_num
        for i in range(0, len(self.hyper_edges), self.batch_size):

            batch = self.hyper_edges[i:i + self.batch_size]
            LP_model = self.args.LP_model
            if (LP_model == 'transe'):
                inputs = process_batch(batch, self.ent_encoder, self.rel_encoder, self.args.device)
            elif (LP_model == 'rescal'):
                inputs = process_batch_rescal(batch, self.ent_encoder, self.rel_encoder, self.relation_transform, self.args.device, self.rescal_dim)
            elif (LP_model == 'analogy'):
                inputs = process_batch_analogy(batch, self.ent_encoder, self.rel_encoder,self.ent_re_encoder,self.rel_re_encoder,self.ent_im_encoder,self.rel_im_encoder, self.args.device)
            elif (LP_model == 'simple'):
                inputs = process_batch_simple(batch, self.ent_encoder, self.rel_encoder,self.rel_inv_encoder, self.args.device)
            elif (LP_model == 'complEx'):
                inputs = process_batch_complEx(batch,self.ent_re_encoder,self.rel_re_encoder,self.ent_im_encoder,self.rel_im_encoder,self.args.device)
            elif (LP_model == 'transd'):
                inputs = process_batch_transd(batch, self.ent_encoder, self.rel_encoder,self.ent_transfer,self.rel_transfer, self.args.device)
            elif (LP_model == 'transh'):
                inputs = process_batch_transh(batch, self.ent_encoder, self.rel_encoder,self.norm_vector, self.args.device)
            inputs = torch.cat(inputs, dim=0)
            edge_list = list(range(i, i + len(inputs)))
            # print(inputs.size())
            inputs = F.pad(inputs, pad=(0, 0, 0, self.batch_size-len(inputs)))
            logits, hidden = self.forward(inputs, hidden)

            probs = F.softmax(logits, dim=-1)
            log_prob = F.log_softmax(logits, dim=-1)
            entropy = -(log_prob * probs).sum(1, keepdim=False)

            # actions = probs.multinomial(num_samples=1).data
            # actions_list = actions.flatten().cpu().detach().tolist()
            # for edge_idx in range(len(edge_list)):
            #     selected_edges += [self.hyper_edges[edge_list[edge_idx]] for action in actions_list if action == 1]
            #     selected_log_prob = log_prob.gather(
            #         1, get_variable(actions, requires_grad=False))
            #     entropies.append(entropy)
            #     log_probs.append(selected_log_prob[:, 0])
            actions = probs.multinomial(num_samples=1).data
            actions_list = actions.flatten().cpu().detach().tolist()
            selected_edges += [self.hyper_edges[edge_idx] for ind, edge_idx in enumerate(edge_list) if actions_list[ind] == 1]
            selected_log_prob = log_prob.gather(
                1, get_variable(actions, requires_grad=False))
            entropies.append(entropy)
            log_probs.append(selected_log_prob[:, 0])


        return torch.cat(log_probs), torch.cat(entropies), selected_edges
    
    def sample_ns(self, batch_size=1):
        """Samples a set of hyper-edge from the agent, where each hyper-edge is made up of an activation function.
        """
        if batch_size < 1:
            raise Exception(f'Wrong batch_size: {batch_size} < 1')

        # [B, L, H]
        hidden = (torch.randn(self.batch_size, self.args.agent_dim).to(self.args.device),
                  torch.randn(self.batch_size, self.args.agent_dim).to(self.args.device))

        entropies = []
        log_probs = []
        selected_edges = []

        # The LSTM agent alternately outputs an activation,
        # followed by a previous hyper-edge.

        assert len(self.hyper_edges_ns) == self.edge_num_ns
        for i in range(0, len(self.hyper_edges_ns), self.batch_size):

            batch = self.hyper_edges_ns[i:i + self.batch_size]
            LP_model = self.args.LP_model
            if (LP_model == 'transe'):
                inputs = process_batch(batch, self.ent_encoder, self.rel_encoder, self.args.device)
            elif (LP_model == 'rescal'):
                inputs = process_batch_rescal(batch, self.ent_encoder, self.rel_encoder, self.relation_transform, self.args.device, self.rescal_dim)
            elif (LP_model == 'analogy'):
                inputs = process_batch_analogy(batch, self.ent_encoder, self.rel_encoder,self.ent_re_encoder,self.rel_re_encoder,self.ent_im_encoder,self.rel_im_encoder, self.args.device)
            elif (LP_model == 'simple'):
                inputs = process_batch_simple(batch, self.ent_encoder, self.rel_encoder,self.rel_inv_encoder, self.args.device)
            elif (LP_model == 'complEx'):
                inputs = process_batch_complEx(batch,self.ent_re_encoder,self.rel_re_encoder,self.ent_im_encoder,self.rel_im_encoder,self.args.device)
            elif (LP_model == 'transd'):
                inputs = process_batch_transd(batch, self.ent_encoder, self.rel_encoder,self.ent_transfer,self.rel_transfer, self.args.device)
            elif (LP_model == 'transh'):
                inputs = process_batch_transh(batch, self.ent_encoder, self.rel_encoder,self.norm_vector, self.args.device)
            # inputs = process_batch(batch, self.ent_encoder, self.rel_encoder, self.args.device)
            inputs = torch.cat(inputs, dim=0)
            edge_list = list(range(i, i + len(inputs)))
            # print(inputs.size())
            inputs = F.pad(inputs, pad=(0, 0, 0, self.batch_size-len(inputs)))
            logits, hidden = self.forward(inputs, hidden)

            probs = F.softmax(logits, dim=-1)
            log_prob = F.log_softmax(logits, dim=-1)
            entropy = -(log_prob * probs).sum(1, keepdim=False)

            # actions = probs.multinomial(num_samples=1).data
            # actions_list = actions.flatten().cpu().detach().tolist()
            # for edge_idx in range(len(edge_list)):
            #     selected_edges += [self.hyper_edges[edge_list[edge_idx]] for action in actions_list if action == 1]
            #     selected_log_prob = log_prob.gather(
            #         1, get_variable(actions, requires_grad=False))
            #     entropies.append(entropy)
            #     log_probs.append(selected_log_prob[:, 0])
            actions = probs.multinomial(num_samples=1).data
            actions_list = actions.flatten().cpu().detach().tolist()
            selected_edges += [self.hyper_edges_ns[edge_idx] for ind, edge_idx in enumerate(edge_list) if actions_list[ind] == 1]
            selected_log_prob = log_prob.gather(
                1, get_variable(actions, requires_grad=False))
            entropies.append(entropy)
            log_probs.append(selected_log_prob[:, 0])


        return torch.cat(log_probs), torch.cat(entropies), selected_edges
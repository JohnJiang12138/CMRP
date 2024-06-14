from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter

from earlystopping import EarlyStopping
from sample import Sampler
from metric import accuracy, roc_auc_compute_fn
# from deepgcn.utils import load_data, accuracy
# from deepgcn.models import GCN
import sys
sys.path.append("../DropEdge/src")
from DropEdge.src.metric import accuracy
from DropEdge.src.utils import load_citation, load_reddit_data
from DropEdge.src.models import *
from DropEdge.src.earlystopping import EarlyStopping
from DropEdge.src.sample import Sampler




def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


# define the training function.
def train(args, epoch, train_adj, train_fea, idx_train, val_adj=None, val_fea=None):
    model, optimizer, sampler, labels, idx_val, early_stopping, scheduler = \
    args.model, args.optimizer, args.sampler, args.labels, args.idx_val, args.early_stopping_class, args.scheduler

    if val_adj is None:
        val_adj = train_adj
        val_fea = train_fea

    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(train_fea, train_adj)
    # special for reddit
    if sampler.learning_type == "inductive":
        loss_train = F.nll_loss(output, labels[idx_train])
        acc_train = accuracy(output, labels[idx_train])
    else:
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])

    loss_train.backward()
    optimizer.step()
    train_t = time.time() - t
    val_t = time.time()
    # We can not apply the fastmode for the reddit dataset.
    # if sampler.learning_type == "inductive" or not args.fastmode:

    if args.early_stopping > 0 and sampler.dataset != "reddit":
        loss_val = F.nll_loss(output[idx_val], labels[idx_val]).item()
        early_stopping(loss_val, model)

    if not args.fastmode:
        #    # Evaluate validation set performance separately,
        #    # deactivates dropout during validation run.
        model.eval()
        output = model(val_fea, val_adj)
        loss_val = F.nll_loss(output[idx_val], labels[idx_val]).item()
        acc_val = accuracy(output[idx_val], labels[idx_val]).item()
        if sampler.dataset == "reddit":
            early_stopping(loss_val, model)
    else:
        loss_val = 0
        acc_val = 0

    if args.lradjust:
        scheduler.step()

    val_t = time.time() - val_t
    return (loss_train.item(), acc_train.item(), loss_val, acc_val, get_lr(optimizer), train_t, val_t)


def test(model, args, test_adj, test_fea):
    labels, idx_test = args.labels, args.idx_test
    model.eval()
    output = model(test_fea, test_adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    auc_test = roc_auc_compute_fn(output[idx_test], labels[idx_test])
    if args.debug:
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "auc= {:.4f}".format(auc_test),
              "accuracy= {:.4f}".format(acc_test.item()))
        print("accuracy=%.5f" % (acc_test.item()))
    return (loss_test.item(), acc_test.item())


def training_and_test(args, adj=None,selected_nodes=None):
    # Train model
    t_total = time.time()
    loss_train = np.zeros((args.gcn_epochs,))
    acc_train = np.zeros((args.gcn_epochs,))
    loss_val = np.zeros((args.gcn_epochs,))
    acc_val = np.zeros((args.gcn_epochs,))

    model = args.model
    early_stopping = args.early_stopping_class
    sampler = args.sampler
    if adj is not None:
        sampler._update_adj(adj)
    # sampling_t = 0

    for epoch in range(args.gcn_epochs):
        input_idx_train = args.idx_train
        sampling_t = time.time()
        # no sampling
        # randomedge sampling if args.sampling_percent >= 1.0, it behaves the same as stub_sampler.
        (train_adj, train_fea) = args.sampler.randomedge_sampler(percent=args.sampling_percent,
                                                            normalization=args.normalization,
                                                            cuda=args.cuda,
                                                            selected_nodes=selected_nodes)
        if args.mixmode:
            train_adj = train_adj.cuda()

        sampling_t = time.time() - sampling_t

        # The validation set is controlled by idx_val
        # if sampler.learning_type == "transductive":

        (val_adj, val_fea) = sampler.get_test_set(normalization=args.normalization, cuda=args.cuda)
        if args.mixmode:
            val_adj = val_adj.cuda()
        outputs = train(args, epoch, train_adj, train_fea, input_idx_train, val_adj, val_fea)

        # if args.debug and epoch % 1 == 0:
        #     print('Epoch: {}'.format(epoch + 1),
        #           'loss_train: {:.4f}'.format(outputs[0]),
        #           'acc_train: {:.4f}'.format(outputs[1]),
        #           'loss_val: {:.4f}'.format(outputs[2]),
        #           'acc_val: {:.4f}'.format(outputs[3]),
        #           'cur_lr: {:.5f}'.format(outputs[4]),
        #           's_time: {:.4f}s'.format(sampling_t),
        #           't_time: {:.4f}s'.format(outputs[5]),
        #           'v_time: {:.4f}s'.format(outputs[6]))

        if args.no_tensorboard is False:
            args.tb_writer.add_scalars('Loss', {'train': outputs[0], 'val': outputs[2]}, epoch)
            args.tb_writer.add_scalars('Accuracy', {'train': outputs[1], 'val': outputs[3]}, epoch)
            args.tb_writer.add_scalar('lr', outputs[4], epoch)
            args.tb_writer.add_scalars('Time', {'train': outputs[5], 'val': outputs[6]}, epoch)

        loss_train[epoch], acc_train[epoch], loss_val[epoch], acc_val[epoch] = outputs[0], outputs[1], outputs[2], \
        outputs[3]

        if args.early_stopping > 0 and early_stopping.early_stop:
            print("Early stopping.")
            model.load_state_dict(early_stopping.load_checkpoint())
            break

    if args.early_stopping > 0:
        model.load_state_dict(early_stopping.load_checkpoint())

    if args.debug:
        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # Testing
    (test_adj, test_fea) = sampler.get_test_set(normalization=args.normalization, cuda=args.cuda)
    if args.mixmode:
        test_adj = test_adj.cuda()
    (loss_test, acc_test) = test(model, args, test_adj, test_fea)
    # print("%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f" % (
    #     loss_train[-1], loss_val[-1], loss_test, acc_train[-1], acc_val[-1], acc_test))

    return loss_train[-1], loss_val[-1], loss_test, acc_train[-1], acc_val[-1], acc_test, model
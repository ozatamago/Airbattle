# ===== Original Version =====
# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]
#
# =====Modified Version =====
# Copyright (c) 2021-2023 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)

# training

import os
import time
import datetime
import copy
import threading
import random
import bz2
import cloudpickle
import warnings
import queue
from collections import deque

from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import psutil

from .environment import prepare_env, make_env
from .util import map_r, bimap_r, trimap_r, rotate
from .model import to_torch, to_gpu, ModelWrapper
from .losses import compute_target
from .connection import MultiProcessJobExecutor
from .worker import WorkerCluster, WorkerServer
from ASRCAISim1.addons.HandyRLUtility.distribution import getActionDistributionClass
import sys
# ↓パスが通らなかったらコメントアウト外す
# sys.path.append('../')
from Airbattle.scripts.Core import getTotalObservationSize
from Airbattle.scripts.Helper.Printer import Printer, PrintColor
def replace_none(a, b):
    return a if a is not None else b

def make_batch(episodes, args):
    """Making training batch

    Args:
        episodes (Iterable): list of episodes
        args (dict): training configuration

    Returns:
        dict: PyTorch input and target tensors

    Note:
        Basic data shape is (B, T, P, ...) .
        (B is batch size, T is time length, P is player count)
    """

    obss, datum = [], []
    hidden_ins, probs, acts, lacts = [], [], [], []

    policy_to_train = args['policy_to_train']

    for ep in episodes:
        moments_ = sum([cloudpickle.loads(bz2.decompress(ms)) for ms in ep['moment']], [])
        moments = moments_[ep['start'] - ep['base']:ep['end'] - ep['base']]
        players = [idx for idx, policy in enumerate(ep['policy_map'])
            if policy in [policy_to_train] + ['Imitator']
        ]
        if len(players) ==0:
            continue
        # template for padding
        obs_zeros = map_r(args['observation_space'].sample(), lambda x: np.zeros(getTotalObservationSize()))
        action_zeros = args['action_dist_class'].getDefaultAction(args['action_space'])
        prob_ones = map_r(action_zeros, lambda x: np.zeros_like(x))
        legal_actions_zeros = args['action_dist_class'].getDefaultLegalActions(args['action_space'])

        # data that is changed by training configuration (ASRC: Disabled turn-based configuration.)
        obs = [[replace_none(m['observation'][player], obs_zeros) for player in players] for m in moments]
        prob = [[replace_none(m['selected_prob'][player], prob_ones) for player in players] for m in moments]
        act = [[replace_none(m['action'][player], action_zeros) for player in players] for m in moments]
        lact = [[replace_none(m['legal_actions'][player], legal_actions_zeros) for player in players] for m in moments]
        initial_hidden = args['initial_hidden']
        if(initial_hidden is not None):
            hidden_in = [[replace_none(m['hidden_in'][player], initial_hidden) for player in players] for m in moments]
        # print(f"MakeBatch: Moments: {moments}")
        # reshape observation etc.
        obs = rotate(rotate(obs))  # (T, P, ..., ...) -> (P, ..., T, ...) -> (..., T, P, ...)
        # print(f"MakeBatch: Obs: {[s_obs[0].shape for s_obs in obs]}, Obs_zeros: {obs_zeros.shape}")
        obs = bimap_r(obs_zeros, obs, lambda _, o: np.array(o))
        prob = rotate(rotate(prob))
        prob = bimap_r(prob_ones, prob, lambda _, p: np.array(p))
        act = rotate(rotate(act))
        act = bimap_r(action_zeros, act, lambda _, a: np.array(a))
        lact = rotate(rotate(lact))
        lact = bimap_r(legal_actions_zeros, lact, lambda _, a: np.array(a))
        if(initial_hidden is not None):
            hidden_in = rotate(rotate(hidden_in))
            hidden_in = bimap_r(initial_hidden, hidden_in, lambda _, h: np.array(h))
        # datum that is not changed by training configuration
        
        v = np.array([[replace_none(m['value'][player], [0]) for player in players] for m in moments], dtype=np.float32).reshape(len(moments), len(players), -1)
        rew = np.array([[replace_none(m['reward'][player], [0]) for player in players] for m in moments], dtype=np.float32).reshape(len(moments), len(players), -1)
        ret = np.array([[replace_none(m['return'][player], [0]) for player in players] for m in moments], dtype=np.float32).reshape(len(moments), len(players), -1)
        oc = np.array([ep['outcome'][player] for player in players], dtype=np.float32).reshape(1, len(players), -1)
        # score = np.array([ep['score'][player] for player in players], dtype=np.float32).reshape(1, len(players), -1)
        emask = np.ones((len(moments), 1, 1), dtype=np.float32)  # episode mask
        tmask = np.array([[[m['selected_prob'][player] is not None] for player in players] for m in moments], dtype=np.float32)
        omask = np.array([[[m['observation'][player] is not None] for player in players] for m in moments], dtype=np.float32)

        # Imitation flag
        imi = np.array([[[1.0 if ep['policy_map'][player] == 'Imitator' else 0.0] for player in players] for m in moments], dtype=np.float32)

        progress = np.arange(ep['start'], ep['end'], dtype=np.float32)[..., np.newaxis] / ep['total']

        # pad each array if step length is short
        batch_steps = args['burn_in_steps'] + args['forward_steps']
        if len(tmask) < batch_steps:
            pad_len_b = args['burn_in_steps'] - (ep['train_start'] - ep['start'])
            pad_len_a = batch_steps - len(tmask) - pad_len_b
            obs = map_r(obs, lambda o: np.pad(o, [(pad_len_b, pad_len_a)] + [(0, 0)] * (len(o.shape) - 1), 'constant', constant_values=0))
            prob = bimap_r(prob_ones, prob, lambda p_o, p: np.concatenate([np.tile(p_o, [pad_len_b, p.shape[1]]+[1]*(len(p.shape) - 2)), p, np.tile(p_o, [pad_len_a, p.shape[1]]+[1]*(len(p.shape) - 2))]))
            v = np.concatenate([np.pad(v, [(pad_len_b, 0), (0, 0), (0, 0)], 'constant', constant_values=0), np.tile(oc, [pad_len_a, 1, 1])])
            act = bimap_r(action_zeros, act, lambda a_z, a: np.concatenate([np.tile(a_z, [pad_len_b, a.shape[1]]+[1]*(len(a.shape) - 2)), a, np.tile(a_z, [pad_len_a, a.shape[1]]+[1]*(len(a.shape) - 2))]))
            rew = np.pad(rew, [(pad_len_b, pad_len_a), (0, 0), (0, 0)], 'constant', constant_values=0)
            ret = np.pad(ret, [(pad_len_b, pad_len_a), (0, 0), (0, 0)], 'constant', constant_values=0)
            emask = np.pad(emask, [(pad_len_b, pad_len_a), (0, 0), (0, 0)], 'constant', constant_values=0)
            tmask = np.pad(tmask, [(pad_len_b, pad_len_a), (0, 0), (0, 0)], 'constant', constant_values=0)
            omask = np.pad(omask, [(pad_len_b, pad_len_a), (0, 0), (0, 0)], 'constant', constant_values=0)
            lact = bimap_r(legal_actions_zeros, lact, lambda a_z, a: np.concatenate([np.tile(a_z, [pad_len_b, a.shape[1]]+[1]*(len(a.shape) - 2)), a, np.tile(a_z, [pad_len_a, a.shape[1]]+[1]*(len(a.shape) - 2))]))
            if(initial_hidden is not None):
                hidden_in = map_r(hidden_in, lambda h: np.pad(h, [(pad_len_b, pad_len_a)] + [(0, 0)] * (len(h.shape) - 1), 'constant', constant_values=0))
            progress = np.pad(progress, [(pad_len_b, pad_len_a), (0, 0)], 'constant', constant_values=1)
            imi = np.pad(imi, [(pad_len_b, pad_len_a), (0, 0), (0, 0)], 'constant', constant_values=0)
        
        obss.append(obs)
        probs.append(prob)
        acts.append(act)
        lacts.append(lact)
        if(initial_hidden is not None):
            hidden_ins.append(hidden_in)
        datum.append((v, oc, rew, ret, emask, tmask, omask, progress, imi))

    obs = to_torch(bimap_r(obs_zeros, rotate(obss), lambda _, o: np.array(o)))
    prob = to_torch(bimap_r(prob_ones, rotate(probs), lambda _, p: np.array(p)))
    act = to_torch(bimap_r(action_zeros, rotate(acts), lambda _, a: np.array(a)))
    lact = to_torch(bimap_r(legal_actions_zeros, rotate(lacts), lambda _, a: np.array(a)))
    if(initial_hidden is not None):
        hidden_in = to_torch(bimap_r(initial_hidden, rotate(hidden_ins), lambda _, h: np.array(h)))
    else:
        hidden_in = None
    v, oc, rew, ret, emask, tmask, omask, progress, imi= [to_torch(np.array(val)) for val in zip(*datum)]

    ret = {
        'observation': obs,
        'selected_prob': prob,
        'value': v,
        'action': act, 'outcome': oc,
        'reward': rew, 'return': ret,
        'episode_mask': emask,
        'turn_mask': tmask, 'observation_mask': omask,
        'legal_actions': lact,
        'progress': progress,
        'imitation': imi,
        'hidden_in': hidden_in,
    }
    return ret


def forward_prediction(model, hidden, batch, args):
    """Forward calculation via neural network

    Args:
        model (torch.nn.Module): neural network
        hidden: initial hidden state (..., B, P, ...)
        batch (dict): training batch (output of make_batch() function)

    Returns:
        tuple: batch outputs of neural network
    """

    observations = batch['observation']  # (..., B, T, P or 1, ...)
    batch_shape = batch['observation_mask'].size()[:3]  # (B, T, P or 1)

    if hidden is None:
        # feed-forward neural network
        obs = map_r(observations, lambda o: o.flatten(0, 2))  # (..., B * T * P or 1, ...)
        outputs = model(obs, None)
        outputs = map_r(outputs, lambda o: o.unflatten(0, batch_shape))  # (..., B, T, P or 1, ...)
    else:
        # sequential computation with RNN
        # RNN block is assumed to be an instance of ASRCAISim1.addons.HandyRLUtility.RecurrentBlock.RecurrentBlock
        outputs = {}
        hidden = [h[:,0,...] for h in batch['hidden_in']]

        seq_len = batch_shape[1]
        obs = map_r(observations, lambda o: o.transpose(1, 2))  # (..., B , P or 1 , T, ...)
        omask_ = map_r(batch['observation_mask'], lambda o: o.transpose(1, 2))  # (..., B , P or 1 , T, ...)
        omask = map_r(hidden, lambda h: omask_.view([*h.size()[:2], seq_len, *([1] * (h.dim() - 2))]).flatten(0,1))
        hidden_ = map_r(hidden, lambda h: h.flatten(0, 1))  # (..., B * P or 1, ...)
        if args['burn_in_steps'] >0 :
            model.eval()
            with torch.no_grad():
                outputs_ = model(map_r(obs, lambda o:o[:, :, :args['burn_in_steps'], ...].flatten(0, 2)), hidden_, args['burn_in_steps'],mask=map_r(omask,lambda o: o[:,:args['burn_in_steps'],...]))
            hidden_ = outputs_.pop('hidden')
            outputs_ = map_r(outputs_, lambda o: o.unflatten(0, (batch_shape[0], batch_shape[2], args['burn_in_steps'])).transpose(1, 2))
            for k, o in outputs_.items():
                outputs[k] = outputs.get(k, []) + [o]
        if not model.training:
            model.train()
        outputs_ = model(map_r(obs, lambda o:o[:, :, args['burn_in_steps']:, ...].flatten(0, 2)), hidden_,seq_len-args['burn_in_steps'],mask=map_r(omask,lambda o: o[:,args['burn_in_steps']:,...]))
        hidden_ = outputs_.pop('hidden')
        outputs_ = map_r(outputs_, lambda o: o.unflatten(0, (batch_shape[0], batch_shape[2], seq_len-args['burn_in_steps'])).transpose(1, 2))
        for k, o in outputs_.items():
            outputs[k] = outputs.get(k, []) + [o]
        outputs = {k: torch.cat(o, dim=1) for k, o in outputs.items() if o[0] is not None}

    for k, o in outputs.items():
        if k == 'policy':
            o = o.mul(batch['turn_mask'])
            if o.size(2) > 1 and batch_shape[2] == 1:  # turn-alternating batch
                o = o.sum(2, keepdim=True)  # gather turn player's policies
                outputs[k] = o
        else:
            # mask valid target values and cumulative rewards
            outputs[k] = o.mul(batch['observation_mask'])

    return outputs


def compose_losses(outputs, entropies, log_selected_policies, total_advantages, targets, batch, args):
    """Caluculate loss value

    Returns:
        tuple: losses and statistic values and the number of training data
    """

    tmasks = batch['turn_mask']
    omasks = batch['observation_mask']

    losses = {}
    dcnt = tmasks.sum().item()

    losses['p'] = (-log_selected_policies * total_advantages).mul(tmasks).sum()
    if 'value' in outputs:
        losses['v'] = ((outputs['value'] - targets['value']) ** 2).mul(omasks).sum() / 2
    if 'return' in outputs:
        losses['r'] = F.smooth_l1_loss(outputs['return'], targets['return'], reduction='none').mul(omasks).sum()

    entropy = entropies.mul(tmasks.sum(-1,keepdim=True))
    losses['ent'] = entropy.sum()

    base_loss = losses['p'] + losses.get('v', 0) + losses.get('r', 0) + losses.get('r_i', 0)
    entropy_loss = entropy.mul(1 - batch['progress'].unsqueeze(-1) * (1 - args['entropy_regularization_decay'])).sum() * -args['entropy_regularization']
    losses['ent_loss']=entropy_loss
    losses['total'] = base_loss + entropy_loss

    return losses, dcnt


def compute_loss(batch, model, hidden, args):
    outputs = forward_prediction(model, hidden, batch, args)
    if hidden is not None and args['burn_in_steps'] > 0:
        batch = map_r(batch, lambda v: v[:, args['burn_in_steps']:] if v.size(1) > 1 else v)
        outputs = map_r(outputs, lambda v: v[:, args['burn_in_steps']:])

    dist = args['action_dist_class'](outputs['policy'], model, batch['legal_actions'])
    actions = dist.unpack_scalar(batch['action'],keepDim=True)
    log_probs = dist.unpack_scalar(dist.log_prob(batch['action']),keepDim=True)
    selected_probs = dist.unpack_scalar(batch['selected_prob'],keepDim=True)
    entropies = dist.unpack_scalar(dist.entropy(batch['action']),keepDim=True)
    separate_policy_gradients = args.get('separate_policy_gradients',True)
    if not separate_policy_gradients:
        log_probs = [sum(log_probs)]
        selected_probs = [sum(selected_probs)]
        entropies = [sum(entropies)]
    emasks = batch['episode_mask']
    clip_rho_threshold, clip_c_threshold = 1.0, 1.0

    outputs_nograd = {k: o.detach() for k, o in outputs.items()}

    if 'value' in outputs_nograd:
        values_nograd = outputs_nograd['value']
        if args['turn_based_training'] and values_nograd.size(2) == 2:  # two player zerosum game
            values_nograd_opponent = -torch.stack([values_nograd[:, :, 1], values_nograd[:, :, 0]], dim=2)
            values_nograd = (values_nograd + values_nograd_opponent) / (batch['observation_mask'].sum(dim=2, keepdim=True) + 1e-8)
        outputs_nograd['value'] = values_nograd * emasks + batch['outcome'] * (1 - emasks)

    # calculate losses for each action component
    losses_total = {}
    for idx in range(len(log_probs)):
        log_selected_b_policies = selected_probs[idx] * emasks
        log_selected_t_policies = log_probs[idx] * emasks

        # thresholds of importance sampling
        log_rhos = log_selected_t_policies.detach() - log_selected_b_policies
        rhos = torch.exp(log_rhos)
        clipped_rhos = torch.clamp(rhos, 0, clip_rho_threshold)
        cs = torch.clamp(rhos, 0, clip_c_threshold)

        # compute targets and advantage
        targets = {}
        advantages = {}

        value_args = outputs_nograd.get('value', None), batch['outcome'], None, args['lambda'], 1, clipped_rhos, cs
        return_args = outputs_nograd.get('return', None), batch['return'], batch['reward'], args['lambda'], args['gamma'], clipped_rhos, cs

        targets['value'], advantages['value'] = compute_target(args['value_target'], *value_args)
        targets['return'], advantages['return'] = compute_target(args['value_target'], *return_args)

        if args['policy_target'] != args['value_target']:
            _, advantages['value'] = compute_target(args['policy_target'], *value_args)
            _, advantages['return'] = compute_target(args['policy_target'], *return_args)

        # compute policy advantage
        summed_advantages = sum(advantages.values())
        total_advantages = clipped_rhos * summed_advantages
        losses, dcnt = compose_losses(outputs, entropies[idx], log_selected_t_policies, total_advantages, targets, batch, args)
        for k,v in losses.items():
            losses_total[k] = losses_total.get(k,0) + v

        # imitation
        imi = batch['imitation']
        if(imi.sum()>0):
            tmasks = batch['turn_mask']
            imitation_beta = args.get('imitation_beta', 1.0)
            if('imitation_kl_threshold' in args):
                clip_imitation_kl_threshold = args['imitation_kl_threshold']
                clip_imitation_loss_threshold = args.get('imitation_loss_threshold', clip_imitation_kl_threshold)
            elif('imitation_loss_threshold' in args):
                clip_imitation_loss_threshold = args['imitation_loss_threshold']
                clip_imitation_kl_threshold = args.get('imitation_kl_threshold', clip_imitation_loss_threshold)
            else:
                clip_imitation_kl_threshold = 10.0
                clip_imitation_loss_threshold = 10.0
            imitation_loss_scale=args.get('imitation_loss_scale',1.0)
            if(imitation_beta > 0.0):
                args["imitation_adv_ma"]+=args["imitation_adv_ma_update_rate"]*(float(torch.mean(torch.pow(summed_advantages, 2.0)))-args["imitation_adv_ma"])
                kl = -(torch.exp(imitation_beta*summed_advantages/(1e-8+pow(args["imitation_adv_ma"],0.5))).detach() * log_selected_t_policies).mul(tmasks).mul(imi)
            else:
                kl = -log_selected_t_policies.mul(tmasks).mul(imi)
            kl = torch.clamp(kl,0,clip_imitation_kl_threshold)
            kl = torch.clamp(kl*imitation_loss_scale, 0, clip_imitation_loss_threshold).sum()
            losses_total['imi'] = losses_total.get('imi', 0.0) + kl
            losses_total['total'] += kl

    return losses_total, dcnt


class Batcher:
    def __init__(self, args, episodes):
        self.args = args
        self.episodes = episodes
        self.executor = MultiProcessJobExecutor(self._worker, self._selector(), self.args['num_batchers'])

    def _selector(self):
        while True:
            yield [self.select_episode() for _ in range(self.args['batch_size'])]

    def _worker(self, conn, bid):
        print('started batcher %d' % bid)
        while True:
            episodes = conn.recv()
            batch = make_batch(episodes, self.args)
            try:
                conn.send(batch)
            except RuntimeError as e:
                print(f"{e} ... wait 0.5 sec")
                time.sleep(0.5)
        print('finished batcher %d' % bid)

    def run(self):
        self.executor.start()

    def select_episode(self):
        while True:
            ep_count = min(len(self.episodes), self.args['maximum_episodes'])
            ep_idx = random.randrange(ep_count)
            accept_rate = 1 - (ep_count - 1 - ep_idx) / self.args['maximum_episodes']
            if random.random() < accept_rate:
                break
        ep = self.episodes[ep_idx]
        turn_candidates = 1 + max(0, ep['steps'] - self.args['forward_steps'])  # change start turn by sequence length
        train_st = random.randrange(turn_candidates)
        st = max(0, train_st - self.args['burn_in_steps'])
        ed = min(train_st + self.args['forward_steps'], ep['steps'])
        st_block = st // self.args['compress_steps']
        ed_block = (ed - 1) // self.args['compress_steps'] + 1
        ep_minimum = {
            'args': ep['args'], 'outcome': ep['outcome'],
            'moment': ep['moment'][st_block:ed_block],
            'base': st_block * self.args['compress_steps'],
            'start': st, 'end': ed, 'train_start': train_st, 'total': ep['steps'],
            'policy_map': ep['policy_map'],
        }
        return ep_minimum

    def batch(self):
        return self.executor.recv()


class Trainer:
    def __init__(self, args, model, summary_writer):
        self.episodes = deque()
        self.args = args
        self.gpu = torch.cuda.device_count()
        self.model = model
        self.summary_writer = summary_writer
        self.args['initial_hidden'] = self.model.init_hidden()
        self.args['observation_space'] = self.model.observation_space
        self.args['action_space'] = self.model.action_space
        self.args['action_dist_class'] = self.model.action_dist_class
        self.default_lr = 3e-8
        self.data_cnt_ema = self.args['batch_size'] * self.args['forward_steps']
        self.params = list(self.model.parameters())
        lr = self.default_lr * self.data_cnt_ema
        self.optimizer = optim.Adam(self.params, lr=lr, weight_decay=1e-5) if len(self.params) > 0 else None
        self.steps = 0
        # print(f"Trainer args: {self.args}")
        self.batcher = Batcher(self.args, self.episodes)
        self.update_flag = False
        self.update_queue = queue.Queue(maxsize=1)

        self.wrapped_model = ModelWrapper(self.model)
        self.trained_model = self.wrapped_model
        if self.gpu > 1:
            self.trained_model = nn.DataParallel(self.wrapped_model)
        # imitation
        self.args["imitation_adv_ma_initial"]=self.args.get("imitation_adv_ma_initial",100.0)
        self.args["imitation_adv_ma_update_rate"]=self.args.get("imitation_adv_ma_update_rate",1.0e-8)
        self.args["imitation_adv_ma"]=self.args["imitation_adv_ma_initial"]
        self.update_count = 0
        self.reset_weight_queue = queue.Queue(maxsize=1)

    def reset_weight(self, path):
        try:
            self.reset_weight_queue.put_nowait(path)
        except queue.Full:
            pass

    def update(self):
        self.update_flag = True
        model, steps = self.update_queue.get()
        self.update_count += 1
        return model, steps

    def train(self):
        if self.optimizer is None:  # non-parametric model
            print("Optimizer is None!")
            time.sleep(0.1)
            return self.model
        print("Train model!")
        batch_cnt, data_cnt, loss_sum = 0, 0, {}
        if self.gpu > 0:
            self.trained_model.cuda()
        self.trained_model.train()

        while data_cnt == 0 or not self.update_flag:
            batch = self.batcher.batch()
            batch_size = batch['value'].size(0)
            player_count = batch['value'].size(2)
            hidden = self.wrapped_model.init_hidden([batch_size, player_count])
            if self.gpu > 0:
                batch = to_gpu(batch)
                hidden = to_gpu(hidden)
            self.trained_model.updateNetworks(obs=batch['observation'],rew=batch['reward'],action_space=self.args['action_space'])
            # losses, dcnt, = compute_loss(batch, self.trained_model, hidden, self.args)

            #self.optimizer.zero_grad()
            #losses['total'].backward()
            #nn.utils.clip_grad_norm_(self.params, 4.0)
            #self.optimizer.step()

            batch_cnt += 1
            data_cnt += batch['turn_mask'].sum().item()
            #for k, l in losses.items():
            #    loss_sum[k] = loss_sum.get(k, 0.0) + l.item()

            self.steps += 1

        for k, l in loss_sum.items():
            self.summary_writer.add_scalar("trainer/loss/"+k,l/data_cnt,self.update_count)
        self.summary_writer.add_scalar("trainer/num_updates",self.steps,self.update_count)
        self.summary_writer.flush()
        print('loss = %s' % ' '.join([k + ':' + '%.3f' % (l / data_cnt) for k, l in loss_sum.items()]))

        self.data_cnt_ema = self.data_cnt_ema * 0.8 + data_cnt / (1e-2 + batch_cnt) * 0.2
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.default_lr * self.data_cnt_ema / (1 + self.steps * 1e-5)
        self.model.cpu()
        self.model.eval()
        reset_weight_path = None
        while True:
            try:
                reset_weight_path = self.reset_weight_queue.get_nowait()
            except queue.Empty:
                break
        if reset_weight_path is not None:
            print("============reset weight========",reset_weight_path)
            # reset_weight=torch.load(reset_weight_path)
            # self.model.load_state_dict(reset_weight, strict=False)
            self.model.load_state_dict(None, strict=False)
        return copy.deepcopy(self.model)

    def run(self):
        print('waiting training')
        while len(self.episodes) < self.args['minimum_episodes']:
            time.sleep(1)
        if self.optimizer is not None:
            self.batcher.run()
            print('started training')
        while True:
            model = self.train()
            self.update_flag = False
            self.update_queue.put((model, self.steps))
        print('finished training')


class Learner:
    def __init__(self, args, net=None, remote=False):
        #保存先の設定
        self.save_dir_base = args.get('save_dir',os.path.abspath('.'))
        self.save_dir = os.path.join(
            self.save_dir_base,
            datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        )
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(os.path.join(self.save_dir,'policies', 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir,'policies', 'initial_weights'), exist_ok=True)
        #SummaryWriterの生成
        self.summary_writer = SummaryWriter(os.path.join(self.save_dir,'logs'))
        #引数の修正
        train_args = args['train_args']
        train_args['policy_to_imitate']=train_args.get('policy_to_imitate',[])
        env_args = args['env_args']
        env_args['policy_config'] = args['policy_config']
        env_args['teams'] = args['match_maker_args'].get('teams',['Blue','Red'])
        train_args['env'] = env_args
        match_maker_args = args['match_maker_args']
        match_maker_args['weight_pool'] = os.path.join(self.save_dir, 'policies', 'weight_pool')
        match_maker_args['log_prefix'] = os.path.join(self.save_dir, 'matches', 'matches')
        match_maker_args['policy_config'] = args['policy_config']
        train_args['match_maker'] = match_maker_args
        args = train_args

        self.args = args
        self.name = args['name']
        random.seed(args['seed'])
        # print(env_args)
        self.env = make_env(env_args)
        eval_modify_rate = (args['update_episodes'] ** 0.85) / args['update_episodes']
        self.eval_rate = max(args['eval_rate'], eval_modify_rate)
        self.shutdown_flag = False
        self.flags = set()

        #MatchMakerの生成
        self.custom_classes = self.args['custom_classes']
        self.match_maker = self.custom_classes[self.args["match_maker_class"]](match_maker_args)

        # trained datum
        self.policy_to_train = self.args['policy_to_train']
        self.model_epoch = 0
        self.models = {policyName: self.get_model(policyName)
            for policyName in self.env.policy_config.keys()
        }

        #初期重みの読み込みとpoolへの追加
        populate_config = self.match_maker.checkInitialPopulation()
        for policyName, policyConfig in self.env.policy_config.items():
            model = self.get_model(policyName)
            initial_weight_path = policyConfig.get('initial_weight', None)
            if(initial_weight_path is not None):
                model.load_state_dict(torch.load(initial_weight_path, map_location = torch.device('cpu')), strict = True)
            torch.save(model.state_dict(), self.initial_model_path(policyName))
            self.models[policyName] = model
            if(not policyName == self.policy_to_train):
                self.models[policyName].eval()
            if(policyName in populate_config):
                self.populate(policyName, populate_config[policyName]['weight_id'], populate_config[policyName]['reset'])

        # generated datum
        self.generation_results = {}
        self.num_episodes = 0
        self.num_returned_episodes = 0

        # evaluated datum
        self.results = {}
        self.results_per_opponent = {}
        self.num_results = 0
        self.num_returned_results = 0

        # multiprocess or remote connection
        self.worker = WorkerServer(args) if remote else WorkerCluster(args)

        # thread connection
        self.trainer = Trainer(args, self.models[self.policy_to_train], self.summary_writer)

    def get_model(self,policyName):
        model_class = self.custom_classes[self.env.net(policyName)]
        obs_space = self.env.policy_config[policyName]['observation_space']
        ac_space = self.env.policy_config[policyName]['action_space']
        model_config = self.env.policy_config[policyName].get('model_config',{})
        if('actionDistributionClassGetter' in model_config):
            action_dist_class = self.custom_classes[model_config['actionDistributionClassGetter']](ac_space)
        else:
            action_dist_class = getActionDistributionClass(ac_space)
        model_config['custom_classes']=self.custom_classes
        return model_class(obs_space, ac_space, action_dist_class, model_config)

    def model_path(self, policyName, model_id):
        return os.path.join(self.save_dir, 'policies', 'checkpoints', policyName+'-'+str(model_id) + '.pth')

    def latest_model_path(self, policyName):
        return os.path.join(self.save_dir, 'policies', 'checkpoints', policyName+'-latest.pth')

    def initial_model_path(self, policyName):
        return os.path.join(self.save_dir, 'policies', 'initial_weights', policyName+'.pth')
    
    def populate(self, policyName, weight_id, reset):
        dstPath = os.path.join(self.match_maker.weight_pool, policyName+"-"+str(weight_id)+".pth")
        os.makedirs(os.path.dirname(dstPath), exist_ok=True)
        torch.save(self.models[policyName].state_dict(), dstPath)
        if(reset):
            self.models[policyName].load_state_dict(torch.load(self.initial_model_path(policyName), map_location=torch.device('cpu')),strict=True)
            if(policyName==self.policy_to_train):
                self.trainer.reset_weight(self.initial_model_path(policyName))
        print("=====weight populated===== ",policyName," -> ",os.path.basename(dstPath), ("(reset)" if reset else ""))

    def update_model(self, model, steps):
        # get latest model and save it
        print('updated model(%d)' % steps)
        self.model_epoch += 1
        self.models[self.policy_to_train] = model
        torch.save(model.state_dict(), self.model_path(self.policy_to_train,self.model_epoch))
        torch.save(model.state_dict(), self.latest_model_path(self.policy_to_train))
        path=os.path.join(self.save_dir, 'matches', 'checkpoints', 'MatchMaker-'+str(self.model_epoch) + '.dat')
        self.match_maker.save(path)

    def feed_episodes(self, episodes):
        # analyze generated episodes
        def get_policy_label_for_log(info):
            ret = info['Policy']
            if info['Suffix'] != '':
                weight_id = info['Weight']
                if weight_id == 0:
                    ret = ret + "*"
                elif weight_id > 0:
                    ret = ret + str(weight_id)
            return ret
        print(f"FEED EPISODES {len(episodes)}")
        for episode in episodes:
            if episode is None:
                print("NO EPISODE!")
                continue
            # print(f"EPISODE : {episode}")
            for team, info in episode['args']['match_info'].items():
                policyName=info['Policy']#+info['Suffix']
                weight_id=info['Weight']
                outcome = np.mean([episode['outcome'][p] for p in episode['args']['player']
                    if episode['policy_map'][p] == info['Policy']+info['Suffix']
                ])
                score = np.mean([episode['score'][p] for p in episode['args']['player']
                    if episode['policy_map'][p] == info['Policy']+info['Suffix']
                ])
                moments_ = sum([cloudpickle.loads(bz2.decompress(ms)) for ms in episode['moment']], [])
                rewards = [
                    np.array([replace_none(m['reward'][p], [0]) for m in moments_], dtype=np.float32) for p in episode['args']['player']
                    if episode['policy_map'][p] == info['Policy']+info['Suffix']
                ]
                reward_mean = np.mean(rewards)
                reward_total = np.sum(rewards)
                if(policyName==self.policy_to_train and weight_id<0):
                    model_epoch = episode['args']['model_epoch']
                    n, r, r2 = self.generation_results.get(model_epoch, (0, 0, 0))
                    self.generation_results[model_epoch] = n + 1, r + score, r2 + score ** 2

                policyLabelForLog = get_policy_label_for_log(info)
                self.summary_writer.add_scalar("generation/outcome/"+policyLabelForLog,outcome,self.num_returned_episodes)
                self.summary_writer.add_scalar("generation/score/"+policyLabelForLog,score,self.num_returned_episodes)
                self.summary_writer.add_scalar("generation/reward_mean/"+policyLabelForLog,reward_mean,self.num_returned_episodes)
                self.summary_writer.add_scalar("generation/reward_total/"+policyLabelForLog,reward_total,self.num_returned_episodes)

            populate_config = self.match_maker.onEpisodeEnd(episode['args']['match_info'],episode['match_maker_result'])
            for policyName,conf in populate_config.items():
                self.populate(policyName, conf['weight_id'], conf['reset'])
            match_maker_metrics = self.match_maker.get_metrics(episode['args']['match_info'],episode['match_maker_result'])
            for k,v in match_maker_metrics.items():
                self.summary_writer.add_scalar("match_maker/"+k,v,self.num_returned_episodes)
            self.num_returned_episodes += 1

        # store generated episodes
        self.trainer.episodes.extend([e for e in episodes if e is not None])

        mem_percent = psutil.virtual_memory().percent
        mem_ok = mem_percent <= 95
        maximum_episodes = self.args['maximum_episodes'] if mem_ok else int(len(self.trainer.episodes) * 95 / mem_percent)
        self.summary_writer.add_scalar("mem_percent",mem_percent,self.num_returned_episodes)

        if not mem_ok and 'memory_over' not in self.flags:
            warnings.warn("memory usage %.1f%% with buffer size %d" % (mem_percent, len(self.trainer.episodes)))
            self.flags.add('memory_over')

        while len(self.trainer.episodes) > maximum_episodes:
            self.trainer.episodes.popleft()
        self.summary_writer.flush()

    def feed_results(self, results):
        # store evaluation results
        def get_policy_label_for_log(info):
            ret = info['Policy']
            if info['Suffix'] != '':
                weight_id = info['Weight']
                if weight_id == 0:
                    ret = ret + "*"
                elif weight_id > 0:
                    ret = ret + str(weight_id)
            return ret

        for result in results:
            if result is None:
                continue
            for team, info in result['args']['match_info'].items():
                policyName=info['Policy']#+info['Suffix']
                weight_id=info['Weight']
                res = np.mean([result['score'][p] for p in result['args']['player']
                    if result['policy_map'][p] == info['Policy']+info['Suffix']
                ])
                if(policyName==self.policy_to_train and weight_id<=0):
                    model_epoch = result['args']['model_epoch']
                    n, r, r2 = self.results.get(model_epoch, (0, 0, 0))
                    self.results[model_epoch] = n + 1, r + res, r2 + res ** 2
                    if model_epoch not in self.results_per_opponent:
                        self.results_per_opponent[model_epoch] = {}
                    opponent = ','.join(sorted([get_policy_label_for_log(op) for t,op in result['args']['match_info'].items() if t!=team]))
                    n, r, r2 = self.results_per_opponent[model_epoch].get(opponent, (0, 0, 0))
                    self.results_per_opponent[model_epoch][opponent] = n + 1, r + res, r2 + res ** 2
                self.summary_writer.add_scalar("evaluation/score/"+get_policy_label_for_log(info),res,self.num_returned_results)

            populate_config = self.match_maker.onEpisodeEnd(result['args']['match_info'],result['match_maker_result'])
            for policyName,conf in populate_config.items():
                self.populate(policyName, conf['weight_id'], conf['reset'])
            match_maker_metrics = self.match_maker.get_metrics(result['args']['match_info'],result['match_maker_result'])
            for k,v in match_maker_metrics.items():
                self.summary_writer.add_scalar("match_maker/"+k,v,self.num_returned_episodes)
            self.num_returned_results += 1
        self.summary_writer.flush()

    def update(self):
        # call update to every component
        print()
        print('epoch %d' % self.model_epoch)

        if self.model_epoch not in self.results:
            print('evaluation stats = Nan (0)')
        else:
            def output_wp(name, results):
                n, r, r2 = results
                mean = r / (n + 1e-6)
                name_tag = ' (%s)' % name if name != '' else ''
                std = (r2 / (n + 1e-6) - mean ** 2) ** 0.5
                print('evaluation stats%s = %.3f +- %.3f' % (name_tag, mean, std))

            keys = self.results_per_opponent[self.model_epoch]
            output_wp('total', self.results[self.model_epoch])
            for key in sorted(list(self.results_per_opponent[self.model_epoch])):
                output_wp(key, self.results_per_opponent[self.model_epoch][key])

        if self.model_epoch not in self.generation_results:
            print('generation stats = Nan (0)')
        else:
            n, r, r2 = self.generation_results[self.model_epoch]
            mean = r / (n + 1e-6)
            std = (r2 / (n + 1e-6) - mean ** 2) ** 0.5
            print('generation stats = %.3f +- %.3f' % (mean, std))

        model, steps = self.trainer.update()
        if model is None:
            model = self.models[self.policy_to_train]
        self.update_model(model, steps)

        # clear flags
        self.flags = set()

    def server(self):
        # central conductor server
        # returns as list if getting multiple requests as list
        print('started server')
        prev_update_episodes = self.args['minimum_episodes']
        # no update call before storing minimum number of episodes + 1 epoch
        next_update_episodes = prev_update_episodes + self.args['update_episodes']

        while self.worker.connection_count() > 0 or not self.shutdown_flag:
            try:
                conn, (req, data) = self.worker.recv(timeout=0.3)
            except queue.Empty:
                continue

            multi_req = isinstance(data, list)
            if not multi_req:
                data = [data]
            send_data = []

            if req == 'args':
                if self.shutdown_flag:
                    send_data = [None] * len(data)
                else:
                    for _ in data:
                        args = {'model_epoch': self.model_epoch}

                        # decide role
                        if self.num_results < self.eval_rate * self.num_episodes:
                            args['role'] = 'e'
                        else:
                            args['role'] = 'g'
                        args['match_type'] = self.name+':'+args['role']
                        args['match_info'] = self.match_maker.makeNextMatch(args['match_type'])
                        det_rate=self.args.get('deterministic',{'g':0.0,'e':0.0}).get(args['role'],0.0)
                        for team in args['match_info']:
                            args['match_info'][team]['deterministic'] = True if random.random() < det_rate else False
                        if args['role'] == 'g':
                            # genatation configuration
                            self.num_episodes += 1

                        elif args['role'] == 'e':
                            # evaluation configuration
                            self.num_results += 1

                        send_data.append(args)

            elif req == 'episode':
                # report generated episodes
                self.feed_episodes(data)
                send_data = [None] * len(data)

            elif req == 'result':
                # report evaluation results
                self.feed_results(data)
                send_data = [None] * len(data)

            elif req == 'model':
                for policyName,model_id in data:
                    model = self.models[policyName]
                    if self.policy_to_train == policyName and model_id != self.model_epoch and model_id > 0:
                        try:
                            model = self.get_model(policyName)
                            model.load_state_dict(torch.load(self.model_path(policyName,model_id), map_location=torch.device('cpu')), strict=True)
                        except:
                            # return latest model if failed to load specified model
                            pass
                    send_data.append(cloudpickle.dumps(model))

            if not multi_req and len(send_data) == 1:
                send_data = send_data[0]
            self.worker.send(conn, send_data)

            if self.num_returned_episodes >= next_update_episodes:
                prev_update_episodes = next_update_episodes
                next_update_episodes = prev_update_episodes + self.args['update_episodes']
                self.update()
                if self.args['epochs'] >= 0 and self.model_epoch >= self.args['epochs']:
                    self.shutdown_flag = True
        self.summary_writer.close()
        print('finished server')

    def run(self):
        # open training thread
        threading.Thread(target=self.trainer.run, daemon=True).start()
        # open generator, evaluator
        self.worker.run()
        self.server()


def train_main(args):
    prepare_env(args['env_args'])  # preparing environment is needed in stand-alone mode
    learner = Learner(args=args)
    learner.run()


def train_server_main(args):
    learner = Learner(args=args, remote=True)
    learner.run()

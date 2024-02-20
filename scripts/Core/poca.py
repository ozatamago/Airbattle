import sys
from typing import Any, Mapping
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import *
import copy
import gym
import numpy as np
from ..Helper.Printer import Printer
from ..Helper.TensorExtension import TensorExtension
from ..Helper.DictExtension import DictExtension
import warnings
from . import util
from gymnasium import spaces
from ASRCAISim1.addons.HandyRLUtility.model import ModelBase

# from trainer.policy_trainer import PolicyNetworkTrainer
# from scripts.Core.v_network import VNetwork as v
# from scripts.Core.q_network import QNetwork as q


def getBatchSize(obs,space):
    if(isinstance(space,spaces.Dict)):
        k=next(iter(space))
        return getBatchSize(obs[k],space[k])
    elif(isinstance(space,spaces.Tuple)):
        return  getBatchSize(obs[0],space[0])
    else:
        return obs.shape[0]


class RSABlock(nn.Module):
    def __init__(self, input_size: int, output_size: int, num_heads: int = 4):
        super(RSABlock,self).__init__()
        self.input_layer = nn.Linear(input_size, output_size)
        self.output_layer = nn.Linear(output_size, output_size)
        self.attention = nn.MultiheadAttention(output_size, num_heads)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.input_layer(inputs)) # (batch_size, num_agents, output_size)
        attention_output, _ = self.attention(x, x, x) # (batch_size, num_agents, output_size)
        context = torch.add(attention_output, x) # (batch_size, num_agents, output_size)
        ret = F.relu(self.output_layer(context)).clone() # (batch_size, num_agents, output_size)
        return ret

class V(nn.Module):
    def __init__(self,input_dim: int, v_hid_dim: int = 64, rsa_heads: int = 4,lr: float=0.001,lam = 0.5,gam = 0.5,lam_eps = 1e-8,gam_eps = 1e-8):
        super(V,self).__init__()
        self.lam = lam
        self.lam_eps = lam_eps
        self.gam = gam
        self.gam_eps = gam_eps
        self.value_net = nn.Sequential(
            RSABlock(input_dim,v_hid_dim,rsa_heads), # V値計算用RSA
            nn.LeakyReLU(),
            nn.Linear(v_hid_dim,1) # V値計算
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.loss_func = nn.MSELoss()

    def forward(self,inputs):
        if isinstance(inputs,(list,tuple)):
            return torch.cat([self.forward(input) for input in inputs],dim=0)
        elif isinstance(inputs,torch.Tensor):
            if inputs.size(-2) > 0:
                if inputs.dim() < 3:
                    input = inputs.unsqueeze(0)
                else:
                    input = inputs
                v_value = self.value_net(input) # Vφ (batch_size,num_agents,1)
                return torch.sum(v_value, dim=-2) # (batch_size, 1)
            else:
                return torch.tensor([[0]])
        else:
            raise ValueError(f"Calucrate V value Failed! UnknownType: inputs:{type(inputs)}")
     
    def yt_lambda(self,encoded_obs_seq: list,rew_seq: torch.Tensor,t: int):
        total = torch.zeros((1))
        if self.lam == 1.0:
            return total
        max_n = len(encoded_obs_seq) - t
        p_lam = 1.0
        for n in range(1,max_n):
            if p_lam >= self.lam_eps:
                total = total + p_lam*self.Gt_lambda(encoded_obs_seq,rew_seq,t,n)
                p_lam *= self.lam
            else:
                p_lam = .0
            break
        return total*(1.0 - self.lam)

    def Gt_lambda(self,encoded_obs_seq: list,rew_seq: torch.Tensor,t: int,n: int):
        total = torch.zeros((1))
        p_gam = 1.0
        for l in range(1,n):
            if p_gam >= self.gam_eps:
                total = total + p_gam*rew_seq[t+l]
                p_gam *= self.gam
            else:
                p_gam = .0
            break
        return  ((self.forward(encoded_obs_seq[t+n])*p_gam) if p_gam >= self.gam_eps else torch.zeros((1))) + total

    def update(self,encoded_obs: list, actives_list: list, reward: torch.Tensor, ts: list):
        for t in ts:
            if actives_list[t] > 0:
                self.optimizer.zero_grad()
                yt_lam = self.yt_lambda(encoded_obs,reward,t)
                pre_value = self.forward(encoded_obs[t])
                value = pre_value.view(-1,pre_value.size(1))
                loss = self.loss_func(value,yt_lam.expand(value.size()))
                loss.backward(retain_graph=True)
                self.optimizer.step()
   
class Q(nn.Module):
    def __init__(self,input_dim: int, q_hid_dim: int = 64, rsa_heads: int = 4, lr: float=0.001):
        super(Q,self).__init__()
        self.value_net = nn.Sequential(
            RSABlock(input_dim,q_hid_dim,rsa_heads), # Q値計算用RSA
            nn.LeakyReLU(),
            nn.Linear(q_hid_dim,1) # Q値計算
        )
        self.loss_func = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self,encoded_counterfactual: torch.Tensor):
        q_value = self.value_net(encoded_counterfactual.view(-1,encoded_counterfactual.shape[-2],encoded_counterfactual.shape[-1])) # (batch_size*num_agents,num_agents,q_hid_dim)
        q_value = q_value.view(-1,encoded_counterfactual.shape[1],encoded_counterfactual.shape[2],q_value.shape[-1]) # Qψ (batch_size,num_agents,num_agents,1)
        q_value = torch.sum(q_value, dim=2) # (batch_size,num_agents, 1)
        return q_value
    
    def update(self,encoded_counterfactual: torch.Tensor,yt_lambda: torch.Tensor):
        self.optimizer.zero_grad()
        pre_q_value = self.forward(encoded_counterfactual)
        q_value = pre_q_value.view(-1,pre_q_value.size(1))
        q_loss = self.loss_func(q_value,yt_lambda.expand(q_value.size()))
        q_loss.backward(retain_graph=True)
        self.optimizer.step()

class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, rho=0.05,beta=3 ,lr=0.001):
        super(SparseAutoencoder, self).__init__()
        # エンコーダとデコーダの層を定義
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.ReLU()
        )
        # 最適化アルゴリズムを設定
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.rho = torch.tensor([rho])
        self.beta = beta
    def forward(self, input, get_hidden: bool=True):
        h = self.encoder(input)
        if get_hidden:
            return h
        y = self.decoder(h)
        return y
    def updateEncoder(self, batch):
        self.optimizer.zero_grad()
        outputs = self.forward(batch, get_hidden=False)
        mse_loss = self.criterion(outputs, batch)
        # KLダイバージェンスによるスパース性制約
        rho_hat = torch.mean(torch.sigmoid(self.encoder(batch)), 0)
        sparsity_loss = F.kl_div(torch.log(rho_hat), self.rho,reduction='sum')
        loss = mse_loss + self.beta * sparsity_loss
        loss.backward(retain_graph=True)
        self.optimizer.step()

class StateEncoder(nn.Module):
    def __init__(self,obs_dim: int,act_dim: int, out_dim: int):
        super(StateEncoder,self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.out_dim = out_dim
        self.obs_encode_layer = SparseAutoencoder(obs_dim,out_dim)
        self.obs_act_encode_layer = SparseAutoencoder(obs_dim+act_dim,out_dim)

    def encode_obs(self,observations):
        if isinstance(observations,(list,tuple)):
            return type(observations)([self.encode_obs(obs) for obs in observations])
        elif isinstance(observations,torch.Tensor):
            obs = observations.view(-1,self.obs_dim) # (batch_size*num_agents, obs_dim)
            return self.obs_encode_layer(obs)
        else:
            raise ValueError(f"Encode Observation Failed! UnknownType: observations:{type(observations)}")
    
    def encode_obs_acts(self,observations,actions):
        if isinstance(observations,(list,tuple)) or isinstance(actions,(list,tuple)):
            return type(observations)([self.encode_obs_acts(observations[i],actions[i]) for i in len(observations)])
        elif isinstance(observations,torch.Tensor) and isinstance(actions,torch.Tensor):
            obs = observations.view(-1,self.obs_dim)
            acts = actions.view(-1,self.act_dim)
            return self.obs_act_encode_layer(torch.cat([obs,acts],dim=-1)) # f (batch_size*num_agents, out_dim)
        else:
            raise ValueError(f"Encode Observation Action Pair Failed! UnknownType: observations:{type(observations)} and actions:{type(actions)}")
    
    def encode_counterfactual(self,encoded_observations,encoded_actions):
        if isinstance(encoded_observations,(list,tuple)) or isinstance(encoded_actions,(list,tuple)):
            return type(encoded_observations)([self.encode_obs_acts(encoded_observations[i],encoded_actions[i]) for i in len(encoded_observations)])
        elif isinstance(encoded_observations,torch.Tensor) and isinstance(encoded_actions,torch.Tensor):
            num_agents = encoded_observations.size(-2)
            encoded_obs: torch.Tensor = encoded_observations.view(-1,num_agents,self.out_dim) # g (batch_size,num_agents, out_dim)
            encoded_obs_acts: torch.Tensor = encoded_actions.view(-1,num_agents,self.out_dim) # f (batch_size,num_agents, out_dim)
            indices = [torch.tensor(i) for i in range(num_agents)]
            extracts = [(encoded_obs.index_select(1,indice),TensorExtension.extractSelect(1,encoded_obs_acts,indice)) for indice in indices]
            encoded_counterfactual = torch.stack([(g if f is None else torch.cat([g,f],dim=1)) for g,f in extracts],dim=1) # (g,f) (batch_size, num_agents, num_agents, out_dim)
            return encoded_counterfactual
        else:
            raise ValueError(f"Encode Failed! UnknownType: encoded_observations:{type(encoded_observations)} and actions:{type(encoded_actions)}")

    def forward(self,observations,actions):
        return self.encode_counterfactual(observations,actions)
    
    def updateObsEncoder(self,observations):
        if isinstance(observations,(list,tuple)):
            for i in range(len(observations)):
                self.updateObsEncoder(observations[i])
        elif isinstance(observations,torch.Tensor):
            if observations.size(-2) > 0:
                obs = observations.view(-1,self.obs_dim)
                self.obs_encode_layer.updateEncoder(obs)
        else:
            raise ValueError(f"ObsEncoder update Failed! UnknownType: observations:{type(observations)}")
        
    def updateObsActsEncoder(self,observations,actions):
        if isinstance(observations,(list,tuple)) or isinstance(actions,(list,tuple)):
            for i in range(len(observations)):
                self.updateObsActsEncoder(observations[i],actions[i])
        elif isinstance(observations,torch.Tensor) and isinstance(actions,torch.Tensor):
            if observations.size(-2) > 0:
                obs = observations.view(-1,self.obs_dim)
                acts = actions.view(-1,self.act_dim)
                self.obs_act_encode_layer.updateEncoder(torch.cat([obs,acts],dim=-1))
        else:
            raise ValueError(f"ObsActsEncoder update Failed! UnknownType: observations:{type(observations)} and actions:{type(actions)}")

class Actor(nn.Module):
    def __init__(self, encoded_obs_dim: int, act_dim: int, lr: float=0.001):
        super(Actor, self).__init__()
        self.act_dim = act_dim
        self.obs_dim = encoded_obs_dim
        self.policy_net = RSABlock(encoded_obs_dim,act_dim,2)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr)

    def forward(self, encoded_observations_list: list, actives_list: list, hidden: torch.Tensor = None) -> list:
        rets = list()
        for obs,active in zip(encoded_observations_list, actives_list):
            ret = dict()
            if active == 0:
                logits = torch.zeros((1,0,self.act_dim), dtype=torch.float32)
            else:
                logits = self.policy_net(obs)
            ret = {'policy':logits,'logits': logits, 'hidden': hidden}
            DictExtension.reduceNone(ret)
            rets.append(ret)
        return rets
    
    def update(self,observations_list: list, actives_list: list,reward: torch.Tensor,action_space: gym.spaces,v_model: V,q_model: Q,state_encoder: StateEncoder,ts: list):
        act_space = [s.n for s in action_space]
        act_split_index = [sum(act_space[:ai]) for ai in range(1,len(act_space))]
        encoded_observations_list = state_encoder.encode_obs(observations_list)
        print("Updating VModel")
        v_model.update(encoded_observations_list,actives_list,reward,ts)
        print("VModel Updated")
        print("Updating ActorModel, ObservationActionPairEncoder, QModel")
        for t in ts:
            if actives_list[t] > 0:
                action_policy: torch.Tensor = self.forward(encoded_observations_list[t:t+1],actives_list[t:t+1])[0]['policy']
                state_encoder.updateObsActsEncoder(observations_list[t],action_policy)
                encoded_obs_acts = state_encoder.encode_obs_acts(observations_list[t],action_policy)
                encoded_counterfactual: torch.Tensor= state_encoder.encode_counterfactual(encoded_observations_list[t],encoded_obs_acts)
                yt_lam = v_model.yt_lambda(encoded_observations_list,reward,t)
                q_model.update(encoded_counterfactual,yt_lam)
                self.optimizer.zero_grad()
                q_value = -q_model.forward(encoded_counterfactual).view(-1,1)
                adv = q_value + yt_lam.expand(q_value.size())
                flattened_action_policy = action_policy.view(-1)
                split_indexes = [asi for asi in act_split_index if asi <= flattened_action_policy.size(-1)]
                action_prob = torch.cat([torch.softmax(al,-1) for al in  torch.tensor_split(flattened_action_policy,split_indexes,-1)],-1).view(-1,self.act_dim)
                actors_loss = torch.mean(torch.mul(action_prob,adv.expand(action_prob.size())),-1)
                torch.mean(actors_loss).backward(retain_graph=True)
                self.optimizer.step()
        print("ActorModel, ObservationActionPairEncoder, QModel Updated")
                    
# ActorCritic class
class MAPOCA(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, max_agents: int, lr: float, hid_dim: int = 128, v_hid_dim: int = 64, v_rsa_heads: int = 4,q_hid_dim: int = 64, q_rsa_heads: int = 4):
        super(MAPOCA, self).__init__()
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.state_encoder = StateEncoder(obs_dim+1,act_dim,hid_dim) # g(o), f(o,a), (g,f)
        self.actor = Actor(hid_dim, act_dim)
        self.v_model = V(hid_dim,v_hid_dim,v_rsa_heads) # V値計算
        self.q_model = Q(hid_dim,q_hid_dim,q_rsa_heads) # Q値計算
        self.lr = lr
        self.max_agents = max_agents
    
    def forward(self, obs: torch.Tensor, hidden: torch.Tensor = None, get_only_one_action_output: bool = True) -> dict:
        observations_list, actives_list = self.split_to_obs_acts_list(obs)
        encoded_obs = self.state_encoder.encode_obs(observations_list)
        actor_output = self.actor.forward(encoded_obs, actives_list, hidden)
        if get_only_one_action_output:
            outputs: dict= actor_output[0]
            DictExtension.reduceNone(outputs,lambda o: TensorExtension.tensor_padding(o,self.max_agents,-2))
            outputs['policy'] =outputs['policy'].view(-1,self.max_agents*self.act_dim)
            outputs['logits'] =outputs['logits'].view(-1,self.max_agents*self.act_dim)
            return outputs
        return actor_output
    
    def split_to_obs_acts_list(self, obs: torch.Tensor):
        batch_size = obs.size(0)
        # obs == (batch_size,1+obs_dim*max_agents)
        actives: torch.Tensor = obs.index_select(-1,torch.tensor([0])) # (batch_size,1)
        actives_list = [active.item() for active in actives.to(torch.int32)]
        observations = torch.split(obs[:,1:],1,dim=0) # tuple((1,obs_dim*max_agents),...,(1,obs_dim*max_agents)) length == batch_size
        return [TensorExtension.tensor_padding(torch.stack(observations[batch_index].split(self.obs_dim,-1)[:actives_list[batch_index]],dim=1),1,-1,False,True,actives_list[batch_index]/self.max_agents) if actives_list[batch_index] > 0 else torch.zeros((0,1+self.obs_dim)) for batch_index in range(batch_size)], actives_list # list((num_agents,1 + obs_dim),...,(num_agents,1 +obs_dim), list(num_agents,...,num_agents)

    def updateNetworks(self,obs: torch.Tensor,rew: torch.Tensor,action_space: gym.spaces):
        warnings.simplefilter('error')
        print("UpdateNetworks...")
        obs = obs.flatten(0,2)
        reward = rew.flatten(0,2)
        rng = np.random.default_rng()
        times = list(range(obs.size(0)))
        observations_list, actives_list = self.split_to_obs_acts_list(obs)
        print("Updating ObservationEncoder")
        self.state_encoder.updateObsEncoder(observations_list)
        print("ObservationEncoder Updated")
        self.actor.update(observations_list,actives_list,reward,action_space,self.v_model,self.q_model,self.state_encoder,rng.permutation(times))
        print("UpdateNetworks Done")
        
    def init_hidden(self, hidden=None):
        # RNNを使用しない場合、ダミーの隠れ状態を返す
        return None
    
    def load_model(self,model: nn.Module,modelname:str,folderpath:str,strict:bool=True):
        model.load_state_dict(torch.load(folderpath+modelname+".pth"),strict)

    def load_state_dict(self, folderpath:str,strict:bool=True):
        self.load_model(self.actor,getActorModelName(),folderpath,strict)
        self.load_model(self.v_model,getVModelName(),folderpath,strict)
        self.load_model(self.state_encoder,getStateEncoderModelName(),folderpath,strict)
        self.load_model(self.q_model,getQModelName(),folderpath,strict)

    def save_model(self,model: nn.Module,modelname:str,folderpath:str):
        torch.save(model.state_dict(),folderpath+modelname+".pth")

    def save_state_dict(self,folderpath:str):
        self.save_model(self.actor,getActorModelName(),folderpath)
        self.save_model(self.v_model,getVModelName(),folderpath)
        self.save_model(self.state_encoder,getStateEncoderModelName(),folderpath)
        self.save_model(self.q_model,getQModelName(),folderpath)

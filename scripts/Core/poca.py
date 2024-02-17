import sys
from typing import Any, Mapping
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import *
import copy
import numpy as np
from ..Helper.Printer import Printer
from ..Helper.TensorExtension import TensorExtension
from ..Helper.DictExtension import DictExtension
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
    def __init__(self, input_size: int, output_size: int, num_heads: int):
        super(RSABlock,self).__init__()
        self.input_layer = nn.Linear(input_size, output_size)
        self.output_layer = nn.Linear(output_size, output_size)
        self.attention = nn.MultiheadAttention(output_size, num_heads)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.input_layer(inputs)) # (batch_size, num_agents, output_size)
        attention_output, _ = self.attention(x, x, x) # (batch_size, num_agents, output_size)
        context = torch.add(attention_output, x) # (batch_size, num_agents, output_size)
        context = F.relu(self.output_layer(context)) # (batch_size, num_agents, output_size)
        return context

class Q(nn.Module):
    def __init__(self,input_dim: int, q_hid_dim: int = 64, rsa_heads: int = 4):
        super(Q,self).__init__()
        self.q_rsa = RSABlock(input_dim,q_hid_dim,rsa_heads) # Q値計算用RSA
        self.q_layer = nn.Linear(q_hid_dim,1) # Q値計算

    def forward(self,inputs: torch.Tensor):
        q_value = self.q_rsa(inputs.view(-1,inputs.shape[-2],inputs.shape[-1])) # (batch_size*num_agents,num_agents,q_hid_dim)
        q_value = q_value.view(-1,inputs.shape[1],inputs.shape[2],q_value.shape[-1])
        q_value = self.q_layer(q_value) # Qψ (batch_size,num_agents,num_agents,1)
        q_value = torch.sum(q_value, dim=2) # (batch_size,num_agents, 1)
        return q_value

class V(nn.Module):
    def __init__(self,input_dim: int, v_hid_dim: int = 64, rsa_heads: int = 4):
        super(V,self).__init__()
        self.v_rsa = RSABlock(input_dim,v_hid_dim,rsa_heads) # V値計算用RSA
        self.v_layer = nn.Linear(v_hid_dim,1) # V値計算

    def forward(self,inputs: torch.Tensor):
        v_value = self.v_rsa(inputs) # (batch_size,num_agents,v_hid_dim)
        v_value = F.leaky_relu(self.v_layer(v_value)) # Vφ (batch_size,num_agents,1)
        v_value = torch.sum(v_value, dim=2) # (batch_size, 1)
        return v_value

# Actor class(Policy) 任意のネットワークアーキテクチャでよい
# Actor network
class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hid_dim: int = 128):
        super(Actor, self).__init__()
        self.hidden_layer1 = nn.Linear(obs_dim, hid_dim * 2) # first hidden layer
        self.hidden_layer2 = nn.Linear(hid_dim * 2, hid_dim) # second hidden layer
        self.action_layer = nn.Linear(hid_dim, act_dim) # action output layer
        self.softmax = nn.Softmax(dim=-1) # softmax activation function for log_prob output

    def forward(self, obs: torch.Tensor, hidden: torch.Tensor = None) -> dict:
        x = F.leaky_relu(self.hidden_layer1(obs)) # pass through the first hidden layer and apply relu
        x = F.leaky_relu(self.hidden_layer2(x)) # pass through the second hidden layer and apply relu
        logits = F.gelu(self.action_layer(x)) # pass through the action layer for logits output
        ret = {'policy':logits,'logits': logits, 'hidden': hidden}
        DictExtension.reduceNone(ret)
        return ret

# MA-POCA Criticモデルの定義
class Critic(nn.Module):
    def __init__(self, encoded_obs_dim: int, max_agents: int, hid_dim: int = 128, num_heads: int = 4):
        """
        価値関数を計算するCriticネットワーク
        param: encoded_obs_dim: Agentのエンコード後の観測テンソルの次元数
        param: max_agents: Agentの最大値
        param: hid_dim: 内部のRSAの隠れ層の次元数
        param: num_heads: 内部のRSAのヘッド数
        return: V値 (batch_size,1)
        """
        super(Critic,self).__init__()
        self.max_agents = max_agents
        self.v_layer = V(encoded_obs_dim + 1,hid_dim,num_heads)

    def forward(self, encoded_obs: torch.Tensor, actives: torch.Tensor) -> torch.Tensor:
        """
        エンコード後の全Agentの観測のテンソルを受け取って価値を返す
        param: encoded_obs: エンコード後の観測テンソル (batch_size,num_agents,encoded_obs_dim)
        param: actives: Agent数のテンソル (batch_size,1)
        returns: value (batch_size,1), sliced_obs list(tensor (1,encoded_obs_dim)) length == batch_size
        """
        indices = [torch.arange(act.item()) for act in actives]
        # 頭に残りのAgentの数をつけておく
        x = torch.cat([encoded_obs,torch.div(actives,self.max_agents).expand(encoded_obs.shape[0],encoded_obs.shape[1],1)],dim=-1)
        sliced_x = [x[bi].index_select(0,indice)[:,:-1].unsqueeze(0) for bi,indice in enumerate(indices)]
        value = torch.cat([self.v_layer(x) for x in sliced_x],dim=0) # (batch_size,1)
        return value, sliced_x

class StateEncoder(nn.Module):
    def __init__(self,obs_dim: int,act_dim: int, out_dim: int):
        super(StateEncoder,self).__init__()
        self.obs_encode_layer = nn.Linear(obs_dim,out_dim)
        self.obs_act_encode_layer = nn.Linear(obs_dim+act_dim,out_dim)

    def forward(self,observations: torch.Tensor,actions: torch.Tensor):
        num_agents = observations.size(1)
        indices = [torch.tensor(i) for i in range(num_agents)]
        encoded_obs: torch.Tensor = self.obs_encode_layer(observations) # g (batch_size,num_agents,out_dim)
        encoded_obs_act: torch.Tensor = self.obs_act_encode_layer(torch.cat([observations,actions],dim=-1)) # f (batch_size, num_agents, out_dim)
        extracts = [(encoded_obs.index_select(1,indice),TensorExtension.extractSelect(1,encoded_obs_act,indice)) for indice in indices]
        encoded_counterfactual = torch.stack([(g if f is None else torch.cat([g,f],dim=1)) for g,f in extracts],dim=1)
        return encoded_obs, encoded_obs_act, encoded_counterfactual

# ActorCritic class
class MAPOCA(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, max_agents: int, lr: float, hid_dim: int = 128, q_hid_dim: int = 64, q_rsa_heads: int = 4):
        super(MAPOCA, self).__init__()
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.actor = Actor(obs_dim, act_dim, hid_dim)
        self.state_encoder = StateEncoder(obs_dim,act_dim,hid_dim) # g(o), f(o,a), (g,f)
        self.critic = Critic(hid_dim, max_agents) # V値計算
        self.q_layer = Q(hid_dim,q_hid_dim) # Q値計算
        self.lr = lr
        self.max_agents = max_agents
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)
    
    def forward(self, obs: torch.Tensor, hidden: torch.Tensor = None) -> dict:
        r_outputs = dict()
        for b_obs in obs: # bobs = (num_agents,obs_dim)
            outputs = dict()
            active_obs = []
            for sobs in b_obs: # sobs = (obs_dim,)
                if sobs[0] == 1:
                    sobs = sobs.unsqueeze(0) # sobs = (1,obs_dim)
                    active_obs.append(sobs)
                    output = self.actor(sobs, hidden)
                    DictExtension.StackItems(outputs,output)
            DictExtension.reduceNone(outputs,lambda o: torch.stack(o,dim=1))
            actives = len(active_obs)
            active_obs = torch.stack(active_obs,dim=1)
            encoded_obs, _, encoded_counterfactual = self.state_encoder(active_obs,outputs['policy']) # g (1,num_agents,hid_dim), f (1, num_agents, hid_dim), (1, num_agents, num_agents, hid_dim)
            outputs['q_values'] = self.q_layer(encoded_counterfactual)
            outputs['encoded_obs'] = encoded_obs
            DictExtension.reduceNone(outputs,lambda o: TensorExtension.tensor_padding(o,self.max_agents,1))
            outputs['actives'] = torch.tensor([[actives]]) # (1,1) 切り出し用
            outputs['policy'] =outputs['policy'].view(-1,self.max_agents*self.act_dim)
            outputs['logits'] =outputs['logits'].view(-1,self.max_agents*self.act_dim)
            # ↓ 出力確認用
            # print(Printer.tensorDictPrint(outputs,False))
            DictExtension.StackItems(r_outputs,outputs)
        DictExtension.reduceNone(r_outputs,lambda o: torch.cat(o,dim=0))
        # ↓ 出力確認用
        # print(Printer.tensorDictPrint(r_outputs,False))
        return outputs
    
    def init_hidden(self, hidden=None):
        # RNNを使用しない場合、ダミーの隠れ状態を返す
        return None
    
    def load_state_dict(self, folderpath:str,strict=True):
        self.actor.load_state_dict(torch.load(folderpath+getActorModelName()),strict)
        self.critic.load_state_dict(torch.load(folderpath+getCriticModelName()),strict)

    def save_state_dict(self,folderpath:str):
        torch.save(self.actor.state_dict(),folderpath+getActorModelName())
        torch.save(self.critic.state_dict(),folderpath+getCriticModelName())

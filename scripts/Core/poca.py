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
    

# Actor class(Policy) 任意のネットワークアーキテクチャでよい
# Actor network
class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hid_dim: int = 128):
        super(Actor, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hid_dim = hid_dim
        self.hidden_layer1 = nn.Linear(obs_dim, hid_dim * 2) # first hidden layer
        self.hidden_layer2 = nn.Linear(hid_dim * 2, hid_dim) # second hidden layer
        self.action_layer = nn.Linear(hid_dim, act_dim) # action output layer
        self.softmax = nn.Softmax(dim=-1) # softmax activation function for log_prob output

    def forward(self, obs: torch.Tensor, hidden: torch.Tensor = None) -> dict:
        """
        Takes an observation tensor and a hidden state tensor as input and returns a dictionary containing action tensor, state value tensor, hidden state tensor, logits tensor, and log_prob tensor as output.
        obs: observation tensor of shape (batch_size, obs_dim)
        hidden: hidden state tensor of shape (batch_size, hid_dim)
        returns: a dictionary containing the following tensors:
            - policy: action tensor of shape (batch_size, act_dim)
            - value: state value tensor of shape (batch_size, 1)
            - hidden: hidden state tensor of shape (batch_size, hid_dim)
            - logits: logits tensor of shape (batch_size, act_dim)
            - log_prob: log_prob tensor of shape (batch_size, act_dim)
        """
        x = F.relu(self.hidden_layer1(obs)) # pass through the first hidden layer and apply relu
        x = F.relu(self.hidden_layer2(x)) # pass through the second hidden layer and apply relu
        logits = self.action_layer(x) # pass through the action layer for logits output
        log_prob = self.softmax(logits) # apply softmax for log_prob output
        return {'policy': logits, 'obs': obs, 'hidden': hidden, 'logits': logits, 'log_prob': log_prob}

# RSAブロックの定義
class RSABlock(nn.Module):
    def __init__(self, input_size: int, output_size: int, num_heads: int):
        super(RSABlock,self).__init__()
        self.input_size = input_size # 入力の次元
        self.output_size = output_size # 出力の次元
        self.num_heads = num_heads # 注意力のヘッド数
        # 入力を処理する全結合層
        self.input_layer = nn.Linear(input_size, output_size)
        # 注意力のためのクエリ、キー、バリューの全結合層
        self.query = nn.Linear(output_size, output_size)
        self.key = nn.Linear(output_size, output_size)
        self.value = nn.Linear(output_size, output_size)
        # 関係性のための全結合層
        self.relation = nn.Linear(output_size, output_size)
        # 注意力の出力を処理する全結合層
        self.output_layer = nn.Linear(output_size, output_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Takes an input tensor of shape (batch_size, num_agents, input_size) and returns an output tensor of shape (batch_size, num_agents, output_size).
        inputs: input tensor
        returns: output tensor
        """
        # 全結合層で特徴量を抽出する
        x = F.relu(self.input_layer(inputs)) # (batch_size, num_agents, output_size)
        # クエリ、キー、バリューを計算する
        query = self.query(x) # (batch_size, num_agents, output_size)
        key = self.key(x) # (batch_size, num_agents, output_size)
        value = self.value(x) # (batch_size, num_agents, output_size)
        # 関係性を計算する
        relation = self.relation(x) # (batch_size, num_agents, output_size)
        # クエリとキーの内積でAttentionの重みを計算する
        # (batch_size, num_agents, output_size) x (batch_size, num_agents, output_size) -> (batch_size, num_agents, num_agents)
        attention_weights = torch.matmul(query, key.transpose(-1, -2))
        # Attentionの重みをソフトマックスで正規化する
        attention_weights = F.softmax(attention_weights, dim=-1)
        # Attentionの重みとバリューの積でコンテキストベクトルを計算する
        # (batch_size, num_agents, num_agents) x (batch_size, num_agents, output_size) -> (batch_size, num_agents, output_size)
        context = torch.matmul(attention_weights, value)
        # 関係性を足す
        torch.add(context, relation, out=context) # (batch_size, num_agents, output_size)
        # 全結合層で処理する
        context = F.relu(self.output_layer(context)) # (batch_size, num_agents, output_size)
        return context


# MA-POCA Criticモデルの定義
class Critic(nn.Module):
    def __init__(self, input_size: int, hid_dim: int = 128, num_heads: int = 4):
        """
        価値関数を計算するCriticネットワーク
        param: input_size: Agentのエンコード後の観測テンソルの次元数 => (batch_size,num_agents,input_size)
        return: V値 (batch_size,1)
        """
        super(Critic,self).__init__()
        # 全エージェントの観測と行動を結合した入力の次元
        self.input_size = input_size
        # RSAブロック
        self.rsa = RSABlock(input_size + 1, hid_dim, num_heads)
        # 最終的な価値関数の出力層
        self.value_layer = nn.Linear(hid_dim, 1)

    def forward(self, inputs: torch.Tensor, remain: float) -> torch.Tensor:
        """
        エンコード後の全Agentの観測のテンソルを受け取って価値を返す
        param: inputs: エンコード後の観測テンソル (batch_size,num_agents,input_size)
        param: remain: 正規化したAgentの数
        returns: value (batch_size,1)
        """
        # 頭に残りのAgentの数をつけておく
        x = torch.cat([inputs,torch.tensor([[[remain]]*inputs.shape[1]]*inputs.shape[0])],dim=-1)
        # RSAブロックで注意力を計算する
        x = self.rsa(x) # (batch_size, num_agents, hid_dim + 1)
        # 最終的な価値関数の出力を計算する
        value = self.value_layer(x) # (batch_size, num_agents, 1)
        value = torch.sum(value, dim=1) # (batch_size, 1)
        return value

class Q(nn.Module):
    def __init__(self,input_dim: int, q_hid_dim: int = 64, rsa_heads: int = 4):
        super(Q,self).__init__()
        self.q_rsa = RSABlock(input_dim,q_hid_dim,rsa_heads) # Q値計算用RSA
        self.q_layer = nn.Linear(q_hid_dim,1) # Q値計算

    def forward(self,inputs: torch.Tensor):
        q_value = self.q_rsa(inputs) # (batch_size,num_agents,num_agents,q_hid_dim)
        q_value = self.q_layer(q_value) # Qψ (batch_size,num_agents,num_agents,1)
        q_value = torch.sum(q_value, dim=2) # (batch_size,num_agents, 1)
        return q_value

# ActorCritic class
class MAPOCA(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, max_agents: int, lr: float, hid_dim: int = 128, q_hid_dim: int = 64, q_rsa_heads: int = 4):
        super(MAPOCA, self).__init__()
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.actor = Actor(obs_dim, act_dim)
        self.obs_encoder = nn.Linear(obs_dim,hid_dim) # g(o)
        self.critic = Critic(hid_dim) # V値計算
        self.obs_act_encoder = nn.Linear(obs_dim+act_dim,hid_dim) # f(o,a)
        self.q_layer = Q(hid_dim,q_hid_dim) # Q値計算
        self.lr = lr
        self.max_agents = max_agents
        # Define the actor optimizer
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        # Define the critic optimizer
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)
        # Store experiences
        self.experiences = list()
    
    def forward(self, obs: torch.Tensor, hidden: torch.Tensor = None) -> dict:
        outputs = dict()
        actives = obs.shape[1]
        for i in range(actives):
            sobs = obs[:,i,:]
            if hidden is not None:
                hidden_s = hidden[:,i,:,:,:] # (batch_size,2,layers,hid_dim)
                hidden_s = (hidden_s[:,0,:,:],hidden_s[:,1,:,:]) #  (batch_size,layers,hid_dim),(batch_size,layers,hid_dim)
            else:
                hidden_s = None
            output = self.actor(sobs, hidden_s)
            for k, v in output.items():
                if isinstance(v,tuple):
                    v = torch.stack(list(v),dim=1).unsqueeze(1) # (batch_size,1(stack),2,layers,hid_dim)
                    if k in outputs:
                        outputs[k] = torch.cat([outputs[k],v], dim=1)
                    else:
                        outputs[k] = v 
                elif isinstance(v,torch.Tensor):
                    v = v.unsqueeze(1)
                    if k in outputs:
                        outputs[k] = torch.cat([outputs[k],v], dim=1)
                    else:
                        outputs[k] = v
        encoded_obs = self.obs_encoder(obs) # g (batch_size,num_agents,hid_dim)
        value = self.critic(encoded_obs,actives/self.max_agents) # Vφ (batch_size,1)
        acts = outputs['policy'] # (batch_size,num_agents,act_dim)
        encoded_obs_acts = self.obs_act_encoder(torch.cat([obs,acts],dim=-1)) # g (batch_size,num_agents,hid_dim)
        combineds = None
        for i in range(actives):
            sencobs = encoded_obs[:,i,:].unsqueeze(1)
            others = TensorExtension.extractSelect(encoded_obs_acts,[i],1)
            combined = torch.cat([sencobs,others],dim=1) if others is not None else sencobs
            combined = combined.unsqueeze(1)
            if combineds is None:
                combineds = combined
            else:
                combineds = torch.cat([combineds,combined],dim=1)
        outputs['q_values'] = self.q_layer(combineds)
        outputs['active'] = torch.tensor([actives]).unsqueeze(1)
        outputs['value'] = value
        outputs['policy'] = TensorExtension.tensor_padding(outputs['policy'],self.max_agents,1)
        outputs['logits'] = TensorExtension.tensor_padding(outputs['logits'],self.max_agents,1)
        outputs['policy'] = outputs['policy'].view(-1,self.max_agents*self.act_dim)
        outputs['logits'] = outputs['logits'].view(-1,self.max_agents*self.act_dim)
        
        """ # ↓ 出力確認用
        print("{")
        for okey,output in outputs.items():
            print(f"\t'{okey}' : {Printer.tensorPrint(output,False)}")
        print("}") """
        return outputs
    

    # Training function
    def train(self, gamma, lam):
        # Initialize the actor and critic losses
        actor_loss = 0
        critic_loss = 0

        # Loop over the experiences in reverse order
        for obs, act, next_obs, reward, done in reversed(self.experiences):
            # Compute the target value function using the next observation and the target critic network
            with torch.no_grad():
                next_value = self.critic(next_obs, self.actor(next_obs)['policy'])
                # Mask the next value by the done flag
                next_value = next_value * (1 - done)
                # Compute the target Q-value using the reward and the discount factor
                target_q = reward + gamma * next_value

            # Compute the current Q-value using the observation and the critic network
            current_q = self.critic(obs, act)
            # Compute the critic loss as the mean squared error
            critic_loss += F.mse_loss(current_q, target_q)

            # Compute the current value function using the observation and the critic network
            current_value = self.critic(obs, self.actor(obs)['policy'])
            # Compute the advantage function as the difference between the Q-value and the value function
            advantage = current_q - current_value
            # Compute the actor loss as the negative of the expected log probability weighted by the advantage
            actor_loss -= (self.actor(obs)['log_prob'] * advantage).mean()

        # Update the actor and critic networks using the optimizers
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update the target networks using the polyak averaging
        self.actor.update_target(lam)
        self.critic.update_target(lam)


            
    def addExperience(self, obs, act, n_obs,reward, num_active,done):
        # obs: (num_agents, obs_dim)
        # act: (num_agents, act_dim)
        # n_obs: (num_agents, obs_dim) 
        # reward: scaler value of team (total) reward
        # num_active: scaler value of number of alive agents
        # done: episode ends
        self.experiences.append((obs, act, n_obs, reward, num_active, done))

    def init_hidden(self, hidden=None):
        # RNNを使用しない場合、ダミーの隠れ状態を返す
        return None
    
    def load_state_dict(self, folderpath:str,strict=True):
        self.actor.load_state_dict(torch.load(folderpath+getActorModelName()),strict)
        self.critic.load_state_dict(torch.load(folderpath+getCriticModelName()),strict)

    def save_state_dict(self,folderpath:str):
        torch.save(self.actor.state_dict(),folderpath+getActorModelName())
        torch.save(self.critic.state_dict(),folderpath+getCriticModelName())

from typing import Any, Mapping
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import *
import copy
import numpy as np
from ..Helper.Printer import Printer
from gymnasium import spaces
from ASRCAISim1.addons.HandyRLUtility.model import ModelBase

"""
class ModelBase(nn.Module):
    ""ASRCAISim1のHandyRLサンプルを使用する際のNNモデルの基底クラス。
    出力のキー'policy'には各action要素の出力をconcatしたものを与える。
    ""
    def __init__(self, obs_space, ac_space, action_dist_class, model_config):
        super().__init__()
        self.observation_space = obs_space
        self.action_space = ac_space
        self.action_dist_class = action_dist_class
        self.action_dim = self.action_dist_class.get_param_dim(ac_space) #=出力ノード数
        self.model_config = copy.deepcopy(model_config)
    def forward(self, obs, state, seq_len=None, mask=None):
        raise NotImplementedError
        ""
        return {
            'policy': p,
            'value': 0.0,
            'return': 0.0
        }
        ""
"""

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
    def __init__(self, obs_dim, act_dim, hid_dim = 128):
        super(Actor, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hid_dim = hid_dim
        self.lastobs:torch.Tensor = None
        self.lasthidden:tuple = None
        self.env_predictor = EnvPredictor(obs_dim, hid_dim) # EnvPredictor instance
        self.fc1 = nn.Linear(obs_dim, hid_dim*2) # first fully connected layer
        self.fc2 = nn.Linear(hid_dim*2, hid_dim) # second fully connected layer
        self.fc3 = nn.Linear(hid_dim, act_dim) # third fully connected layer for action output
        self.fc4 = nn.Linear(hid_dim, 1) # fourth fully connected layer for state value output
        # self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1) # softmax activation function for log_prob output

    def forward(self, obs, hidden = None):
        # obs: observation tensor of shape (batch_size, obs_dim)
        # hidden: hidden state tensor of shape (batch_size, hid_dim)
        # returns: a dictionary containing action tensor of shape (batch_size, act_dim), state value tensor of shape (batch_size, 1), next observation tensor of shape (batch_size, obs_dim), hidden state tensor of shape (batch_size, hid_dim), hidden cell tensor of shape (batch_size, hid_dim), logits tensor of shape (batch_size, act_dim), and log_prob tensor of shape (batch_size, act_dim)
        # print(Printer.tensorPrint(obs,False))
        """
        obs = obs.unsqueeze(1)
        # update the weights of the env_predictor based on the error between the current obs and the predicted obs
        if self.lastobs != None:
            p_obs, _ = self.env_predictor(self.lastobs, self.lasthidden) # get the next obs and the hidden state from the env_predictor
            error = torch.sum((obs - p_obs) ** 2, dim=-1) # compute the squared error between the current obs and the predicted obs
            loss = torch.mean(error).detach().requires_grad_(True) # compute the mean loss
            self.env_predictor.optimizer.zero_grad() # reset the gradients of the env_predictor
            loss.backward() # backpropagate the loss
            self.env_predictor.optimizer.step() # update the weights of the env_predictor
            print(f"Predict Env MSE:{loss.item():.4f}")
        n_obs, hidden = self.env_predictor(obs, hidden) # get the next obs and the hidden state from the env_predictor
        n_obs = n_obs.squeeze(1)
        self.lastobs = obs
        self.lasthidden = hidden
        obs = obs.squeeze(1)
        """
        # use the current obs and the next obs to output the action and the state value
        #x = torch.cat((obs, n_obs), dim=-1) # concatenate the current obs and the next obs
        # print("Actor cat:",Printer.tensorPrint(x,False))
        x = F.relu(self.fc1(obs)) # pass through the first layer and apply relu
        # print("Actor fc1:",Printer.tensorPrint(x,False))
        x = F.relu(self.fc2(x)) # pass through the second layer and apply relu
        # print("Actor fc2:",Printer.tensorPrint(x,False))
        # x = self.tanh(x)
        # print("Actor tanh:",Printer.tensorPrint(x,False))
        logits = self.fc3(x) # pass through the third layer for logits output
        # print("Actor fc3:",Printer.tensorPrint(logits,False))
        value = self.fc4(x) # pass through the fourth layer for state value output
        log_prob = self.softmax(logits) # apply softmax for log_prob output
        return {'policy':logits,'logits':logits,'q_value': value , 'hidden': hidden, 'log_prob': log_prob} #'n_obs':n_obs 

# EnvPredictor network
class EnvPredictor(nn.Module):
    def __init__(self, obs_dim, hid_dim = 128):
        super(EnvPredictor, self).__init__()
        self.obs_dim = obs_dim
        self.hid_dim = hid_dim
        self.lstm = nn.LSTM(obs_dim, hid_dim, batch_first=True) # LSTM layer
        self.fc = nn.Linear(hid_dim, obs_dim) # fully connected layer for next observation output
        self.optimizer = torch.optim.Adam(self.parameters()) # optimizer

    def forward(self, obs = None, hidden = None):
        # print("EnvPredictor:",Printer.tensorPrint(hidden,False))
        # obs: observation tensor of shape (batch_size, seq_len, obs_dim)
        # pass the obs through the LSTM layer
        x, (hidden, cell) = self.lstm(obs, hidden) # x: (batch_size, seq_len, hid_dim), hidden: (batch_size,layers, hid_dim), cell: (batch_size,layers, hid_dim)
        # pass the output through the fully connected layer for next obs output
        n_obs = self.fc(x) # n_obs: (batch_size, seq_len, obs_dim)
        return n_obs, (hidden, cell)


# RSAブロックの定義
class RSABlock(nn.Module):
    def __init__(self, input_size: int, output_size: int, num_heads: int):
        super(RSABlock,self).__init__()
        self.input_size = input_size # 入力の次元
        self.output_size = output_size # 出力の次元
        self.num_heads = num_heads # 注意力のヘッド数
        # 入力を処理する全結合層
        self.fc1 = nn.Linear(input_size, output_size)
        # 注意力のためのクエリ、キー、バリューの全結合層
        self.query = nn.Linear(output_size, output_size)
        self.key = nn.Linear(output_size, output_size)
        self.value = nn.Linear(output_size, output_size)
        # 関係性のための全結合層
        self.relation = nn.Linear(output_size, output_size)
        # 注意力の出力を処理する全結合層
        self.fc2 = nn.Linear(output_size, output_size)

    def forward(self, inputs: torch.Tensor):
        # inputs: (batch_size, num_agents, input_size)のテンソル
        # 全結合層で特徴量を抽出する
        x = F.relu(self.fc1(inputs)) # (batch_size, num_agents, output_size)
        # print(Printer.tensorPrint(x,False))
        # クエリ、キー、バリューを計算する
        query = self.query(x).view(-1, self.num_heads, self.output_size // self.num_heads) # (batch_size * num_agents, num_heads, output_size // num_heads)
        key = self.key(x).view(-1, self.num_heads, self.output_size // self.num_heads) # (batch_size * num_agents, num_heads, output_size // num_heads)
        value = self.value(x).view(-1, self.num_heads, self.output_size // self.num_heads) # (batch_size * num_agents, num_heads, output_size // num_heads)
        # 関係性を計算する
        relation = self.relation(x).view(-1, self.num_heads, self.output_size // self.num_heads) # (batch_size * num_agents, num_heads, output_size // num_heads)
        # クエリとキーの内積でAttentionの重みを計算する
        # (batch_size * num_agents, num_heads, output_size // num_heads) x (batch_size * num_agents, output_size // num_heads, num_agents) -> (batch_size * num_agents, num_heads, num_agents)
        attention_weights = torch.bmm(query, key.transpose(1, 2))
        # Attentionの重みをソフトマックスで正規化する
        attention_weights = F.softmax(attention_weights, dim=-1)
        # Attentionの重みとバリューの積でコンテキストベクトルを計算する
        # (batch_size * num_agents, num_heads, num_agents) x (batch_size * num_agents, num_heads, output_size // num_heads) -> (batch_size * num_agents, num_heads, output_size // num_heads)
        context = torch.bmm(attention_weights, value)
        # 関係性を足す
        context += relation
        # ヘッドを結合する
        context = context.view(-1, self.output_size) # (batch_size * num_agents, output_size)
        # 全結合層で処理する
        context = F.relu(self.fc2(context))
        # 元の形に戻す
        context = context.view(-1, inputs.shape[1], self.output_size) # (batch_size, num_agents, output_size)
        return context

# MA-POCA Criticモデルの定義
class Critic(nn.Module):
    def __init__(self, observation_size: int, action_size: int, hidden_dim: int = 128, num_heads: int = 4):
        super(Critic,self).__init__()
        self.observation_size = observation_size
        self.action_size = action_size
        # 全エージェントの観測と行動を結合した入力の次元
        self.input_size = (observation_size + action_size)
        # 全エージェントの観測と行動を結合した入力を処理する全結合層
        self.fc1 = nn.Linear(self.input_size, hidden_dim)
        # RSAブロック
        self.rsa = RSABlock(hidden_dim, hidden_dim//2, num_heads)
        # 最終的な価値関数の出力層
        self.fc2 = nn.Linear(hidden_dim//2, 1)

    def forward(self, observations: torch.Tensor, actions: torch.Tensor):
        # observations: (batch_size, num_agents, observation_size)のテンソル
        # actions: (batch_size, num_agents, action_size)のテンソル
        # 全エージェントの観測と行動を結合して(batch_size, input_size)のテンソルにする
        #print(Printer.tensorPrint(observations,False))
        #print(Printer.tensorPrint(actions,False))
        inputs = torch.cat([observations, actions], dim=-1).view(-1, self.input_size)
        # print(Printer.tensorPrint(inputs,False))
        # 全結合層で特徴量を抽出する
        x = F.relu(self.fc1(inputs))
        # print(Printer.tensorPrint(x,False))
        # RSAブロックで注意力を計算する
        x = self.rsa(x.view(-1,observations.shape[1],x.shape[1])) # (batch_size, num_agents, 64)
        # print(Printer.tensorPrint(x,False))
        value = 0
        for i in range(observations.shape[1]):
            # 対象エージェントの注意力を取り出す
            # 最終的な価値関数の出力を計算する
            value += self.fc2(x[:, i, :])
        return value


# ActorCritic class
class MAPOCA(nn.Module):
    def __init__(self, obs_dim, act_dim, max_agents, lr):
        super(MAPOCA, self).__init__()
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        # Define the actor network
        self.actor = Actor(obs_dim, act_dim)
        # Define the critic network
        self.critic = Critic(obs_dim, act_dim)
        # Define the learning rate
        self.lr = lr
        self.max_agents = max_agents
        # Define the actor optimizer
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        # Define the critic optimizer
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)
        # Store experiences
        self.experiences = list()
    
    def forward(self, obs, hidden=None):
        # Take a vector observations and active agents (batch_size,num_agents,) boolean array as input and return the action manipulating variable and y value
        rets = dict()
        actives = obs.shape[1]
        # print("actives:",actives)
        for i in range(actives):
            sobs = obs[:,i,:]
            if hidden != None:
                hidden_s = hidden[:,i,:,:,:] # (batch_size,2,layers,hid_dim)
                # print(Printer.tensorPrint(hidden_s,False))
                hidden_s = (hidden_s[:,0,:,:],hidden_s[:,1,:,:]) #  (batch_size,layers,hid_dim),(batch_size,layers,hid_dim)
            else:
                hidden_s = None
            ret = self.actor(sobs, hidden_s)
            for k, v in ret.items():
                if isinstance(v,tuple):
                    v = torch.stack(list(v),dim=1).unsqueeze(1) # (batch_size,1(stack),2,layers,hid_dim)
                    if k in rets:
                        rets[k] = torch.cat([rets[k],v], dim=1)
                    else:
                        rets[k] = v 
                elif isinstance(v,torch.Tensor):
                    v = v.unsqueeze(1)
                    if k in rets:
                        rets[k] = torch.cat([rets[k],v], dim=1)
                    else:
                        rets[k] = v
        value = self.critic(obs, rets['policy'])
        rets['active'] = torch.tensor([actives]).unsqueeze(1)
        rets['value'] = value.unsqueeze(1)
        if actives < self.max_agents:
            pads = torch.stack([torch.zeros_like(rets['policy'][:,0,:]) for _ in range(self.max_agents - actives)],dim=1)
            rets['policy'] = torch.cat([rets['policy'],pads], dim=1)
            pads = torch.stack([torch.zeros_like(rets['logits'][:,0,:]) for _ in range(self.max_agents - actives)],dim=1)
            rets['logits'] = torch.cat([rets['logits'],pads], dim=1)
        rets['policy'] = rets['policy'].view(-1,self.act_dim*self.max_agents)
        # 今回はpolicyにlogitsを渡しているため、logitsも同じものをセットしておく
        rets['logits'] = rets['logits'].view(-1,self.act_dim*self.max_agents)
        """
        for k in rets:
            ret = rets[k]
            print(k,Printer.tensorPrint(ret,False))
        """

        return rets
    
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

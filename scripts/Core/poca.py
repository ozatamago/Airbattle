from typing import Any, Mapping
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import *
import copy
from ..Helper.Tensor import Tensor
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
class Actor(nn.Module):
    
    def __init__(self, obs_dim,act_dim):
        super(Actor, self).__init__()
        hidden_dim = 128 #あとでhyperparameterに
        # Define the network layers
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, act_dim)
        self.obsp = EnvironmentPredictor(hidden_dim,obs_dim)
        self.value = nn.Linear(hidden_dim, 1) # 状態価値を出力するモデル
    
    def forward(self, obs, hidden=None):
        # obsは自分と味方の観測情報と視界を結合して正規化したベクトル 
        # hiddenはLSTMの隠れ状態とセル状態
        print("Observation:",obs)
        print("shape:",obs.shape)
        x = F.relu(self.fc1(obs)) # 全結合層とReLU関数
        print("fc1:shape:",x.shape)
        x = F.relu(self.fc2(x)) # 全結合層とReLU関数
        print("fc2:shape:",x.shape)
        self.update_obsp(x)
        next_obs, x = self.obsp(x, hidden)
        # 全結合層でActionProb(embedded_dim,)を計算する
        action_logits = self.fc3(x)
        # ソフトマックス関数で確率分布に変換する
        action_probs = F.softmax(action_logits, dim=-1) # ソフトマックス関数
        value = self.value(x) # 状態価値を出力するモデル
        # action_probs = F.softmax(action_logits, dim=-1)
        # B=getBatchSize(obs,self.observation_space)
        # print("B: ", B)
        # print("self.action_dim: ", self.action_dim)
        # p = torch.ones([B,self.action_dim],dtype=torch.float32)
        print(f"action_probs:shape{action_probs.shape}",action_probs)
        print(f"action_probs:shape{next_obs.shape}",next_obs)
        print(f"action_probs:shape{value.shape}",value)
        ret = {"policy": action_probs.ravel(), "n_obs": next_obs.ravel(), "value": value.ravel()}
        print(ret)
        return ret
    
    def init_hidden(self,hidden=None):
        # RNNを使用しない場合、ダミーの隠れ状態を返す
        return None
    
    def update_obsp(self,obs):
        if self.obsp.obsMemory.shape[1] > 0:
            # EnvironmentPredictorの重みを修正するための関数
            # lossは、EnvironmentPredictorの出力と目標値との誤差
            # 勾配を初期化する
            self.obsp.optimizer.zero_grad()
            # 誤差逆伝播法で勾配を計算する
            self.obsp.loss(self.obsp.forward(),obs).backward()
            # 勾配をクリップする
            torch.nn.utils.clip_grad_norm_(self.obsp.parameters(), 1.0)
            # 重みを更新する
            self.obsp.optimizer.step()
    

class EnvironmentPredictor(nn.Module):
    # A model that predicts the next environment
    def __init__(self, embedded_dim, obs_dim, batch_size=2, max_memory=5, num_heads=4):
        super(EnvironmentPredictor, self).__init__()
        self.max_memory = max_memory
        self.obsMemory = torch.empty(batch_size, 0, embedded_dim)
        self.lstm = nn.LSTM(embedded_dim, embedded_dim) # LSTM
        self.attn = nn.MultiheadAttention(embedded_dim, num_heads) # Attention mechanism
        self.pred = nn.Linear(embedded_dim, obs_dim)
        self.loss = nn.MSELoss() # Mean squared error
        self.optimizer = torch.optim.Adam(self.parameters()) # Optimization algorithm
    
    def forward(self, embedded_obs = None, hidden=None):
        """
        :params embedded_obs: shape == (batch_size, embedded_dim)
        :params hidden
        :return n_obs: the prediction of the next observation, shape == (batch_size, obs_dim)
                hidden: the hidden state of the LSTM, shape == (batch_size, embedded_dim)
        """
        if embedded_obs != None:
            self.obsMemory = torch.cat([self.obsMemory, embedded_obs.unsqueeze(1)], dim=1)
            if self.obsMemory.shape[1] > self.max_memory: # If the sequence length exceeds the maximum length
                self.obsMemory = torch.narrow(self.obsMemory, 1, 1, self.max_memory) # Cut out the elements from the first dimension of obsMemory, starting from the first element, for max_len elements
        x, hidden = self.lstm(self.obsMemory, hidden) # LSTM
        x, _ = self.attn(x, x, x) # Attention mechanism
        # Get the output of the last time step
        x = x[:, -1, :]
        n_obs = self.pred(x) # Predict the next environment
        return n_obs, x

class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hid_dim = 128):
        super(Critic,self).__init__()
        # 入力の次元数を保存
        self.in_dim = obs_dim + act_dim
        # Attentionのための線形層を一つにまとめる
        self.att = nn.Linear(self.in_dim, hid_dim * 3)
        # 状態価値関数の出力層
        self.out = nn.Linear(hid_dim, 1)

    def forward(self, obs, act):
        # 入力をエージェントごとに結合
        x = torch.cat([obs, act], dim=-1)
        # Attentionのための線形変換を一度に行う
        q, k, v = self.att(x).chunk(3, dim=-1)
        # Attentionのためのスコア計算
        score = torch.matmul(q, k.transpose(-2, -1))
        # Attentionのための正規化
        score = F.softmax(score / torch.sqrt(torch.tensor(self.in_dim, dtype=torch.float32)), dim=-1)
        # Attentionのための重み付き和
        att = torch.matmul(score, v)
        # 状態価値関数の出力
        out = self.out(att)
        return out

# ActorCritic class
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, num_agents, lr):
        print("obs_dim: ", obs_dim)
        print("act_dim: ", act_dim)
        print("num_agents: ", num_agents)
        # print("lr: ", lr)
        super(ActorCritic, self).__init__()
        # Actorクラスの初期化部分
        #obs_dim = 78  # Box型から取得した観測空間の次元数
        #act_dim = sum([11, 9, 3, 5, 6])  # MultiDiscrete型から取得したアクションの総数
        #num_agents = 1  # 例として2を設定
        # Define the actor network
        self.actor = Actor(obs_dim, act_dim)
        # Define the critic network
        self.critic = Critic(obs_dim, act_dim)
        # Define the learning rate
        self.lr = lr
        # Define the actor optimizer
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        # Define the critic optimizer
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)
         # Store experiences
        self.experiences = list()
       
    
    def forward(self, obs,hidden=None):
        # Take a vector observations and active agents (num_agents,) boolean array as input and return the action manipulating variable and y value
        rets = list()
        for sobs in obs:
            rets.append(self.actor(sobs,hidden))
        value = self.critic(obs, Tensor.tensorify([ret['policy'] for ret in rets]))
        return rets[0], value
    
    # Training function
    def train(self, gamma, lam):
        # gamma: the discount factor
        # lam: the lambda parameter for TD lambda
        
        # Initialize the eligibility trace
        z = torch.zeros(self.num_agents, self.act_dim)
        # Initialize the TD error
        delta = torch.zeros(self.num_agents, 1)
        # Initialize the team reward
        R = 0
        # Initialize the actor's loss
        actor_loss = 0 # Use a scalar value
        # Initialize the critic's loss
        critic_loss = 0
        # Loop over the experiences in reverse order
        for obs, act, rew, next_obs, done in reversed(self.experiences):
            # Compute the team reward
            R = rew + gamma * R * (1 - done)
            # Compute the value for the current and next observations
            _,value = self.critic(obs, act)
            _,next_value = self.critic(next_obs, act)
            # Compute the TD error
            delta = R + gamma * next_value - value
            # Update the eligibility trace
            z = gamma * lam * z * (1 - done) + act.log_prob()
            # Update the actor's loss
            actor_loss -= (delta * z).mean()
            # Update the critic's loss
            critic_loss += F.mse_loss(value, R)
        # Perform a gradient step for the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        # Perform a gradient step for the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
            
    def addExperience(self,state,act,reward,done,n_state):
        # state: (num_agents, obs_dim)
        # act: (num_agents, act_dim)
        # active: (num_agents,) boolean tensor indicating which agents are active
        # reward: scaler value of team reward
        # done: episode ends
        # n_state:  (num_agents, obs_dim) next state from env.step()
        # n_active:  (num_agents,) boolean tensor indicating which agents are active from environment
        self.experiences.append((state,act,reward,n_state,done))

    def init_hidden(self,hidden=None):
        # RNNを使用しない場合、ダミーの隠れ状態を返す
        return None
    
    def load_state_dict(self, folderpath:str,strict=True):
        self.actor.load_state_dict(torch.load(folderpath+getActorModelName()),strict)
        self.critic.load_state_dict(torch.load(folderpath+getCriticModelName()),strict)

    def save_state_dict(self,folderpath:str):
        torch.save(self.actor.state_dict(),folderpath+getActorModelName())
        torch.save(self.critic.state_dict(),folderpath+getCriticModelName())

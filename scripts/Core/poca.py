from typing import Any, Mapping
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import *
import copy
from ..Helper import Tensor
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
class Actor(ModelBase,nn.Module):
    def __init__(self, obs_space, ac_space, action_dist_class, model_config):
        super().__init__(obs_space, ac_space, action_dist_class, model_config)
        self.observation_space = obs_space
        self.action_space = ac_space
        self.action_dist_class = action_dist_class
        self.action_dim = self.action_dist_class.get_param_dim(ac_space) #=出力ノード数
        self.model_config = copy.deepcopy(model_config)
        obs_dim = obs_space.shape[0]
        act_dim = len(ac_space.nvec)
        print(ac_space)

        # Define the network layers
        self.fc1 = nn.Linear(obs_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, self.action_dim)
        self.attn = nn.MultiheadAttention(64, 4) # Attentionメカニズム
        self.lstm = nn.LSTM(64, 64) # LSTM
        self.pred = nn.Linear(64, obs_dim) # 次の環境を予測するモデル
        self.value = nn.Linear(64, 1) # 状態価値を出力するモデル
        self.loss = nn.MSELoss() # 平均二乗誤差
        self.optimizer = torch.optim.Adam(self.parameters()) # 最適化アルゴリズム

        # Save the action distribution class
        self.action_dist_class = action_dist_class
    
    def forward(self, obs, hidden=None):
        # obsは自分と味方の観測情報と視界を結合して正規化したベクトル 
        # hiddenはLSTMの隠れ状態とセル状態
        x = F.relu(self.fc1(obs)) # 全結合層とReLU関数
        x = F.relu(self.fc2(x)) # 全結合層とReLU関数
        x = x.unsqueeze(0) # Attentionメカニズムに入力するために次元を追加
        x, _ = self.attn(x, x, x) # Attentionメカニズム
        x = x.squeeze(0) # 次元を削除
        x, hidden = self.lstm(x, hidden) # LSTM
        action_logits = self.fc3(x) # 全結合層
        action_probs = F.softmax(action_logits, dim=-1) # ソフトマックス関数
        next_obs = self.pred(x) # 次の環境を予測するモデル
        value = self.value(x) # 状態価値を出力するモデル
        # action_probs = F.softmax(action_logits, dim=-1)

        # B=getBatchSize(obs,self.observation_space)
        # print("B: ", B)
        # print("self.action_dim: ", self.action_dim)
        # p = torch.ones([B,self.action_dim],dtype=torch.float32)
        ret = {"policy": action_probs, "n_obs": next_obs, "value": value}
        print(ret)
        return ret
    
    def init_hidden(self):
        # RNNを使用しない場合、ダミーの隠れ状態を返す
        return None
    
    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        return super().load_state_dict(state_dict, strict)
    
    def state_dict(self):
        return super().state_dict()
    
    def parameters(self):
        return super().parameters()
    

# to update policy parameter
# Critic class
class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, num_agents):
        super(Critic, self).__init__()
        # Define the network layers
        self.fc1 = nn.Linear(obs_dim + act_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        # Define the attention layer
        self.attn = nn.MultiheadAttention(64, 4) # 4 heads
        # Define the mask for inactive agents
        self.mask = torch.ones(num_agents, num_agents)
    
    def forward(self, obs, act, active):
        # Take all observations and actions as input and return a scalar value
        # obs: (num_agents, obs_dim)
        # act: (num_agents, act_dim)
        # active: (num_agents,) boolean tensor indicating which agents are active
        # Concatenate observations and actions
        x = torch.cat([obs, act], dim=1)
        # Apply the first fully connected layer
        x = F.relu(self.fc1(x))
        # Apply the attention layer
        # Reshape x to (seq_len, batch_size, embed_dim)
        x = x.transpose(0, 1).unsqueeze(0)
        # Create the mask for the attention layer
        mask = self.mask.clone()
        mask[~active, :] = 0 # Set the rows of inactive agents to zero
        mask[:, ~active] = 0 # Set the columns of inactive agents to zero
        mask = mask.bool() # Convert to boolean tensor
        # Apply the attention layer
        x, _ = self.attn(x, x, x, key_padding_mask=mask)
        # Reshape x back to (batch_size, seq_len, embed_dim)
        x = x.squeeze(0).transpose(0, 1)
        # Apply the second fully connected layer
        x = F.relu(self.fc2(x))
        # Apply the third fully connected layer
        x = self.fc3(x)
        # Sum over the agents dimension
        x = x.sum(dim=0)
        return x

# ActorCritic class
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, num_agents, lr):
        print("obs_dim: ", obs_dim)
        print("act_dim: ", act_dim)
        print("num_agents: ", num_agents)
        # print("lr: ", lr)
        super(ActorCritic, self).__init__()
        # Actorクラスの初期化部分
        obs_dim = 78  # Box型から取得した観測空間の次元数
        act_dim = sum([11, 9, 3, 5, 6])  # MultiDiscrete型から取得したアクションの総数
        num_agents = 1  # 例として2を設定
        # Define the actor network
        self.actor = Actor(obs_dim, act_dim)
        # Define the critic network
        self.critic = Critic(obs_dim, act_dim, num_agents)
        # Define the learning rate
        self.lr = lr['model']['hyperParameters']['actor']['learningRate']
        # Define the actor optimizer
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        # Define the critic optimizer
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
         # Store experiences
        self.experiences = list()
       
    
    def forward(self, obs,active):
        # Take a vector observations and active agents (num_agents,) boolean array as input and return the action manipulating variable and y value
        rets = list()
        for sobs in obs:
            rets.append(self.actor(sobs))
        value = self.critic(obs, Tensor.tensorify([ret['policy'] for ret in rets]), active) 
        return rets, value
    
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
        for obs, act, rew, next_obs, done, active in reversed(self.experiences):
            # Compute the team reward
            R = rew + gamma * R * (1 - done)
            # Compute the value for the current and next observations
            _,value = self.critic(obs, act, active)
            _,next_value = self.critic(next_obs, act, active)
            # Compute the TD error
            delta = R - value
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
            
    def addExperience(self,state,act,active,reward,done,n_state):
        # state: (num_agents, obs_dim)
        # act: (num_agents, act_dim)
        # active: (num_agents,) boolean tensor indicating which agents are active
        # reward: scaler value of team reward
        # done: episode ends
        # n_state:  (num_agents, obs_dim) next state from env.step()
        # n_active:  (num_agents,) boolean tensor indicating which agents are active from environment
        self.experiences.append((state,act,reward,n_state,done,active))

    def init_hidden(self):
        # RNNを使用しない場合、ダミーの隠れ状態を返す
        return None
    
    def load_state_dict(self, folderpath:str,strict=True):
        self.actor.load_state_dict(torch.load(folderpath+getActorModelName()),strict)
        self.critic.load_state_dict(torch.load(folderpath+getCriticModelName()),strict)

    def save_state_dict(self,folderpath:str):
        torch.save(self.actor.state_dict(),folderpath+getActorModelName())
        torch.save(self.critic.state_dict(),folderpath+getCriticModelName())

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
        x = F.leaky_relu(self.hidden_layer1(obs)) # pass through the first hidden layer and apply relu
        x = F.leaky_relu(self.hidden_layer2(x)) # pass through the second hidden layer and apply relu
        logits = self.action_layer(x) # pass through the action layer for logits output
        log_prob = self.softmax(logits) # apply softmax for log_prob output
        return {'policy': logits, 'obs': obs, 'hidden': hidden, 'logits': logits, 'log_prob': log_prob}

class RSABlock(nn.Module):
    def __init__(self, input_size: int, output_size: int, num_heads: int):
        super(RSABlock,self).__init__()
        self.input_layer = nn.Linear(input_size, output_size)
        self.output_layer = nn.Linear(output_size, output_size)
        self.attention = nn.MultiheadAttention(output_size, num_heads)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(self.input_layer(inputs)) # (batch_size, num_agents, output_size)
        attention_output, _ = self.attention(x, x, x) # (batch_size, num_agents, output_size)
        context = torch.add(attention_output, x) # (batch_size, num_agents, output_size)
        context = F.leaky_relu(self.output_layer(context)) # (batch_size, num_agents, output_size)
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
        q_value = self.q_rsa(inputs.view(-1,inputs.shape[-2],inputs.shape[-1])) # (batch_size*num_agents,num_agents,q_hid_dim)
        q_value = q_value.view(-1,inputs.shape[1],inputs.shape[2],q_value.shape[-1])
        q_value = self.q_layer(q_value) # Qψ (batch_size,num_agents,num_agents,1)
        q_value = torch.sum(q_value, dim=2) # (batch_size,num_agents, 1)
        return q_value

class ObservationEncoder(nn.Module):
    def __init__(self,obs_dim: int, out_dim: int):
        super(ObservationEncoder,self).__init__()
        self.encode_layer = nn.Linear(obs_dim,out_dim) # Q値計算

    def forward(self,observations: torch.Tensor):
        encoded = self.encode_layer(observations)
        return encoded

class ObservationActionPairEncoder(nn.Module):
    def __init__(self,obs_dim: int,act_dim: int, out_dim: int):
        super(ObservationActionPairEncoder,self).__init__()
        self.encode_layer = nn.Linear(obs_dim+act_dim,out_dim) # Q値計算

    def forward(self,observations: torch.Tensor,actions: torch.Tensor):
        encoded = self.encode_layer(torch.cat([observations,actions],dim=-1))
        return encoded


# ActorCritic class
class MAPOCA(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, max_agents: int, lr: float, hid_dim: int = 128, q_hid_dim: int = 64, q_rsa_heads: int = 4):
        super(MAPOCA, self).__init__()
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.actor = Actor(obs_dim, act_dim)
        self.obs_encoder = ObservationEncoder(obs_dim,hid_dim) # g(o)
        self.critic = Critic(hid_dim) # V値計算
        self.obs_act_encoder = ObservationActionPairEncoder(obs_dim,act_dim,hid_dim) # f(o,a)
        self.q_layer = Q(hid_dim,q_hid_dim) # Q値計算
        self.lr = lr
        self.max_agents = max_agents
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)
        # Store experiences
        self.experiences = list()
    
    def forward(self, obs: torch.Tensor, hidden: torch.Tensor = None) -> dict:
        r_outputs = dict()
        for b_obs in obs: # bobs = (num_agents,obs_dim)
            outputs = dict()
            actives = 0
            for sobs in b_obs: # sobs = (obs_dim,)
                if sobs[0] == 1:
                    actives += 1
                    sobs = sobs.unsqueeze(0) # sobs = (1,obs_dim)
                    output = self.actor(sobs, hidden)
                    for k, v in output.items():
                        if v is None:
                            outputs[k] = None
                        elif k in outputs:
                            outputs[k].append(v)
                        else:
                            outputs[k] = [v]
            for o_key in outputs.keys():
                if outputs[o_key] is not None:
                    outputs[o_key] = torch.stack(outputs[o_key],dim=1)
            encoded_obs = self.obs_encoder(outputs['obs']) # g (1,num_agents,hid_dim)
            encoded_obs_acts = self.obs_act_encoder(outputs['obs'],outputs['policy']) # g (1,num_agents,hid_dim)
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
            outputs['combineds'] = combineds
            for o_key in outputs.keys():
                outputs[o_key] = TensorExtension.tensor_padding(outputs[o_key],self.max_agents,1)
            outputs['active'] = torch.tensor([[actives]]) # agents (1,1)
            outputs['value'] = self.critic(encoded_obs,actives/self.max_agents) # Vφ (1,1)
            outputs['policy'] =outputs['policy'].view(-1,self.max_agents*self.act_dim)
            outputs['logits'] =outputs['logits'].view(-1,self.max_agents*self.act_dim)
            # ↓ 出力確認用
            """ print("{")
            for okey,output in outputs.items():
                print(f"\t'{okey}' : {Printer.tensorPrint(output,False)}")
            print("}") """
            for k, v in outputs.items():
                if v is None:
                    r_outputs[k] = None
                elif k in r_outputs:
                    r_outputs[k].append(v)
                else:
                    r_outputs[k] = [v]
        for o_key in r_outputs.keys():
            if r_outputs[o_key] is not None:
                r_outputs[o_key] = torch.cat(r_outputs[o_key],dim=0)
        # ↓ 出力確認用
        """ print("{")
        for okey,output in r_outputs.items():
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

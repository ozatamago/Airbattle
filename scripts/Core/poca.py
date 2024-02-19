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
        self.act_dim = act_dim
        self.hidden_layer1 = nn.Linear(obs_dim, hid_dim * 2) # first hidden layer
        self.hidden_layer2 = nn.Linear(hid_dim * 2, hid_dim) # second hidden layer
        self.action_layer = nn.Linear(hid_dim, act_dim) # action output layer
        self.mish = nn.Mish()
        self.softmax = nn.Softmax(dim=-1) # softmax activation function for log_prob output

    def forward(self, obs: torch.Tensor, hidden: torch.Tensor = None) -> dict:
        if obs.size(0) == 0:
            print(f"obs size is {obs.shape}")
            logits = torch.zeros((1,0,self.act_dim), dtype=torch.float32)
        else:
            x = self.mish(self.hidden_layer1(obs)) # pass through the first hidden layer and apply relu
            x = F.relu(self.hidden_layer2(x)) # pass through the second hidden layer and apply relu
            logits = F.relu(self.action_layer(x)) + 1e-8 # log0 にならないように微小な値を足す 
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
        param: encoded_obs: エンコード後の観測テンソル (batch_size,max_agents,encoded_obs_dim)
        param: actives: Agent数のテンソル (batch_size,1)
        returns: value (batch_size,1), sliced_obs list(tensor (num_agents,encoded_obs_dim)) length == batch_size
        """
        # print(Printer.tensorPrint(actives))
        indices = [torch.arange(act.item()) for act in actives]
        # 頭に残りのAgentの数をつけておく
        x = torch.cat([torch.div(actives,self.max_agents).expand((encoded_obs.shape[0],encoded_obs.shape[1],1)),encoded_obs],dim=-1)
        sliced_x = [x[bi].index_select(0,indice).unsqueeze(0) for bi,indice in enumerate(indices)]
        # print(f"sliced_x : {sliced_x}")
        value = torch.cat([(self.v_layer(sub_x) if sub_x.size(1) > 0 else torch.tensor([[0]])) for sub_x in sliced_x],dim=0) # (batch_size,1)
        return value
class SparseAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, sparsity_target: float=0.05, sparsity_weight: float=0.2, lr=0.01):
        super(SparseAutoencoder, self).__init__()
        # 入力層と隠れ層のサイズを設定
        self.input_size = input_size
        self.hidden_size = hidden_size
        # スパース性の目標値と重みを設定
        self.sparsity_target = torch.tensor([sparsity_target])
        self.sparsity_weight = sparsity_weight
        # エンコーダとデコーダの層を定義
        self.encoder = nn.Linear(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, input_size)
        # 最適化アルゴリズムを設定
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, input, only_get_hidden: bool=True):
        # x = input.view(-1,self.input_size)
        x = input
        if only_get_hidden:
            return self.encoder(x)
        # エンコーダの出力をシグモイド関数で活性化
        h = torch.sigmoid(self.encoder(x))
        # デコーダの出力をシグモイド関数で活性化
        y = torch.sigmoid(self.decoder(h))
        # 隠れ層の平均活性化度を計算
        rho = torch.mean(h, dim=0)
        # スパース性の損失関数を計算
        sparsity_loss = self.sparsity_weight * torch.sum(torch.nn.functional.kl_div(self.sparsity_target, rho))
        # 再構築の損失関数を計算
        reconstruction_loss = F.binary_cross_entropy(y, x, reduction='sum') / x.shape[0]
        # 全体の損失関数を計算
        loss = reconstruction_loss + sparsity_loss
        return y, loss
    def updateEncoder(self, input):
        # x = input.view(-1,self.input_size)
        x = input
        _, loss = self(x,False)
        # 勾配をゼロにする
        self.optimizer.zero_grad()
        # 逆伝播で勾配を計算
        loss.backward()
        # パラメータを更新
        self.optimizer.step()

class StateEncoder(nn.Module):
    def __init__(self,obs_dim: int,act_dim: int, out_dim: int):
        super(StateEncoder,self).__init__()
        self.out_dim = out_dim
        self.obs_encode_layer = SparseAutoencoder(obs_dim,out_dim)
        self.obs_act_encode_layer = SparseAutoencoder(obs_dim+act_dim,out_dim)

    def forward(self,observations: torch.Tensor,actions: torch.Tensor):
        num_agents = observations.size(1)
        if num_agents == 0:
            encoded_obs = torch.zeros((observations.size(0),0,self.out_dim),dtype=torch.float32)
            encoded_obs_act = torch.zeros((observations.size(0),0,self.out_dim),dtype=torch.float32)
            encoded_counterfactual = torch.zeros((observations.size(0),0,self.out_dim),dtype=torch.float32)
        else:
            indices = [torch.tensor(i) for i in range(num_agents)]
            encoded_obs: torch.Tensor = self.obs_encode_layer(observations) # g (batch_size,num_agents,out_dim)
            encoded_obs_act: torch.Tensor = self.obs_act_encode_layer(torch.cat([observations,actions],dim=-1)) # f (batch_size, num_agents, out_dim)
            extracts = [(encoded_obs.index_select(1,indice),TensorExtension.extractSelect(1,encoded_obs_act,indice)) for indice in indices]
            encoded_counterfactual = torch.stack([(g if f is None else torch.cat([g,f],dim=1)) for g,f in extracts],dim=1) # (g,f) (batch_size, num_agents, out_dim)
        return encoded_obs, encoded_obs_act, encoded_counterfactual
    
    def updateEncoder(self,observations: torch.Tensor,actions: torch.Tensor):
        self.obs_encode_layer.updateEncoder(observations)
        self.obs_act_encode_layer.updateEncoder(torch.cat([observations,actions],dim=-1))

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
        self.v_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)
        self.q_optimizer = torch.optim.Adam(self.q_layer.parameters(), lr=self.lr)
    
    def forward(self, obs: torch.Tensor, hidden: torch.Tensor = None) -> dict:
        r_outputs = dict()
        obs, actives = self.split_to_obs_actives(obs)
        # print(Printer.tensorPrint(obs,False))
        for bi in range(obs.size(0)):
            outputs, batch_active, _, _, _ = self.predict_actions(bi,obs,actives,hidden)
            DictExtension.reduceNone(outputs,lambda o: TensorExtension.tensor_padding(o,self.max_agents,1))
            outputs['actives'] = torch.tensor([[batch_active]]) # (1,1) 切り出し用
            outputs['policy'] =outputs['policy'].view(-1,self.max_agents*self.act_dim)
            outputs['logits'] =outputs['logits'].view(-1,self.max_agents*self.act_dim)
            # ↓ 出力確認用
            # print(Printer.tensorDictPrint(outputs,False))
            DictExtension.StackItems(r_outputs,outputs)
        DictExtension.reduceNone(r_outputs,lambda o: torch.cat(o,dim=0))
        # ↓ 出力確認用
        # print(Printer.tensorDictPrint(r_outputs,False))
        return outputs
    
    def split_to_obs_actives(self, obs: torch.Tensor):
        obs = obs.view(-1,obs.size(-1))
        actives = obs[:,0]
        actives = actives.view(-1,1).to(torch.int32)
        obs = torch.stack(torch.split(obs[:,1:],self.obs_dim,dim=-1),dim=1).to(torch.float32)
        return obs, actives

    def predict_actions(self, batch_num,obs: torch.Tensor,actives :torch.Tensor, hidden: torch.Tensor = None):
        if isinstance(batch_num,list):
            return [(self.predict_actions(bi,obs,actives,hidden)) for bi in batch_num]
        batch_obs = obs.index_select(0,torch.tensor(batch_num)) # (1,max_agents,obs_dim)
        batch_hidden = hidden.index_select(0,torch.tensor(batch_num)) if hidden is not None else None
        batch_active = int(actives[batch_num].item())
        active_obs = batch_obs.index_select(1,torch.arange(batch_active))
        outputs = self.actor(active_obs.view(-1,self.obs_dim), batch_hidden)
        DictExtension.reduceNone(outputs,lambda o: (o.view(-1,batch_active,self.act_dim) if batch_active > 0 else o)) 
        encoded_obs, _, encoded_counterfactual = self.state_encoder(active_obs,outputs['policy']) # g (1,num_agents,hid_dim), f (1, num_agents, hid_dim), (1, num_agents, num_agents, hid_dim)
        # outputs['encoded_counterfactual'] = encoded_counterfactual
        # outputs['encoded_obs'] = encoded_obs
        return outputs, batch_active, batch_obs, encoded_counterfactual, encoded_obs
    
    def updateStateEncoder(self, obs: torch.Tensor, hidden: torch.Tensor = None):
        obs, actives = self.split_to_obs_actives(obs)
        for bi in range(obs.size(0)):
            outputs, batch_active, batch_obs, _, _ = self.predict_actions(bi,obs,actives,hidden)
            if batch_active > 0:
                self.state_encoder.updateEncoder(batch_obs,outputs['policy'])

    def updateNetworks(self,obs: torch.Tensor,rew: torch.Tensor,action_space: gym.spaces):
        torch.autograd.set_detect_anomaly(True)
        obs = obs.flatten(0,2)
        rew = rew.flatten(0,2)
        act_space = [s.n for s in action_space]
        act_split_index = [sum(act_space[:ai]) for ai in range(1,len(act_space))]
        rng = np.random.default_rng()
        times = list(range(obs.size(0)))
        train_ts = rng.permutation(times)
        obs, actives = self.split_to_obs_actives(obs)
        # print(Printer.anotateLine(Printer.tensorPrint(obs,False)+" / "+Printer.tensorPrint(actives,False)))
        predicteds = list(map(list, zip(*self.predict_actions(times,obs,actives))))
        for t in train_ts:
            if predicteds[1][t] > 0:
                observations_t = predicteds[2][t]
                actions_t = predicteds[0][t]['policy']
                print(Printer.tensorPrint(observations_t,False)+" / "+Printer.tensorPrint(actions_t,False))
                self.state_encoder.updateEncoder(predicteds[2][t],predicteds[0][t]['policy'])
                # print(Printer.anotateLine())
                yt_lam = util.yt_lambda(predicteds[4],actives,rew,self.critic,t)
                # print(Printer.anotateLine())
                v_criterion = nn.MSELoss()
                q_criterion = nn.MSELoss()
                self.v_optimizer.zero_grad()
                self.q_optimizer.zero_grad()
                v_value = self.critic(predicteds[4][t],actives[t])
                print(Printer.tensorPrint(v_value,False))
                q_value = self.q_layer(predicteds[3][t])
                print(Printer.tensorPrint(q_value,False))
                v_loss = v_criterion(v_value.view(-1,v_value.size(1)),yt_lam)
                q_loss = q_criterion(q_value.view(-1,q_value.size(1)),yt_lam)
                v_loss.backward()
                q_loss.backward()
                self.v_optimizer.step()
                self.q_optimizer.step()
                # print(Printer.anotateLine())
        for t in train_ts:
            if predicteds[1][t] > 0:
                yt_lam = util.yt_lambda(predicteds[4],actives,rew,self.critic,t)
                adv = -self.q_layer(predicteds[3][t]) + yt_lam
                action_logits: torch.Tensor = predicteds[0][t]['logits']
                print(Printer.tensorPrint(action_logits))
                action_prob = torch.cat([torch.softmax(al) for al in action_logits.view(-1).split(act_split_index,-1)],0).view(-1,actives[t].item(),self.act_dim)
                print(Printer.tensorPrint(action_prob))
                actors_loss = torch.mean(torch.mul(action_prob,adv.repeat(1,1,action_prob.size(-1))),-1)
                print(Printer.tensorPrint(actors_loss))
                for actor_loss in actors_loss[0]:
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward(retain_graph=True)
                    self.actor_optimizer.step()

    def init_hidden(self, hidden=None):
        # RNNを使用しない場合、ダミーの隠れ状態を返す
        return None
    
    def load_model(self,model: nn.Module,modelname:str,folderpath:str,strict:bool=True):
        model.load_state_dict(torch.load(folderpath+modelname),strict)

    def load_state_dict(self, folderpath:str,strict:bool=True):
        self.load_model(self.actor,getActorModelName(),folderpath,strict)
        self.load_model(self.critic,getCriticModelName(),folderpath,strict)
        self.load_model(self.state_encoder,getStateEncoderModelName(),folderpath,strict)
        self.load_model(self.q_layer,getQModelName(),folderpath,strict)

    def save_model(self,model: nn.Module,modelname:str,folderpath:str):
        torch.save(model.state_dict(),folderpath+modelname)

    def save_state_dict(self,folderpath:str):
        self.save_model(self.actor,getActorModelName(),folderpath)
        self.save_model(self.critic,getCriticModelName(),folderpath)
        self.save_model(self.state_encoder,getStateEncoderModelName(),folderpath)
        self.save_model(self.q_layer,getQModelName(),folderpath)

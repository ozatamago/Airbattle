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
from ..Helper.ListExtension import ListExtension
import warnings
from gymnasium import spaces
from ASRCAISim1.addons.HandyRLUtility.model import ModelBase

def getBatchSize(obs,space):
    if(isinstance(space,spaces.Dict)):
        k=next(iter(space))
        return getBatchSize(obs[k],space[k])
    elif(isinstance(space,spaces.Tuple)):
        return  getBatchSize(obs[0],space[0])
    else:
        return obs.shape[0]

class RSABlock(nn.Module):
    """
    マルチヘッドアテンションと線形層を用いて、状態エンコーディングを行うモデル
    """
    def __init__(self, input_size: int, output_size: int, num_heads: int = 4):
        """MA-POCA RSA ブロックの初期化

        Args:
            input_size: 入力エンコーディングのサイズ
            output_size: 出力エンコーディングのサイズ
            num_heads: マルチヘッドアテンションのヘッド数（デフォルト: 4）
        """
        super(RSABlock, self).__init__()
        self.input_layer = nn.Linear(input_size, output_size)
        self.output_layer = nn.Linear(output_size, output_size)
        self.attention = nn.MultiheadAttention(output_size, num_heads)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """状態エンコーディングの計算

        Args:
            inputs: 入力エンコーディング（エージェント状態ベクトル）

        Returns:
            更新された状態エンコーディング
        """
        x = F.relu(self.input_layer(inputs))  # (batch_size, num_agents, output_size)
        attention_output, _ = self.attention(x, x, x)  # (batch_size, num_agents, output_size)
        context = torch.add(attention_output, x)  # (batch_size, num_agents, output_size)
        return F.relu(self.output_layer(context)).clone()  # (batch_size, num_agents, output_size)

class V(nn.Module):
    """
    反事実状態エンコーディングに基づいて状態価値(V値)を計算するモデル
    """
    def __init__(self,input_dim: int, v_hid_dim: int = 64, rsa_heads: int = 4,lr: float=0.001,lam = 0.5,gam = 0.5,lam_eps = 1e-8,gam_eps = 1e-8):
        """MA-POCA V 値計算モデルの初期化

        Args:
            input_dim: 入力次元のサイズ
            v_hid_dim: V 値計算用隠れ層のサイズ（デフォルト: 64）
            rsa_heads: RSA モジュール内のヘッド数（デフォルト: 4）
            lr: 学習率（デフォルト: 0.001）
            lam: 割引報酬に関する減衰率lambda（デフォルト: 0.5）
            gam: 割引報酬に関する減衰率gamma（デフォルト: 0.5）
            lam_eps: lambda 計算の打ち切り閾値（デフォルト: 1e-8）
            gam_eps: gamma 計算の打ち切り閾値（デフォルト: 1e-8）
        """
        super(V,self).__init__()
        self.lam = lam
        self.lam_eps = lam_eps
        self.gam = gam
        self.gam_eps = gam_eps
        mid_dim = int((input_dim+v_hid_dim)/(2*rsa_heads))*rsa_heads
        self.value_net = nn.Sequential(
            RSABlock(input_dim,mid_dim,rsa_heads), # V値計算用RSA
            nn.Linear(mid_dim,v_hid_dim),
            nn.LeakyReLU(),
            nn.Linear(v_hid_dim,1) # V値計算
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.loss_func = nn.MSELoss()

    def forward(self,inputs):
        """V 値のフォワード計算 

        Args:
            inputs: 状態入力。リスト、タプル、またはテンソル型 

        Returns:
            状態価値（スカラー値）を表現するテンソル
        """ 
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
        """割引価値の代理 yt_lambda の計算

        Args:
            encoded_obs_seq: エンコードされた状態シーケンス
            rew_seq: 報酬シーケンス
            t: 現在のタイムステップ

        Returns:
            yt_lambda 値（スカラー値を表現するテンソル）
        """
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
        """yt_lambda 計算における補助項 Gt_lambda の計算

        Args:
            encoded_obs_seq: エンコードされた状態シーケンス
            rew_seq: 報酬シーケンス
            t: 現在のタイムステップ
            n: 計算に用いる未来のステップ数

        Returns:
            Gt_lambda 値（スカラー値を表現するテンソル）
        """
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
        """MA-POCA における V 値モデルの更新

        Args:
            encoded_obs: エンコードされた状態シーケンス
            actives_list: 各タイムステップにおけるエージェントのアクティブ度
            reward: 報酬シーケンス
            ts: 学習に用いるタイムステップのリスト
        """
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
    """
    反事実状態エンコーディングと行動に基づいて行動価値(Q値)を計算するモデル。
    """
    def __init__(self,input_dim: int, q_hid_dim: int = 64, rsa_heads: int = 4, lr: float=0.001):
        """MA-POCA Q ネットワークの初期化

        Args:
            input_dim: 入力エンコーディングのサイズ
            q_hid_dim: Q 値計算用隠れ層のサイズ（デフォルト: 64）
            rsa_heads: マルチヘッドアテンションのヘッド数（デフォルト: 4）
            lr: 学習率（デフォルト: 0.001）
        """
        super(Q,self).__init__()
        mid_dim = int((input_dim+q_hid_dim)/(2*rsa_heads))*rsa_heads
        self.value_net = nn.Sequential(
            RSABlock(input_dim,mid_dim,rsa_heads), # Q値計算用RSA
            nn.Linear(mid_dim,q_hid_dim),
            nn.LeakyReLU(),
            nn.Linear(q_hid_dim,1) # Q値計算
        )
        self.loss_func = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self,encoded_counterfactual: torch.Tensor):
        """Q値のフォワード計算

        Args:
            encoded_counterfactual: 状態エンコーディングに基づいた反事実状態

        Returns:
            行動価値 Q(s, a) を表現するテンソル
        """
        q_value = self.value_net(encoded_counterfactual.view(-1,encoded_counterfactual.shape[-2],encoded_counterfactual.shape[-1])) # (batch_size*num_agents,num_agents,q_hid_dim)
        q_value = q_value.view(-1,encoded_counterfactual.shape[1],encoded_counterfactual.shape[2],q_value.shape[-1]) # Qψ (batch_size,num_agents,num_agents,1)
        q_value = torch.sum(q_value, dim=2) # (batch_size,num_agents, 1)
        return q_value
    
    def update(self,encoded_counterfactual: torch.Tensor,yt_lambda: torch.Tensor):
        """Q ネットワークの更新

        Args:
            encoded_counterfactual: 反事実状態
            yt_lambda: yt_lambda 値
        """
        self.optimizer.zero_grad()
        pre_q_value = self.forward(encoded_counterfactual)
        q_value = pre_q_value.view(-1,pre_q_value.size(1))
        q_loss = self.loss_func(q_value,yt_lambda.expand(q_value.size()))
        q_loss.backward(retain_graph=True)
        self.optimizer.step()

class SparseAutoencoder(nn.Module):
    """
    スパース性制約付き自己符号化器モデル。入力データを低次元空間にエンコードし、元の高次元空間に再構成する。
    """
    def __init__(self, input_dim, hidden_dim, rho=0.05,beta=3 ,lr=0.001):
        """スパース自己符号化器の初期化

        Args:
            input_dim: 入力データの次元
            hidden_dim: 隠れ層の次元
            rho: スパース性制約係数 (デフォルト: 0.05)
            beta: スパース性制約項の重み係数 (デフォルト: 3)
            lr: 学習率 (デフォルト: 0.001)
        """
        super(SparseAutoencoder, self).__init__()
        middle_dim = int((input_dim + hidden_dim)/2)
        # エンコーダとデコーダの層を定義
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, middle_dim, bias=False),
            nn.Sigmoid(),
            nn.Dropout(),
            nn.Linear(middle_dim, hidden_dim, bias=False),
            nn.Sigmoid(),
            nn.Dropout()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim, bias=False)
        )
        # 最適化アルゴリズムを設定
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.rho = torch.tensor([rho])
        self.beta = beta
    def forward(self, input, get_hidden: bool=True):
        """入力データのエンコードと再構成

        Args:
            input: 入力データ
            get_hidden: 隠れ層の出力を返すかどうか (デフォルト: True)

        Returns:
            再構成されたデータ
            get_hidden が True の場合は、隠れ層の出力を返す
        """
        h = self.encoder(input)
        if get_hidden:
            return h
        y = self.decoder(h)
        return y
    def updateEncoder(self, batch):
        """エンコーダの更新

        Args:
            batch: ミニバッチデータ
        """
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
    """
    状態と行動を入力とし、反事実状態 (counterfactual state) を生成するエンコーダモデル
    """
    def __init__(self,obs_dim: int,act_dim: int, out_dim: int):
        """状態エンコーダの初期化

        Args:
            obs_dim: 状態ベクトルの次元数
            act_dim: 行動ベクトルの次元数
            out_dim: エンコード後の次元数
        """
        super(StateEncoder,self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.out_dim = out_dim
        self.obs_encode_layer = SparseAutoencoder(obs_dim,out_dim)
        self.obs_act_encode_layer = SparseAutoencoder(obs_dim+act_dim,out_dim)

    def encode_obs(self,observations):
        """状態ベクトルのエンコード

        Args:
            observations: 状態ベクトル。リスト、タプル、または torch.Tensor

        Returns:
            エンコードされた状態ベクトル (torch.Tensor)
        """
        if isinstance(observations,(list,tuple)):
            return type(observations)([self.encode_obs(obs) for obs in observations])
        elif isinstance(observations,torch.Tensor):
            obs = observations.view(-1,self.obs_dim) # (batch_size*num_agents, obs_dim)
            return self.obs_encode_layer(obs)
        else:
            raise ValueError(f"Encode Observation Failed! UnknownType: observations:{type(observations)}")
    
    def encode_obs_acts(self,observations,actions):
        """状態・行動ペアのエンコード

        Args:
            observations: 状態ベクトル。リスト、タプル、または torch.Tensor
            actions: 行動ベクトル。リスト、タプル、または torch.Tensor

        Returns:
            エンコードされた状態・行動ペア (torch.Tensor)
        """ 
        if isinstance(observations,(list,tuple)) or isinstance(actions,(list,tuple)):
            return type(observations)([self.encode_obs_acts(observations[i],actions[i]) for i in len(observations)])
        elif isinstance(observations,torch.Tensor) and isinstance(actions,torch.Tensor):
            obs = observations.view(-1,self.obs_dim)
            acts = actions.view(-1,self.act_dim)
            return self.obs_act_encode_layer(torch.cat([obs,acts],dim=-1)) # f (batch_size*num_agents, out_dim)
        else:
            raise ValueError(f"Encode Observation Action Pair Failed! UnknownType: observations:{type(observations)} and actions:{type(actions)}")
    
    def encode_counterfactual(self,encoded_observations,encoded_actions):
        """反事実状態の生成

        Args:
            encoded_observations: エンコードされた状態ベクトル
            encoded_actions: エンコードされた行動ベクトル

        Returns:
            反事実状態を表現するテンソル
        """
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
        """状態・行動から反事実状態を生成するショートカットメソッド"""
        return self.encode_counterfactual(self.encode_obs(observations),self.encode_obs_acts(observations,actions))
    
    def updateObsEncoder(self,observations):
        """状態エンコーダ (obs_encode_layer) の更新

        Args:
            observations: 状態ベクトル
        """
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
        """状態・行動エンコーダ (obs_act_encode_layer) の更新

        Args:
            observations: 状態ベクトル
            actions: 行動ベクトル
        """
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
    """
    強化学習におけるエージェントの行動選択を行うためのモデル。 反事実状態エンコーディングを元に、行動確率を出力する
    """
    def __init__(self, encoded_obs_dim: int, act_dim: int, num_heads: int=3, lr: float=0.001):
        """行動選択モデル（Actorモデル）の初期化

        Args:
            encoded_obs_dim: 反事実状態エンコーディングの次元数
            act_dim: 行動空間の次元数
            num_heads: マルチヘッドアテンションのヘッド数 (デフォルト: 3)
            lr: 学習率 (デフォルト: 0.001)
        """   
        super(Actor, self).__init__()
        self.act_dim = act_dim
        self.obs_dim = encoded_obs_dim
        mid_dim = int((encoded_obs_dim+act_dim)/(2*num_heads))*num_heads
        self.policy_net = nn.Sequential(
            RSABlock(encoded_obs_dim,mid_dim,num_heads),
            nn.Linear(mid_dim,mid_dim),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(mid_dim,act_dim),
            nn.ReLU()
        )
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr)

    def forward(self, encoded_observations_list: list, actives_list: list, hidden: torch.Tensor = None) -> list:
        """行動選択

        Args:
            encoded_observations_list: エンコードされた状態ベクトルのリスト
            actives_list: 各エージェントのアクティブ数のリスト 
            hidden: 隠れ層の状態（省略可）

        Returns:
            各エージェントの行動選択に関する情報を格納した辞書リスト
            * policy: 行動確率のテンソル (各エージェントの行動次元に一致)
            * logits: 行動のロジット (policy を生成する前のネットワーク出力)
            * hidden: 隠れ層の状態
        """
        rets = list()
        for obs,active in zip(encoded_observations_list, actives_list):
            ret = dict()
            if active == 0:
                logits = torch.zeros((1,0,self.act_dim), dtype=torch.float32)
            else:
                logits = self.policy_net(obs) + 1e-6
            ret = {'policy':logits,'logits': logits, 'hidden': hidden}
            DictExtension.reduceNone(ret)
            rets.append(ret)
        return rets
    
    def update(self,observations_list: list, actives_list: list,reward: torch.Tensor,action_space: gym.spaces,v_model: V,q_model: Q,state_encoder: StateEncoder,ts: list):
        """Actor モデルの更新

        Args:
            observations_list: 状態観測のリスト
            actives_list: 各エージェントのアクティブ数のリスト
            reward: 報酬
            action_space: 行動空間
            v_model: V値モデル
            q_model: Q値モデル
            state_encoder: 状態エンコーダ
            ts: 学習に用いるタイムステップのリスト
        """
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

class MAPOCA(nn.Module):
    """
    Multi-Agent POlicy with Counterfactual Action (MAPOCA) アルゴリズムに基づく強化学習モデル。
    複数のエージェントが協調する強化学習環境を想定し、状態エンコーダ、Actor、V値モデル、Q値モデルから構成される。
    """
    def __init__(self, obs_dim: int, act_dim: int, max_agents: int, lr: float, hid_dim: int = 128, v_hid_dim: int = 64, v_rsa_heads: int = 4,q_hid_dim: int = 64, q_rsa_heads: int = 4):
        """MAPOCAモデルの初期化

        Args:
            obs_dim: 各エージェントの状態観測ベクトルの次元数 
            act_dim: 各エージェントの行動空間の次元数
            max_agents: 環境内の最大エージェント数
            lr: 学習率
            hid_dim: エンコーディング／内部状態表現の次元数 (デフォルト: 128)
            v_hid_dim: V値モデルの隠れ層次元数 (デフォルト: 64)
            v_rsa_heads: V値モデルのRSAモジュール内ヘッド数 (デフォルト: 4)
            q_hid_dim: Q値モデルの隠れ層次元数 (デフォルト: 64)
            q_rsa_heads: Q値モデルのRSAモジュール内ヘッド数 (デフォルト: 4)
        """
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
        """エージェントの行動選択

        Args:
            obs: 状態観測のテンソル
            hidden: 隠れ層状態（省略可）
            get_only_one_action_output: 単一の辞書形式で結果を返すかどうか (デフォルト: True)

        Returns:
            actor_output: 各エージェントの行動情報を含む辞書(複数エージェントの結果) または、
                get_only_one_action_output = True の場合は、1バッチ分の行動情報を含む辞書
                * policy: 行動確率のテンソル ((batch_size * max_agents) * act_dim)
                * logits: 行動のロジット (policy を生成する前のネットワーク出力)
                * hidden: 隠れ層状態 
        """
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
        """状態観測テンソルから状態リストとアクティブ数リストを生成

        Args:
            obs: 状態観測のテンソル

        Returns:
            observations_list: 状態観測リスト（各エージェントの状態観測がテンソルで格納されたリスト）
            actives_list: 各エージェントのアクティブ数のリスト
        """
        batch_size = obs.size(0)
        # obs == (batch_size,2+obs_dim*max_agents)
        times: torch.Tensor = obs.index_select(-1,torch.tensor([0])) # (batch_size,1)
        _, times_idx = torch.sort(times, dim=0)
        sorted_obs = torch.gather(obs, 0, times_idx.expand(obs.size()))
        actives: torch.Tensor = sorted_obs.index_select(-1,torch.tensor([1])) # (batch_size,1)
        actives_list = [active.item() for active in actives.to(torch.int32)]
        observations = torch.split(sorted_obs[:,2:],1,dim=0) # tuple((1,obs_dim*max_agents),...,(1,obs_dim*max_agents)) length == batch_size
        observations_list = [TensorExtension.tensor_padding(torch.stack(observations[batch_index].split(self.obs_dim,-1)[:actives_list[batch_index]],dim=1),1,-1,False,True,actives_list[batch_index]/self.max_agents) if actives_list[batch_index] > 0 else torch.zeros((0,1+self.obs_dim)) for batch_index in range(batch_size)]
        return observations_list, actives_list # list((num_agents,1 + obs_dim),...,(num_agents,1 +obs_dim), list(num_agents,...,num_agents)

    def updateNetworks(self,obs: torch.Tensor,rew: torch.Tensor,action_space: gym.spaces):
        """各モデルの更新

        Args:
            obs: 状態観測のテンソル
            rew: 報酬のテンソル
            action_space: 行動空間
        """   
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
        """隠れ状態の初期化 (ダミー実装) """
        return None
    
    def load_model(self,model: nn.Module,modelname:str,folderpath:str,strict:bool=True):
        """モデルのパラメータ読み込み """
        model.load_state_dict(torch.load(folderpath+modelname+".pth"),strict)

    def load_state_dict(self, folderpath:str,strict:bool=True):
        """各コンポーネントモデルのパラメータ読み込み """
        self.load_model(self.actor,getActorModelName(),folderpath,strict)
        self.load_model(self.v_model,getVModelName(),folderpath,strict)
        self.load_model(self.state_encoder,getStateEncoderModelName(),folderpath,strict)
        self.load_model(self.q_model,getQModelName(),folderpath,strict)

    def save_model(self,model: nn.Module,modelname:str,folderpath:str):
        """モデルのパラメータ保存 """
        torch.save(model.state_dict(),folderpath+modelname+".pth")

    def save_state_dict(self,folderpath:str):
        """各コンポーネントモデルのパラメータ保存 """
        self.save_model(self.actor,getActorModelName(),folderpath)
        self.save_model(self.v_model,getVModelName(),folderpath)
        self.save_model(self.state_encoder,getStateEncoderModelName(),folderpath)
        self.save_model(self.q_model,getQModelName(),folderpath)

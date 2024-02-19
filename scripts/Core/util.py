import torch
import torch.nn as nn
import numpy as np

def Advt(encoded_counterfactual: torch.Tensor,q_model: nn.Module,t: int,yt_lam: float):
   return -q_model(encoded_counterfactual[t]) + yt_lam

def yt_lambda(encoded_obs_seq: list,actives: torch.Tensor,rew_seq: torch.Tensor,v_model: nn.Module,t: int,lam = 0.5,gam = 0.5,lam_eps = 1e-8,gam_eps = 1e-8):
    if lam == 1.0:
        return .0
    total = torch.zeros((1))
    max_n = len(encoded_obs_seq) - t
    p_lam = 1.0
    for n in range(1,max_n):
        if p_lam >= lam_eps:
          total = total + p_lam*Gt_lambda(encoded_obs_seq,actives,rew_seq,v_model,t,n,gam,gam_eps)
          p_lam *= lam
        else:
          p_lam = .0
          break
    return total*(1.0 - lam)

def Gt_lambda(encoded_obs_seq: list,actives: torch.Tensor,rew_seq: torch.Tensor,v_model: nn.Module,t: int,n,gam = 0.5,gam_eps = 1e-8):
    total = torch.zeros((1))
    p_gam = 1.0
    for l in range(1,n):
        if p_gam >= gam_eps:
          total = total + p_gam*rew_seq[t+l]
          p_gam *= gam
        else:
          p_gam = .0
          break
    return  ((v_model(encoded_obs_seq[t+n],actives[t+n])*p_gam) if p_gam >= gam_eps else torch.zeros((1))) + total

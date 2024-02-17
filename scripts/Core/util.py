import torch

def calcLosses_t(obs_seq,act_seq,rew_seq,g_model,f_model,v_model,q_model,t,lam = 0.5,gam = 0.5,lam_eps = 1e-8,gam_eps = 1e-8):
   g_out = g_model(obs_seq)
   f_out = f_model(act_seq)
   yt_lam = yt_lambda(g_out,rew_seq,v_model,t,lam,gam,lam_eps,gam_eps)
   j_phai_t = Jt_phai(g_out,v_model,t,yt_lam)

def Jt_phai(obs_seq,v_model,t,yt_lambda):
   return (v_model(obs_seq[t])-yt_lambda)**2

def Advj(g_out,f_out,q_model,j,yt_lambda):
   return yt_lambda - q_model(torch.cat([g_out[j],torch.stack([f_out[i] for i in range(len(f_out)) if i != j],dim=0)],dim=0))

def Jt_thetai_grad(g_out,f_out,q_model,i,t,yt_lambda):
   q_out = q_model(torch.cat([g_out,torch.stack([f_out[j] for j in range(len(f_out)) if j != i],dim=0)],dim=0))


def yt_lambda(obs_seq,rew_seq,v_model,t,lam = 0.5,gam = 0.5,lam_eps = 1e-8,gam_eps = 1e-8):
    if lam == 1.0:
        return .0
    total = .0
    max_n = len(obs_seq) - t
    p_lam = 1.0
    for n in range(1,max_n):
        if p_lam >= lam_eps:
          total += p_lam*Gt_lambda(obs_seq,rew_seq,v_model,t,n,gam,gam_eps)
          p_lam *= lam
        else:
          p_lam = .0
          break
    return (1.0 - lam)*total

def Gt_lambda(obs_seq,rew_seq,v_model,t,n,gam = 0.5,gam_eps = 1e-8):
    total = .0
    p_gam = 1.0
    for l in range(1,n):
        if p_gam >= gam_eps:
          total += p_gam*rew_seq[t+l]
          p_gam *= gam
        else:
          p_gam = .0
          break
    return total + ((p_gam*v_model(obs_seq[t+n])) if p_gam >= gam_eps else .0)

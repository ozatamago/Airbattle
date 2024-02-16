import torch

def forward_view_td_lambda(lamda, gamma, states, rewards, v_network_model):
  lambda_returns = torch.zeros_like(rewards)
  eligibility_trace = torch.zeros(state_dim)
  for t in reversed(range(seq_len)):
    state = states[t]
    reward = rewards[t]
    value = v_network_model(state)
    if t == seq_len - 1:
      target = value
    else:
      next_state = states[t+1]
      next_value = v_network_model(next_state)
      target = reward + gamma * next_value
    eligibility_trace = lamda * gamma * eligibility_trace
    eligibility_trace.scatter_(0, state, 1)
    lambda_returns[t] = target * eligibility_trace
  return lambda_returns
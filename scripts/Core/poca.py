import torch
import torch.nn as nn
import torch.nn.functional as F
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
    

# Actor class(Policy)
class Actor(ModelBase):
    def __init__(self, obs_space, ac_space, action_dist_class, model_config):
        obs_dim = obs_space.shape[0]
        act_dim = len(ac_space.nvec)
        print(ac_space)


        # print("obs_dim: ", obs_dim)
        # print("act_dim: ", act_dim)
        # super(Actor, self).__init__()
        super().__init__(obs_space, ac_space, action_dist_class, model_config)
        self.action_dim = self.action_dist_class.get_param_dim(ac_space) #=出力ノード数
        self.observation_space = obs_space

        # Define the network layers
        self.fc1 = nn.Linear(obs_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, self.action_dim)

        # Save the action distribution class
        self.action_dist_class = action_dist_class
    
    def forward(self, obs, hidden=None):
        # Take a vector observation as input and return a 5-dim action
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        action_logits = self.fc3(x)
        # action_probs = F.softmax(action_logits, dim=-1)

        # B=getBatchSize(obs,self.observation_space)
        # print("B: ", B)
        # print("self.action_dim: ", self.action_dim)
        # p = torch.ones([B,self.action_dim],dtype=torch.float32)

        ret = {"policy": action_logits}
        print(ret)
        return ret
    
    def init_hidden(self):
        # RNNを使用しない場合、ダミーの隠れ状態を返す
        return None
    

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
        # Define the optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr['model']['hyperParameters']['actor']['learningRate'])
        # Define the learning rate
        self.lr = lr['model']['hyperParameters']['actor']['learningRate']

    
    def forward(self, obs, active):
        # Take a vector observation as input and return the action probability and value
        action_prob = self.actor(obs)
        value = self.critic(obs,action_prob,active)
        return action_prob, value 
    
    # Training function
    def train_model(model, experiences, gamma, lam):
        # model: the actor-critic model
        # experiences: a list of dictionaries containing state, action, reward, n_state, done for each step
        # gamma: the discount factor
        # lam: the lambda parameter for TD lambda
        
        # Initialize the eligibility trace
        z = torch.zeros(1)
        # Initialize the TD error
        delta = torch.zeros(1)
        # Initialize the team reward
        R = torch.zeros(1)
        # Initialize the loss
        loss = torch.zeros(1)
        
        # Loop over the experiences in reverse order
        for i in reversed(range(len(experiences))):
            # Get the state, action, reward, next state, and done flag from the experience
            state = experiences[i]["state"]
            action = experiences[i]["action"]
            reward = experiences[i]["reward"]
            n_state = experiences[i]["n_state"]
            done = experiences[i]["done"]
            
            # Add the reward to the team reward
            R += reward
            
            # Get the action probability and value from the model
            action_prob, value = model(state)
            
            # Get the next value from the model
            _, n_value = model(n_state)
            
            # Calculate the TD error
            delta = reward + gamma * n_value * (1 - done) - value
            
            # Update the eligibility trace
            z = gamma * lam * z + action_prob
            
            # Update the loss
            loss += -delta * z
            
        # Optimize the model
        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()
        
        # Return the team reward
        return R
    
    def init_hidden(self):
        # RNNを使用しない場合、ダミーの隠れ状態を返す
        return None
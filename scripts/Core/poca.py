import torch
import torch.nn as nn
import torch.nn.functional as F

# Actor class
class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(Actor, self).__init__()
        # Define the network layers
        self.fc1 = nn.Linear(obs_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, act_dim)
    
    def forward(self, obs):
        # Take a vector observation as input and return a 5-dim action
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x

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
        super(ActorCritic, self).__init__()
        # Define the actor network
        self.actor = Actor(obs_dim, act_dim)
        # Define the critic network
        self.critic = Critic(obs_dim, act_dim, num_agents)
        # Define the optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        # Define the learning rate
        self.lr = lr
    
    def forward(self, obs):
        # Take a vector observation as input and return the action probability and value
        action_prob = self.actor(obs)
        value = self.critic(obs)
        return action_prob, value
    
    # Training function
    def train(model, experiences, gamma, lam):
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
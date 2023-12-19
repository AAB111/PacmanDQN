
import math
import random
import copy
import gym
import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

num_envs = 8
env_name = "CartPole-v1"
env = gym.make(env_name,render_mode = "rgb_array") # a single env

class ActorCritic(nn.Module):  
    def __init__(self,eps=0.2):
        super(ActorCritic, self).__init__()
        self.eps = eps
        self.critic = nn.Sequential(
            nn.Sequential(nn.Linear(in_features=4 + 1, out_features=8),nn.ReLU()),
            nn.Sequential(nn.Linear(in_features=8, out_features=1)),
        )
        nn.init.kaiming_normal_(self.critic[0][0].weight,mode='fan_in',nonlinearity='relu')
        nn.init.xavier_uniform_(self.critic[1][0].weight)
        
        self.actor = nn.Sequential(
            nn.Sequential(nn.Linear(in_features=4, out_features=8),nn.ReLU()),
            nn.Sequential(nn.Linear(in_features=8, out_features=2),nn.Softmax(dim=-1)),
        )
        nn.init.kaiming_normal_(self.actor[0][0].weight,mode='fan_in',nonlinearity='relu')
        nn.init.xavier_uniform_(self.actor[1][0].weight)
        
    def forward(self, states):
        states = torch.from_numpy(states).float().to(device)
        actions,log_probs,entropy = self.act(states)
        values = self.q_value(states,actions)
        return actions, log_probs, values, entropy
    
    def only_act(self,states):
        states = torch.from_numpy(states).float().to(device)
        actions,log_probs,entropy = self.act(states)
        return actions,log_probs,entropy
    
    def act(self,states):
        probs = self.actor(states).to(device)
        dist = Categorical(probs)
        action = dist.sample()
        return action, dist.log_prob(action),dist.entropy().mean()

    def q_value(self, states,actions):
        actions = actions.unsqueeze(0)
        out = torch.cat([states, actions], dim=-1)
        q_value = self.critic(out).to(device)
        return q_value

def computeReturns(rewards,gamma=0.9):
    n_steps = len(rewards) 
    disc_rewards = deque(maxlen=n_steps) 
    for t in range(n_steps)[::-1]:
        disc_return_t = (disc_rewards[0] if len(disc_rewards) > 0 else torch.tensor([0])).to(device)
        disc_rewards.appendleft( gamma*disc_return_t + rewards[t])
    return list(disc_rewards)

num_inputs  = env.observation_space.shape[0]
num_outputs = env.action_space.n

hidden_size = 16
lr          = 1e-4
eps_clip = 0.2

model = ActorCritic().to(device)
optimizer = optim.Adam(model.parameters(),lr=lr)

max_frames   = 200000
frame_idx    = 0

old_model = ActorCritic().to(device)
old_model.load_state_dict(model.state_dict())
rewards_episods = []
while frame_idx < max_frames:
    state = env.reset()[0]
    done = False
    values    = []
    rewards   = []
    ratio     = []
    entropy = 0
    reward_episod = 0
    while not done:
        action, log_probs, value, entrop = model(state)
        action2, log_probs2, entrop = old_model.only_act(state)
        next_state, reward, done, _,_ = env.step(action.cpu().numpy())
        reward_episod += reward
        entropy += entrop #not []
        values.append(value) #[]
        rewards.append(torch.tensor([reward]).to(device))#[]
        ratio.append(torch.exp(log_probs-log_probs2.detach()).unsqueeze(0)) #[]
        state = next_state
    frame_idx += 1
    returns = computeReturns(rewards)
    returns   = torch.cat(returns).detach()
    rewards_episods.append(reward_episod)
    if frame_idx % 50 == 0:
        print(np.array(rewards_episods).mean())
    values    = torch.cat(values).to(device)
    ratio = torch.cat(ratio).to(device)
    eps = np.finfo(np.float32).eps.item()
    returns = (returns - returns.mean() / (returns.std() + eps)).detach()
    advantage = returns - values
    unclip = ratio * -advantage
    clip = torch.clamp(ratio,1 - 0.2,1 + 0.2) * -advantage
    actor_loss =torch.max(unclip,clip).mean()
    critic_loss = advantage.pow(2).mean()
    old_model = copy.deepcopy(model)
    optimizer.zero_grad()
    loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy.detach()
    loss.backward()
    optimizer.step()
    if frame_idx % 100 == 0:
        print(model.actor[0][0].weight)

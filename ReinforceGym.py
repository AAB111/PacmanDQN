import numpy as np
import torch
import copy
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque
import gym
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.actor = nn.Sequential(
            nn.Sequential(nn.Linear(in_features=4, out_features=16),nn.ReLU()),
            nn.Sequential(nn.Linear(in_features=16, out_features=2),nn.Softmax(dim=-1))
        )
        nn.init.kaiming_normal_(self.actor[0][0].weight,mode='fan_in',nonlinearity='relu')
        nn.init.xavier_uniform_(self.actor[1][0].weight)
    def forward(self, states):
        states = torch.from_numpy(states).float().unsqueeze(0).to(device)
        probs = self.actor(states).to(device)
        dist = Categorical(probs)
        actions = dist.sample()
        return actions, dist.log_prob(actions)

class Reinforce:
    def __init__(self,lr = 1e-4,gamma=0.9,render_mode='rgb_array',eps = 0.2):
        self.eps = eps
        self.gamma=gamma
        self.policy = Policy().to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr,weight_decay=0.001)
        print(sum([p.numel() for p in self.policy.parameters() if p.requires_grad]))
        self.backup_model = copy.deepcopy(self.policy)
        self.env = gym.make('CartPole-v1',render_mode='human')

    def loss(self,ratio,returns):
        # unclip = ratio * -returns
        # clip = torch.clamp(ratio,1 - self.eps,1 + self.eps)* -returns
        actor_loss = (ratio * -returns).mean() 
        return actor_loss
    
    def train(self,loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def computeReturns(self,rewards,masks):
        n_steps = len(rewards) 
        disc_rewards = deque(maxlen=n_steps) 
        for t in range(n_steps)[::-1]:
            disc_return_t = (disc_rewards[0] if len(disc_rewards) > 0 else torch.tensor([0])).to(device)
            disc_rewards.appendleft( self.gamma*disc_return_t*masks[t] + rewards[t])
        return list(disc_rewards)
    
        
    def reinforce(self, n_training_episodes,n_steps):
        try:
            eps = np.finfo(np.float32).eps.item()
            ratio = []
            log_probs = []
            masks = []
            rewards   = []
            for i_episode in range(1, n_training_episodes+1):
                state, _ = self.env.reset()
                terminated = False
                step = 0
                rewardsToPrint = 0
                while not terminated:
                    action,log_prob = self.policy.forward(state) 
                    # action2,log_probOld = self.backup_model.forward(state) 
                    next_state, reward, terminated, truncated , info = self.env.step(action.item())
                    rewardsToPrint += reward
                    log_prob.retain_grad()
                    # ratio.append(torch.exp(log_prob-log_probOld.detach()).unsqueeze(0)) 
                    log_probs.append(log_prob)
                    rewards.append(torch.tensor([reward]).to(device))
                    masks.append(torch.tensor(1-terminated).to(device))
                    state = next_state
                    step += 1 
                returns = torch.cat(self.computeReturns(rewards,masks))
                returns = ((returns - returns.mean()) / (returns.std() + eps)).detach()
                print(f'награда = {int(rewardsToPrint)}')
                ratios    = torch.cat(log_probs)
                self.backup_model = copy.deepcopy(self.policy)
                ratios.retain_grad()
                loss = self.loss(ratios,returns)
                self.train(loss)
                del log_probs[:]
                del rewards[:]
                del masks[:]
                if i_episode % 100 ==0:
                    print(list(self.policy.parameters())[0])
                    print(self.policy.actor[0][0].weight)
        except Exception as e:
            print(e)
    
if __name__ == "__main__":
    pacman_hyperparameters = {
        "n_training_episodes": 100000,
        "n_evaluation_episodes": 10,
        'n_steps': 200,
        "gamma": 0.9,
        "lr": 1e-4,
        'render_mode':'human'
    }
    reinf = Reinforce(pacman_hyperparameters['lr'],pacman_hyperparameters['gamma'],pacman_hyperparameters['render_mode'])
    scores = reinf.reinforce(pacman_hyperparameters["n_training_episodes"],pacman_hyperparameters['n_steps'])  
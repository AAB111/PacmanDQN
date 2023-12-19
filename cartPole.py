from collections import deque
import numpy as np
import torch
import gym
import copy
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

class ActorCriticModel(nn.Module):  
    def __init__(self,eps):
        super(ActorCriticModel, self).__init__()
        self.eps = eps
        self.critic = nn.Sequential(
            nn.Sequential(nn.Linear(in_features=2 + 1, out_features=16),nn.ReLU()),
            nn.Sequential(nn.Linear(in_features=8, out_features=1)),
        )
        nn.init.kaiming_normal_(self.critic[0][0].weight,mode='fan_out',nonlinearity='relu'),
        nn.init.xavier_uniform_(self.critic[0][0].weight)
        
        self.actor = nn.Sequential(
            nn.Sequential(nn.Linear(in_features=2, out_features=8),nn.ReLU()),
            nn.Sequential(nn.Linear(in_features=16, out_features=3),nn.Softmax(dim=-1)),
        )
        nn.init.kaiming_normal_(self.actor[0][0].weight,mode='fan_out',nonlinearity='relu'),
        nn.init.xavier_uniform_(self.actor[0][0].weight)
    
    def loss(self, returns, values,entropy,ratio):
        advantage = returns - values # leaf = F req
        unclip = ratio * -advantage
        clip = torch.clamp(ratio,1 - 0.2,1 + 0.2) * -advantage
        actor_loss = torch.max(unclip,clip).mean(dim=0)
        # actor_loss = (log_probs * -advantage).mean(dim=1).mean()
        critic_loss = advantage.pow(2).mean(dim=0)#leaf=F Mean
        loss = (actor_loss + 0.5 * critic_loss - 0.001 * entropy).mean() # subbackward
        return loss
    
    def train(self,loss,optimizer):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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
        # dist = Categorical(probs) # [1,2,2],[,,,],()
        # action = dist.sample()
        # return action, dist.log_prob(action),dist.entropy().mean()
        distributions = [Categorical(row) for row in probs]
        actions = []
        log_probs = []
        entrops = []
        for dist in distributions:
            action = dist.sample().unsqueeze(0)
            log_prob = dist.log_prob(action).unsqueeze(0)
            entropy = dist.entropy().mean()
            actions.append(action)
            log_probs.append(log_prob)
            entrops.append(entropy)
        return torch.cat(actions).to(device), torch.cat(log_probs).to(device), torch.tensor(entrops).sum().to(device)

    def q_value(self, states,actions):
        actions = actions.unsqueeze(1)
        out = torch.cat([states, actions], dim=1)
        q_value = self.critic(out).to(device)
        return q_value
    
class PPO:
    def __init__(self,lr = 1e-3,gamma = 0.9,eps = 0.2,num_envs=3,render_mode = 'rgb_array'):
        self.gamma = gamma
        self.actorCritic = ActorCriticModel(eps).to(device)
        self.optimizer = optim.Adam(self.actorCritic.parameters(),lr=lr)
        print(sum([p.numel() for p in self.actorCritic.parameters() if p.requires_grad]))
        self.backup_model = copy.deepcopy(self.actorCritic)
        self.num_envs = num_envs
        envs = [lambda: self.makeEnv(render_mode) for _ in range(num_envs)]
        self.envs = gym.vector.SyncVectorEnv(envs)
    
    def makeEnv(self,render_mode):
        env_name = "CartPole-v1"
        return gym.make(env_name,render_mode=render_mode)
    
    def computeReturns(self,rewards,masks):
        n_steps = len(rewards) 
        disc_rewards = deque(maxlen=n_steps) 
        for t in range(n_steps)[::-1]:
            disc_return_t = (disc_rewards[0] if len(disc_rewards) > 0 else torch.tensor([0])).to(device)
            disc_rewards.appendleft( self.gamma*disc_return_t * masks[t] + rewards[t])
        return list(disc_rewards)
    
    def train(self,n_episodes,n_steps):
        episod = 0
        try:
            states = self.envs.reset()[0]
            done = torch.tensor([False])
            while episod < n_episodes:
                values    = list()
                ratio = list()
                rewards   = list()
                masks = list()
                # log_probs = list()
                entropy = 0
                episod += 1
                step = 0
                while step < n_steps:
                    if done.any():
                        index= torch.where(done == True)
                        states = self.envs[index].reset()[0]
                    action,prob, value, ent = self.actorCritic.forward(states) # value [[]] AddmmBackward
                    action2,prob2, ent2 = self.backup_model.only_act(states) # value [[]] AddmmBackward
                    next_state, reward, done, _ , _ = self.envs.step(action.cpu().numpy())
                    entropy += ent #[]
                    values.append(value.T) # leaf=F req [[]]
                    ratio.append(torch.exp(prob - prob2.detach()).T) #
                    masks.append(torch.tensor(1-done).unsqueeze(0).to(device))
                    rewards.append(torch.from_numpy(reward).unsqueeze(0).to(device)) # leaf=T 
                    states = next_state
                    step += 1
                returns = torch.cat(self.computeReturns(rewards,masks),dim=0).detach()
                print(f'Кумул. награда = {int(returns.cpu().sum(dim=1).sum().numpy())}')
                values    = torch.cat(values)
                ratio    = torch.cat(ratio)
                eps = np.finfo(np.float32).eps.item()
                returns = ((returns - returns.mean(dim=0)) / (returns.std(dim=0) + eps)).detach()
                self.backup_model = copy.deepcopy(self.actorCritic)
                loss = self.actorCritic.loss(returns,values,entropy.detach(),ratio)
                self.actorCritic.train(loss,self.optimizer)
                print(self.actorCritic.actor[0][0].weight)
        except Exception as e:
            print(e)

if __name__ == "__main__":
    pacman_hyperparameters = {
        "n_training_episodes": 5000000,
        "n_evaluation_episodes": 10,
        "n_steps":30,
        "gamma": 0.95,
        "lr": 1e-3,
        "num_envs":3,
        'render_mode':'human'
    }
    model = PPO(pacman_hyperparameters['lr'],pacman_hyperparameters['gamma'],num_envs=pacman_hyperparameters['num_envs'],render_mode=pacman_hyperparameters['render_mode'])
    model.train(pacman_hyperparameters['n_training_episodes'],pacman_hyperparameters['n_steps'])
    # stat = Statictics()
    # stat.readTotalReward()
    # stat.oneGraph(stat.total_rewards,'step','reward','blue')
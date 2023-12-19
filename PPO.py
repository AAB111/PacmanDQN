
from collections import deque
import numpy as np
from Pacman import Game
import torch
import copy
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import pygame
import matplotlib.pyplot as plt
use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

class ActorCriticModel(nn.Module):  
    def __init__(self,eps):
        super(ActorCriticModel, self).__init__()
        self.eps = eps
        self.cnn = nn.Sequential(
            nn.Sequential(nn.Conv2d(kernel_size=3,stride=2, in_channels=4,out_channels=32,padding="valid",),nn.ReLU(),nn.MaxPool2d(kernel_size=2,stride=2)),
            nn.Sequential(nn.Conv2d(kernel_size=3,stride=1, in_channels=32,out_channels=64,padding="same"),nn.ReLU(),nn.MaxPool2d(kernel_size=2,stride=2),nn.Dropout(0.2)),
            nn.Flatten()
        )
        nn.init.kaiming_normal_(self.cnn[0][0].weight,mode='fan_out',nonlinearity='relu')
        nn.init.kaiming_normal_(self.cnn[1][0].weight,mode='fan_out',nonlinearity='relu')

        self.critic = nn.Sequential(
            nn.Sequential(nn.Linear(in_features=64*3*3 + 2, out_features=64),nn.ReLU(),nn.Dropout(0.2)),
            nn.Sequential(nn.Linear(in_features=64, out_features=32),nn.ReLU(),nn.Dropout(0.2)),
            nn.Sequential(nn.Linear(in_features=32, out_features=1)),
        )
        nn.init.kaiming_normal_(self.critic[0][0].weight,mode='fan_out',nonlinearity='relu'),
        nn.init.xavier_uniform_(self.critic[0][0].weight)
        
        self.actor = nn.Sequential(
            nn.Sequential(nn.Linear(in_features=64*3*3 + 1, out_features=64),nn.ReLU(),nn.Dropout(0.2)),
            nn.Sequential(nn.Linear(in_features=64, out_features=32),nn.ReLU(),nn.Dropout(0.2)),
            nn.Sequential(nn.Linear(in_features=32, out_features=4),nn.Softmax(dim=1)),
        )
        nn.init.kaiming_normal_(self.actor[0][0].weight,mode='fan_out',nonlinearity='relu'),
        nn.init.xavier_uniform_(self.actor[0][0].weight)

    def printProbsAction(self,probs,action):
        actions = {0:'↑',1:'→',2:'↓',3:'←'}
        print(f'Действие = {actions[action.item()]} Расп = {probs.cpu().detach().numpy()}')
    
    def loss(self, returns, values,entropy,log_probs):
        advantage = returns - values # leaf = F req
        # ratio = ratio.detach()
        # unclip = ratio * advantage
        # clip = torch.clamp(ratio,1 - self.eps,1 + self.eps) * advantage
        # min = torch.min(unclip,clip)
        # actor_loss = -(minRatio * advantage.detach()).mean() #NEgbackward
        actor_loss = (-log_probs*advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()#leaf=F Mean
        loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy # subbackward
        return loss
    
    def train(self,loss,optimizer):
        optimizer.zero_grad()
        loss.to(device)
        loss.backward()
        optimizer.step()

    def forward(self, state,time_game):
        state = torch.from_numpy(state).float().unsqueeze(0)
        state = np.transpose(state, (0, 3, 2, 1)).cpu().to(device)
        time = torch.tensor([[1 - time_game]]).to(device)
        cnn_res = self.cnn(state).to(device)
        action,prob,entropy = self.act(cnn_res,time)
        value = self.q_value(cnn_res,action,time)
        return action, prob, value, entropy
    
    def act(self,state_after_cnn,time_game):
        state_after_cnn = torch.cat([state_after_cnn, time_game], dim=1)
        probs = self.actor(state_after_cnn).to(device)
        dist = Categorical(probs)
        action = dist.sample()
        # self.printProbsAction(probs,action)
        return action.item(), dist.log_prob(action), dist.entropy().mean()

    def q_value(self, state_after_cnn,action,time_game):
        action = torch.tensor([[action]]).to(device)
        out = torch.cat([state_after_cnn, action,time_game], dim=1)
        q_value = self.critic(out).to(device)
        return q_value
    
class PPO:
    def __init__(self,lr = 1e-3,gamma = 0.9,eps = 0.2):
        self.gamma = gamma
        self.actorCritic = ActorCriticModel(eps).to(device)
        self.optimizer = optim.Adam(self.actorCritic.parameters(),lr=lr,amsgrad=True)
        self.backup_model = self.actorCritic
        self.statictics = Statictics()

    def compute_returns(self,rewards):
        n_steps = len(rewards) 
        disc_rewards = deque(maxlen=n_steps) 
        for t in range(n_steps)[::-1]:
            disc_return_t = torch.FloatTensor([(disc_rewards[0] if len(disc_rewards) > 0 else 0)]).to(device)
            disc_rewards.appendleft( self.gamma*disc_return_t + rewards[t])
        return list(disc_rewards)
    
    def printInfoState(self,state,action,reward,total_reward,next_state,time_game):
        actions = {0:'↑',1:'→',2:'↓',3:'←'}
        print(f'Состояние = {state} Действие = {actions[action]} Награда = {reward} Сумм.Награда = {total_reward} След.состояние = {next_state} время вышло = {time_game}\n')

    def writeModel(self):
        state  = {
            'model' : self.actorCritic.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(state, 'PPO/Model.pt')
    
    def readModel(self):
        try:
            state = torch.load('PPO/Model.pt')
            self.actorCritic.load_state_dict(state['model']) 
            self.optimizer.load_state_dict(state['optimizer'])
            print(sum([p.numel() for p in self.actorCritic.parameters() if p.requires_grad]))
            print(self.actorCritic.cnn[0][0].weight)
            print(self.actorCritic.cnn[1][0].weight)
            print(self.actorCritic.actor[1][0].weight)
            print(self.actorCritic.actor[0][0].weight)
            print(self.actorCritic.actor[1][0].weight)
            print(self.actorCritic.critic[0][0].weight)
            print(self.actorCritic.critic[1][0].weight)
        except Exception as e:
            print(e)

    def train(self,n_episodes):
        episod = 0
        self.readModel()
        self.statictics.readTotalReward()
        try:
            while episod < n_episodes:
                values    = list()
                # ratio = list()
                rewards   = list()
                log_probs = list()
                entropy = 0
                episod += 1
                time_game = 1
                is_end = False
                done = False
                env.newLevel()
                state = env.getStateEnv()
                while not done:
                    action,prob, value, ent = self.actorCritic.forward(state,time_game) # value [[]] AddmmBackward
                    # action2,prob2, value2, ent2 = self.backup_model.forward(state,time_game) # value [[]] AddmmBackward
                    # state_pacman_before_action =  env.getStatePacman()
                    next_state, reward, done, time_game = env.step(action)
                    # print(reward)
                    # new_state_pacman = env.getStatePacman()
                    # self.printInfoState(state_pacman_before_action,action,reward,sum(rewards),new_state_pacman,time_game)
                    entropy += ent
                    log_probs.append(prob)
                    values.append(value.squeeze(0)) # leaf=F req
                    # ratio.append(torch.exp(prob/prob2)) #
                    rewards.append(torch.tensor([reward], dtype=torch.float, device=device)) # leaf=T 
                    state = next_state
                    for event in pygame.event.get():
                        if event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_q:
                                done = True
                                is_end = True
                                break
                if is_end:
                    break
                returns = torch.cat(self.compute_returns(rewards)).detach() # leaf = T
                self.statictics.total_rewards = np.append(self.statictics.total_rewards, returns.cpu().sum().numpy())
                print(f'Кумул. награда = {int(returns.cpu().sum().numpy())} Средняя = {int(self.statictics.total_rewards.mean())} Откл = {int(self.statictics.total_rewards.std())}')
                values    = torch.cat(values)#catbackward [,,,,] leaf =F req
                log_probs = torch.cat(log_probs)
                # ratio    = torch.cat(ratio)#catbackward [,,,,] leaf =F req
                eps = np.finfo(np.float32).eps.item()
                returns = (returns - returns.mean()) / (returns.std() + eps).detach()
                # returns = (returns - returns.min()) / (returns.max() - returns.min())
                # self.backup_model = copy.deepcopy(self.actorCritic)
                loss = self.actorCritic.loss(returns,values,entropy.detach(),log_probs)
                self.actorCritic.train(loss,self.optimizer)
                if episod % 50 == 0:
                    self.writeModel()
                    self.statictics.writeTotalReward()
                    print("SAVE MODEL")
        except Exception as e:
            print(e)
        finally:
            self.writeModel()
            self.statictics.writeTotalReward()
            print('SAVE MODEL')
            

class Statictics:
    def __init__(self):
        self.hist_loss = []
        self.hist_val_loss = []
        self.total_rewards = []
    
    def writeTotalReward(self):
        with open('PPO/total_reward.txt','w') as f:
            np.savetxt(f,self.total_rewards,fmt='%f')
    def readTotalReward(self):
        with open('PPO/total_reward.txt','r') as f:
            self.total_rewards = np.loadtxt(f,dtype=float)
    def writeHistLoss(self):
        with open('PPO/history_loss.txt','w') as f:
            np.savetxt(f,self.hist_loss,fmt='%f')
    def readHistLoss(self):
        with open('PPO/history_loss.txt','r') as f:
            self.hist_loss = np.loadtxt(f,dtype=float)
    def writeHistValLoss(self):
        with open('PPO/history_val_loss.txt','w') as f:
            np.savetxt(f,self.hist_val_loss,fmt='%f')
    def readHistValLoss(self):
        with open('PPO/history_val_loss.txt','r') as f:
            self.hist_val_loss = np.loadtxt(f,dtype=float)
    def oneGraph(self,x,xlabel,ylabel,color):
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.plot(range(x.shape[0]),x,color = color)
        plt.show()
    def twoGraph(self,x1,x2,xlabel,ylabel,colorx1,colorx2):
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.plot(range(len(x1)),x1,color = colorx1,label = 'loss')
        plt.plot(range(len(x2)),x2,color = colorx2,label = 'val_loss')
        plt.legend(loc = 'upper left')
        plt.show()


if __name__ == "__main__":
    env = Game(1,0)
    pacman_hyperparameters = {
        "n_training_episodes": 10000,
        "n_evaluation_episodes": 10,
        "gamma": 0.95,
        "lr": 1e-5,
    }
    model = PPO(pacman_hyperparameters['lr'],pacman_hyperparameters['gamma'])
    model.train(pacman_hyperparameters['n_training_episodes'])
    # stat = Statictics()
    # stat.readTotalReward()
    # stat.oneGraph(stat.total_rewards,'step','reward','blue')

import time
from collections import deque
import numpy as np
import torch
import gym
import copy
import ale_py
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

class ActorCriticModel(nn.Module):  
    def __init__(self,eps):
        super(ActorCriticModel, self).__init__()
        self.eps = eps
        self.cnn = nn.Sequential(
            nn.Sequential(nn.Conv2d(kernel_size=8,stride=4, in_channels=3,out_channels=32,padding="valid"),nn.ReLU()),
            nn.Sequential(nn.Conv2d(kernel_size=4,stride=2, in_channels=32,out_channels=64,padding="valid"),nn.ReLU(),nn.Dropout(0.2)),
            nn.Sequential(nn.Conv2d(kernel_size=3,stride=1, in_channels=64,out_channels=64,padding="valid"),nn.ReLU(),nn.Dropout(0.2)),
            nn.Flatten()
        )
        nn.init.kaiming_normal_(self.cnn[0][0].weight,mode='fan_out',nonlinearity='relu')
        nn.init.kaiming_normal_(self.cnn[1][0].weight,mode='fan_out',nonlinearity='relu')
        nn.init.kaiming_normal_(self.cnn[2][0].weight,mode='fan_out',nonlinearity='relu')

        self.critic = nn.Sequential(
            nn.Sequential(nn.Linear(in_features=64*6*7 + 1, out_features=256),nn.ReLU(),nn.Dropout(0.2)),
            # nn.Sequential(nn.Linear(in_features=128, out_features=256),nn.ReLU(),nn.Dropout(0.2)),
            nn.Sequential(nn.Linear(in_features=256, out_features=1)),
        )
        nn.init.kaiming_normal_(self.critic[0][0].weight,mode='fan_out',nonlinearity='relu'),
        # nn.init.kaiming_normal_(self.critic[1][0].weight,mode='fan_out',nonlinearity='relu'),
        nn.init.xavier_uniform_(self.critic[1][0].weight)
        
        self.actor = nn.Sequential(
            nn.Sequential(nn.Linear(in_features=64*6*7, out_features=256,device=device),nn.ReLU(),nn.Dropout(0.2)),
            # nn.Sequential(nn.Linear(in_features=128, out_features=256,device=device),nn.ReLU(),nn.Dropout(0.2)),
            nn.Sequential(nn.Linear(in_features=256, out_features=9,device=device),nn.Softmax(dim=-1)),
        )
        nn.init.kaiming_normal_(self.critic[0][0].weight,mode='fan_out',nonlinearity='relu'),
        # nn.init.kaiming_normal_(self.critic[1][0].weight,mode='fan_out',nonlinearity='relu'),
        nn.init.xavier_uniform_(self.critic[1][0].weight)

    def printProbsAction(self,probs,action):
        actions = {0:'Noop',
                    1:'↑',
                    2:'→',
                    3:'↓',
                    4:'←',
                    5:'↑→',
                    6:'↑←',
                    7:'↓→',
                    8:'↓←'}
        print(f'Действие = {actions[action.item()]} Расп = {probs.cpu().detach().numpy()}')
    
    def loss(self, returns, values,entropy,ratio):
        advantage = returns - values # leaf = F req
        unclip = ratio * advantage
        clip = torch.clamp(ratio,1 - self.eps,1 + self.eps) * advantage
        actor_loss = (-torch.min(unclip,clip)).mean() 
        critic_loss = advantage.pow(2).mean()#leaf=F Mean
        loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy # subbackward
        return loss
    
    def train(self,loss,optimizer):
        optimizer.zero_grad()
        loss.to(device)
        loss.backward()
        optimizer.step()

    def forward(self, states):
        cnn_res = self.cnn(states).to(device)
        actions,log_probs,entropy = self.act(cnn_res)
        values = self.q_value(cnn_res,actions)
        return actions, log_probs, values, entropy
    
    def act(self,states_after_cnn):
        probs = self.actor(states_after_cnn).to(device)
        dist = Categorical(probs)
        actions = dist.sample()
        return actions, dist.log_prob(actions), dist.entropy().mean()

    def q_value(self, states_after_cnn,actions):
        actions = actions.unsqueeze(1)
        out = torch.cat([states_after_cnn, actions], dim=1)
        q_value = self.critic(out).to(device)
        return q_value
    
    def only_act(self,states):
        cnn_res = self.cnn(states).to(device)
        actions,log_probs,entropy = self.act(cnn_res)
        return actions,log_probs,entropy
    
class PPO:
    def __init__(self,lr = 1e-3,gamma = 0.9,eps = 0.2,render_mode = 'rgb_array'):
        self.gamma = gamma
        self.actorCritic = ActorCriticModel(eps).to(device)
        self.optimizer = optim.Adam(self.actorCritic.parameters(),lr=lr,amsgrad=True)
        print(sum([p.numel() for p in self.actorCritic.parameters() if p.requires_grad]))
        self.backup_model = copy.deepcopy(self.actorCritic)
        # self.num_envs = num_envs
        # envs = [lambda: self.makeEnv(render_mode) for _ in range(num_envs)]
        # self.envs = gym.vector.make("MsPacman-v4",num_envs=num_envs,asynchronous=False)
        # self.envs = gym.vector.SyncVectorEnv(envs)
        self.env = gym.make('MsPacman-v4',render_mode=render_mode)
        self.statictics = Statictics()
        self.colorPacman = np.array([210, 164, 74]).mean()
        self.colorWall = np.array([228,111,111]).mean()
        self.colorGhost1 = np.array([200, 72, 72]).mean()
        self.colorGhost2 = np.array([198, 89, 179]).mean()
        self.colorGhost3 = np.array([180, 122, 48]).mean()
        self.colorGhost4 = np.array([84, 184, 153]).mean()

    def makeEnv(self,render_mode):
        env_name = "MsPacman-v4"
        return gym.make(env_name,render_mode=render_mode,device=device)

    def computeReturns(self,rewards,masks):
        n_steps = len(rewards) 
        disc_rewards = deque(maxlen=n_steps) 
        for t in range(n_steps)[::-1]:
            disc_return_t = (disc_rewards[0] if len(disc_rewards) > 0 else torch.tensor([0])).to(device)
            disc_rewards.appendleft( self.gamma*disc_return_t * masks[t] + rewards[t])
        return list(disc_rewards)
    
    def printInfoState(self,state,action,reward,total_reward,next_state,time_game):
        actions = {0:'Noop',
                    1:'↑',
                    2:'→',
                    3:'↓',
                    4:'←',
                    5:'↑→',
                    6:'↑←',
                    7:'↓→',
                    8:'↓←'}
        print(f'Состояние = {state} Действие = {actions[action]} Награда = {reward} Сумм.Награда = {total_reward} След.состояние = {next_state} время вышло = {time_game}\n')

    def writeModel(self):
        state  = {
            'model' : self.actorCritic.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(state, 'PPOgym/Model.pt')
    
    def readModel(self):
        try:
            state = torch.load('PPOgym/Model.pt')
            self.actorCritic.load_state_dict(state['model']) 
            self.optimizer.load_state_dict(state['optimizer'])
            self.backup_model = copy.deepcopy(self.actorCritic)
            print(sum([p.numel() for p in self.actorCritic.parameters() if p.requires_grad]))
            print(self.actorCritic.cnn[1][0].weight)
            print(self.actorCritic.actor[1][0].weight)
            print(self.actorCritic.critic[1][0].weight)
        except Exception as e:
            print(e)
    
    def preprocessState(self,state):
        image = state[1:172:2, ::2]
        image = image.mean(axis=-1)
        image[image==self.colorPacman] = 0
        image[image==self.colorGhost1] = 155
        image[image==self.colorGhost2] = 155
        image[image==self.colorGhost3] = 155
        image[image==self.colorGhost4] = 155
        image[image==self.colorWall] = 255
        # layerWall = np.where(image==colorWall,image,0)
        # layerGhosts =np.where((image==colorGhost1) | (image==colorGhost2) | (image==colorGhost3) | (image==colorGhost4),image,0)
        # layerPacman = np.where(image==colorPacman,image,0)
        # image = np.stack((layerPacman,layerGhosts,layerWall))
        eps = np.finfo(np.float32).eps.item()
        image = (image - image.min()) / (image.max() - image.min() + eps)
        image = image.reshape(1,80,86)
        return image

    def preprocessStates(self,states):
        states = np.array(states).reshape(1,3,80,86)
        return torch.from_numpy(states).float().to(device)
    
    def addState(self,states,state):
        states.pop(0)
        states.append(state)

    def train(self,n_episodes,n_steps):
        episod = 0
        self.readModel()
        self.statictics.readTotalReward()
        try:
            states = []
            zerosNP = np.zeros((1,80,86))
            states.append(zerosNP)
            states.append(zerosNP)
            states.append(zerosNP)
            state = self.preprocessState(self.env.reset()[0])
            self.addState(states,state)
            terminal = False
            lives = 3
            while episod < n_episodes:
                values    = []
                ratio = []
                masks = []
                rewards   = []
                entropy = 0
                episod += 1
                step = 0
                rewardsToPrint = 0
                while step < n_steps:
                    if terminal:
                        states.clear()
                        states.append(zerosNP)
                        states.append(zerosNP)
                        states.append(zerosNP)
                        state = self.preprocessState(self.env.reset()[0])
                        self.addState(states,state)
                        lives = 3
                    pre_states = self.preprocessStates(states)
                    action,log_prob, value, ent = self.actorCritic.forward(pre_states) 
                    action2,log_probOld, ent2 = self.backup_model.only_act(pre_states) 
                    next_state, reward, terminal, truncated , info = self.env.step(action)
                    if reward == 200:
                        reward = 20
                    elif reward == 50:
                        reward = 15
                    entropy += ent
                    rewardsToPrint += reward
                    values.append(value.squeeze(1)) 
                    ratio.append(torch.exp(log_prob-log_probOld.detach())) 
                    rewards.append(torch.tensor([reward]).to(device))
                    masks.append(torch.tensor(1-terminal).to(device))
                    state = self.preprocessState(next_state)
                    self.addState(states,state)
                    step += 1 
                returns = torch.cat(self.computeReturns(rewards,masks)).detach()
                self.statictics.total_rewards = np.append(self.statictics.total_rewards, rewardsToPrint)
                print(f'награда = {int(rewardsToPrint)} Средняя = {int(self.statictics.total_rewards[-300:].mean())} Откл = {int(self.statictics.total_rewards[-300:].std())}')
                values    = torch.cat(values)#catbackward [,,,,] leaf =F req
                ratio    = torch.cat(ratio)#catbackward [,,,,] leaf =F req
                eps = np.finfo(np.float32).eps.item()
                returns = ((returns - returns.mean()) / (returns.std() + eps)).detach()
                self.backup_model = copy.deepcopy(self.actorCritic)
                loss = self.actorCritic.loss(returns,values,entropy.detach(),ratio)
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
        with open('PPOgym/total_reward.txt','w') as f:
            np.savetxt(f,self.total_rewards,fmt='%f')
    def readTotalReward(self):
        with open('PPOgym/total_reward.txt','r') as f:
            self.total_rewards = np.loadtxt(f,dtype=float)
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
    pacman_hyperparameters = {
        "n_training_episodes" : 100000,
        "n_evaluation_episodes" : 10,
        "n_steps" : 600,
        "gamma" : 0.9,
        "lr" : 1e-5,
        'render_mode' : 'human'
    }
    model = PPO(pacman_hyperparameters['lr'],pacman_hyperparameters['gamma'],render_mode=pacman_hyperparameters['render_mode'])
    model.train(pacman_hyperparameters['n_training_episodes'],pacman_hyperparameters['n_steps'])
    # stat = Statictics()
    # stat.readTotalReward()
    # stat.oneGraph(stat.total_rewards,'step','reward','blue')
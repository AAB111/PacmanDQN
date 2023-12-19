import numpy as np
import torch,time,enum
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque
import gym
import matplotlib.pyplot as plt

class HyperParametrs(enum.Enum):
    n_training_episodes = 10000
    n_evaluation_episodes = 10
    n_steps = 1500
    gamma = 0.99
    lr = 0.0001
    weight_decay = 1e-4
    render_mode = 'human'
    skip_episodes = 1
    print_info_after_episods = 10
    save_model_after_episods = 200
    print_lead_time = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')
class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.policy = nn.Sequential(
            nn.Sequential(nn.Conv2d(kernel_size=3,stride=1, in_channels=3,out_channels=30,padding="same",device=device),nn.BatchNorm2d(num_features=30),nn.ReLU(),nn.MaxPool2d(kernel_size=2,stride=2)),
            nn.Sequential(nn.Conv2d(kernel_size=3,stride=1, in_channels=30,out_channels=34,padding="same",device=device),nn.BatchNorm2d(num_features = 34),nn.ReLU(),nn.MaxPool2d(kernel_size=2,stride=2)),
            nn.Sequential(nn.Conv2d(kernel_size=3,stride=1, in_channels=34,out_channels=38,padding="same",device=device),nn.BatchNorm2d(num_features = 38),nn.ReLU(),nn.MaxPool2d(kernel_size=2,stride=2)),
            nn.Sequential(nn.Conv2d(kernel_size=3,stride=1, in_channels=38,out_channels=42,padding="same",device=device),nn.BatchNorm2d(num_features = 42),nn.ReLU(),nn.MaxPool2d(kernel_size=2,stride=2)),
            nn.Flatten(),
            nn.Sequential(nn.Linear(in_features=42*5*5, out_features=256,device=device), nn.Dropout(p=0.3),nn.ReLU()),
            nn.Sequential(nn.Linear(in_features=256, out_features=32,device=device), nn.Dropout(p=0.1),nn.ReLU()),
            nn.Sequential(nn.Linear(in_features=32, out_features=9,device=device), nn.Softmax(dim=-1))
        )
        nn.init.kaiming_normal_(self.policy[0][0].weight,mode='fan_out',nonlinearity='relu')
        nn.init.kaiming_normal_(self.policy[1][0].weight,mode='fan_out',nonlinearity='relu')
        nn.init.kaiming_normal_(self.policy[2][0].weight,mode='fan_out',nonlinearity='relu')
        nn.init.kaiming_normal_(self.policy[3][0].weight,mode='fan_out',nonlinearity='relu')
        nn.init.kaiming_normal_(self.policy[5][0].weight,mode='fan_out',nonlinearity='relu')
        nn.init.kaiming_normal_(self.policy[6][0].weight,mode='fan_out',nonlinearity='relu')
        nn.init.xavier_uniform_(self.policy[7][0].weight)

    def forward(self, states):
        probs = self.policy(states)
        dist = Categorical(probs)
        actions = dist.sample()
        return actions, dist.log_prob(actions),dist.entropy().mean()


class PolicyGradient:
    def __init__(self):
        self.gamma=HyperParametrs.gamma.value
        self.policy = Policy().to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=HyperParametrs.lr.value,weight_decay=HyperParametrs.weight_decay.value)
        # self.scheldure = optim.lr_scheduler.StepLR(self.optimizer,step_size=300,gamma=0.8)
        print(sum([p.numel() for p in self.policy.parameters() if p.requires_grad]))
        self.env = gym.make('MsPacman-v4',render_mode=HyperParametrs.render_mode.value)
        self.colorPacman = np.array([210, 164, 74]).mean()
        self.colorWall = np.array([228,111,111]).mean()
        self.colorGhost1 = np.array([200, 72, 72]).mean()
        self.colorGhost2 = np.array([198, 89, 179]).mean()
        self.colorGhost3 = np.array([180, 122, 48]).mean()
        self.colorGhost4 = np.array([84, 184, 153]).mean()
        self.statictics = Statictics()

    def loss(self,log_probs,returns,entropy):
        loss = log_probs * -returns
        actor_loss = loss.sum() - 0.01 * entropy
        return actor_loss
    
    def step_optimizer(self,loss,optimizer):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def writeModel(self):
        state  = {
            'model' : self.policy.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(state, 'Policy/Model.pt')
    
    def readModel(self):
        try:
            state = torch.load('Policy/Model.pt')
            self.policy.load_state_dict(state['model']) 
            self.optimizer.load_state_dict(state['optimizer'])
            print(sum([p.numel() for p in self.policy.parameters() if p.requires_grad]))
            print(self.policy.policy[5][0].weight)
        except Exception as e:
            print(e)

    def compute_returns(self,rewards,masks):
        n_steps = len(rewards) 
        disc_rewards = deque(maxlen=n_steps) 
        R = 0
        for t in range(n_steps)[::-1]:
            R = self.gamma * R * masks[t] + rewards[t]
            # disc_return_t = (disc_rewards[0] if len(disc_rewards) > 0 else torch.tensor([0])).to(device)
            disc_rewards.appendleft(R)
        return list(disc_rewards)
    
    def preprocessState(self,state):
        image = state[1:172:2, ::2]
        image = image.mean(axis=-1)
        image[image==self.colorPacman] = 0
        image[image==self.colorGhost1] = 155
        image[image==self.colorGhost2] = 155
        image[image==self.colorGhost3] = 155
        image[image==self.colorGhost4] = 155
        image[image==self.colorWall] = 255
        image = image / 255
        image = image.reshape(1,80,86)
        return image

    def preprocessStates(self,states):
        states = np.array(states).reshape(1,3,80,86)
        return torch.from_numpy(states).float().to(device)
    
    def addState(self,states,state):
        states.pop(0)
        states.append(state)
    
    def resetEnv(self, states, zerosNP):
        states.clear()
        states.append(zerosNP)
        states.append(zerosNP)
        states.append(zerosNP)
        state = self.preprocessState(self.env.reset()[0])
        self.addState(states,state)

    def train(self):
        self.readModel()
        self.statictics.readTotalReward()
        try:
            isDead = False
            terminated = False
            states = []
            log_probs = []
            masks = []
            rewards   = []
            zerosNP = np.zeros((1,80,86))
            self.resetEnv(states,zerosNP)
            lives = 3
            for i_episode in range(1, HyperParametrs.n_training_episodes.value + 1):
                entropy = 0
                reward_episod = 0
                if (HyperParametrs.print_lead_time.value):
                    start = time.time()
                if terminated:
                    lives = 3
                    self.resetEnv(states, zerosNP)
                for step in range(1, HyperParametrs.n_steps.value):
                    pre_states = self.preprocessStates(states)
                    action,log_prob,ent = self.policy.forward(pre_states) 
                    next_state, reward, terminated, truncated , info = self.env.step(action)
                    if lives != info['lives']:
                        lives = info['lives']
                        isDead = True
                        reward = -20
                    elif reward == 200:
                        reward = 5
                    elif reward == 50:
                        reward = 15
                    reward_episod += reward
                    entropy += ent
                    log_probs.append(log_prob)
                    rewards.append(torch.tensor([reward],device=device))
                    if (isDead):
                        masks.append(torch.tensor(1 - isDead,device=device))
                        isDead = False
                    else:
                        masks.append(torch.tensor(1-terminated,device=device))
                    state = self.preprocessState(next_state)
                    self.addState(states,state)
                    if terminated:
                        break
                self.statictics.total_rewards = np.append(self.statictics.total_rewards, reward_episod)
                if (i_episode > HyperParametrs.skip_episodes.value):
                    returns = torch.cat(self.compute_returns(rewards,masks))
                    log_prob_tensor    = torch.cat(log_probs)
                    eps = np.finfo(np.float32).eps.item()
                    returns = ((returns - returns.mean()) / (returns.std() + eps)).detach()
                    loss = self.loss(log_prob_tensor,returns,entropy.detach())
                    self.step_optimizer(loss,self.optimizer)
                    del rewards[:]
                    del log_probs[:]
                    del masks[:]
                    if (HyperParametrs.print_lead_time.value):
                        end = time.time() - start
                        print(end)
                if i_episode % HyperParametrs.print_info_after_episods.value == 0:
                    print(f'Эпизод = {i_episode} Последняя награда = {int(reward_episod)} Средняя = {int(self.statictics.total_rewards[-HyperParametrs.print_info_after_episods.value:].mean())} Откл = {int(self.statictics.total_rewards[-HyperParametrs.print_info_after_episods.value:].std())}')
                if i_episode % HyperParametrs.save_model_after_episods.value == 0:
                    print(self.policy.policy[5][0].weight)
                    self.writeModel()
                    self.statictics.writeTotalReward()
                    print("SAVE MODEL")
        except Exception as e:
            print(e)
        finally:
            self.writeModel()
            self.statictics.writeTotalReward()
            print('SAVE MODEL')

    def evaluateAgent(self):
        self.readModel()
        try:
            terminated = False
            states = []
            zerosNP = np.zeros((1,80,86))
            self.resetEnv(states,zerosNP)
            with torch.no_grad():
                for i_episode in range(1, HyperParametrs.n_evaluation_episodes.value+1):
                    reward_episod = 0
                    if terminated:
                        self.resetEnv(states, zerosNP)
                    for step in range(0, HyperParametrs.n_steps.value):
                        pre_states = self.preprocessStates(states)
                        action,log_prob,ent = self.policy.forward(pre_states) 
                        next_state, reward, terminated, truncated , info = self.env.step(action)
                        state = self.preprocessState(next_state)
                        self.addState(states,state)
                        if terminated:
                            break
        except Exception as ex:
            print(ex)

class Statictics:
    def __init__(self):
        self.total_rewards = []
    
    def writeTotalReward(self):
        with open('Policy/total_reward.txt','w') as f:
            np.savetxt(f,self.total_rewards,fmt='%f')
        
    def readTotalReward(self):
        with open('Policy/total_reward.txt','r') as f:
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
    policyGrad = PolicyGradient()
    policyGrad.train()
    # policyGrad.evaluateAgent()
    # policyGrad.statictics.readTotalReward()
    # policyGrad.statictics.oneGraph(policyGrad.statictics.total_rewards,'Эпизод','Награда','blue')
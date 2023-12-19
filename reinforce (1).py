import gym
import numpy as np
from itertools import count
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.policy = nn.Sequential(
            nn.Sequential(nn.Conv2d(kernel_size=8,stride=4, in_channels=3,out_channels=32,padding="valid"),nn.ReLU()),
            nn.Sequential(nn.Conv2d(kernel_size=4,stride=2, in_channels=32,out_channels=64,padding="valid"),nn.ReLU(),nn.Dropout(0.2)),
            nn.Sequential(nn.Conv2d(kernel_size=3,stride=1, in_channels=64,out_channels=64,padding="valid"),nn.ReLU(),nn.Dropout(0.2)),
            nn.Flatten(),
            nn.Sequential(nn.Linear(in_features=64*6*7, out_features=256),nn.ReLU(),nn.Dropout(0.2)),
            nn.Sequential(nn.Linear(in_features=256, out_features=9),nn.Softmax(dim=-1)),
        )
        nn.init.kaiming_normal_(self.cnn[0][0].weight,mode='fan_out',nonlinearity='relu')
        nn.init.kaiming_normal_(self.cnn[1][0].weight,mode='fan_out',nonlinearity='relu')
        nn.init.kaiming_normal_(self.cnn[2][0].weight,mode='fan_out',nonlinearity='relu')
        nn.init.kaiming_normal_(self.actor[4][0].weight,mode='fan_out',nonlinearity='relu'),
        nn.init.xavier_uniform_(self.actor[5][0].weight)

env = gym.make('MsPacman-v4',render_mode='rgb_array')
policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=1e-3)
eps = np.finfo(np.float32).eps.item()
rewards = []
ratio = []

def preprocessState(state):
    image = state[1:172:2, ::2]
    image = image.mean(axis=-1)
    colorPacman = np.array([210, 164, 74]).mean()
    colorWall = np.array([228,111,111]).mean()
    colorGhost1 = np.array([200, 72, 72]).mean()
    colorGhost2 = np.array([198, 89, 179]).mean()
    colorGhost3 = np.array([180, 122, 48]).mean()
    colorGhost4 = np.array([84, 184, 153]).mean()
    image[image==colorPacman] = 0
    image[image==colorGhost1] = 155
    image[image==colorGhost2] = 155
    image[image==colorGhost3] = 155
    image[image==colorGhost4] = 155
    image[image==colorWall] = 255
    eps = np.finfo(np.float32).eps.item()
    image = (image - image.min()) / (image.max() - image.min() + eps)
    image = image.reshape(1,80,86)
    return image

def preprocessStates(states):
        states = np.array(states).reshape(1,3,80,86)
        return torch.from_numpy(states).float()
    
def addState(states,state):
    states.pop(0)
    states.append(state)


def select_action(state,model):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = model(state)
    m = Categorical(probs)
    action = m.sample()
    return action.item(),m.log_prob(action)

def finish_episode():
    R = 0
    returns = deque()
    for r in policy.rewards[::-1]:
        R = r + 0.99 * R
        returns.appendleft(R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    ratios = torch.cat(ratio)
    unclip = ratios * -returns
    clip = torch.clamp(ratios,0.8,1.2) * -returns
    policy_loss = torch.max(unclip,clip).sum()
    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()
    del rewards[:]
    del ratio[:]

def main():
    running_reward = 10
    for i_episode in count(1):
        state, _ = env.reset()
        ep_reward = 0
        states = []
        ratio = []
        masks = []
        rewards   = []
        zerosNP = np.zeros((1,80,86))
        states.append(zerosNP)
        states.append(zerosNP)
        states.append(zerosNP)
        state = preprocessState(env.reset()[0])
        addState(states,state)
        done = False
        step = 0
        rewardsToPrint = 0
        while not done:
            action,log_prob = select_action(state,policy)
            actionOld,log_prob_old = select_action(state,)
            next_state, reward, done, _, _ = env.step(action)
            if reward == 200:
                reward = 20
            elif reward == 50:
                reward = 15
            ratio.append(torch.exp(log_prob-log_prob_old.detach())) 
            rewards.append(torch.tensor([reward]))
            masks.append(torch.tensor(1-done))
            state = preprocessState(next_state)
            addState(states,state)
            step += 1 
            rewards.append(reward)
            rewardsToPrint += reward
            if done:
                break

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        finish_episode()
        if i_episode % 10 == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break


if __name__ == '__main__':
    main()

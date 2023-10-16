import numpy as np
import pygame
from Pacman import game

class QlearningAgent:
    def __init__(self,env, gamma = 0.9, alpha = 0.3, number_episods = 100, epsilon=0.2):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.number_episods = number_episods
        self.epsilon = epsilon

    def egreedy_policy(self,q_values,state):
        if (np.random.random() < self.epsilon):
            return np.random.choice(self.env.pacman.number_actions)
        else:
            return np.argmax(q_values[state])

    def getIndexStateEnvFromCoordinat(self,state):
        index = state[0] * self.env.number_states['col'] + state[1]
        return index

    def writeQvalues(self,q_values):
        with open('Data/Qvalues.txt','w') as f:
            np.savetxt(f,q_values)

    def readQvalues(self):
        with open('Data/Qvalues.txt','r') as f:
            q_values = np.loadtxt(f)
        return q_values
    
    def printInfoState(self,state,action,reward,total_reward,next_state):
        actions = {0:'↑',1:'→',2:'↓',3:'←'}
        print(f'Состояние = {state} Действие = {actions[action]} Награда = {reward} Сумм.Награда = {total_reward} След.Состояние = {next_state}')

    def train(self):
        q_values = self.readQvalues()
        if(len(q_values) == 0):
            q_values = np.zeros([self.env.number_states['col'] * self.env.number_states['row'],self.env.pacman.number_actions])
        clock = pygame.time.Clock()
        totals_reward = []
        isEnd = False
        try:
            game.render()
            for _ in range(self.number_episods):
                state = self.env.getStatePacman()
                done = False
                total_reward = 0
                while not done:
                    clock.tick(30)
                    action = self.egreedy_policy(q_values, state) ## Выбираем оптимальное действие
                    next_state, reward, done = self.env.step(action) ## Выполняем действие
                    total_reward += reward
                    td_target = reward + self.gamma * np.max(q_values[next_state]) ##
                    td_error = td_target - q_values[state][action]
                    q_values[state][action] += self.alpha * td_error
                    self.printInfoState(state,action,reward,total_reward,next_state)
                    state = next_state
                    for event in pygame.event.get():
                        if event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_q:
                                done = True
                                isEnd = True
                                break
                totals_reward.append(np.array(total_reward))
                if isEnd:
                    break
        except Exception as e:
          print(e)
        finally:
            game.recordHighScore()
            self.writeQvalues(q_values)
        print(f'Средняя награда = {np.mean(np.array(totals_reward))}')


if __name__ == "__main__":
    agent = QlearningAgent(env = game,number_episods=10)
    agent.train()


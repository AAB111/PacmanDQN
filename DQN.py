from keras.models import Sequential,load_model,save_model
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from keras.losses import mean_squared_error
from keras.initializers import RandomUniform
import numpy as np
import pygame
from Pacman import Game
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class DQN:
    def __init__(self,env,gamma = 0.9,alpha = 0.4, number_episods = 100, epsilon=0.2):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.number_episods = number_episods 
        self.epsilon = epsilon 
        self.new_weights = 100 # к-во шагов, после к-ых веса копируются в target
        self.interval_of_training = 20 # к-во шагов после к-ых мы корректируем веса
        self.batch_size = 100 #размер батча для обучения
        self.memory_size = 1000 # размер памяти
        self.replay_memory = ReplayMemory(self.memory_size,self.batch_size)
        self.channels = 1 # 1 изображение
        self.size_state = (self.env.number_states['row'],self.env.number_states['col'],self.channels)
        self.count_actions = self.env.pacman.number_actions
        self.main_network = self.createModel()
        self.target_network = self.createModel()
        self.target_network.set_weights(self.main_network.get_weights())

    def egreedyPolicy(self,state,old_state,current_state):
        if (np.random.random() < self.epsilon):
            print('\nRandom action')
            action = np.random.choice(self.count_actions)
            while self.checkValidAction(old_state,current_state,action) == False:
                action = np.random.choice(self.count_actions)
            return action
        else:
            print('\nEgreedy action')
            actions = self.main_network.predict(state,verbose=0)
            print(actions)
            actionsSort = np.argsort(actions)   
            return actionsSort[0,-1] if self.checkValidAction(old_state,current_state,actionsSort[0,-1]) else actionsSort[0,-2]

    def checkValidAction(self,old_state,current_state,action):
        new_state = (0,0)
        match action:
            case 0:
                new_state = (current_state[0] - 1,current_state[1])
            case 1:
                new_state = (current_state[0],current_state[1] + 1)
            case 2:
                new_state = (current_state[0] + 1,current_state[1])
            case 3:
                new_state = (current_state[0],current_state[1] - 1)
        return new_state != old_state
    
    def createModel(self):
        model = Sequential([
            Conv2D(32, kernel_size=(4, 4),strides=2,padding='same', activation='relu', input_shape=self.size_state),
            Conv2D(64,kernel_size= (3, 3),strides=1,padding='same', activation='relu'),
            Flatten(),
            Dense(128, activation='relu', kernel_initializer=RandomUniform(minval=-0.3, maxval=0.3)),
            Dense(self.count_actions,kernel_initializer=RandomUniform(minval=-0.3, maxval=0.3), activation='linear')
        ])
        model.compile(optimizer='adam',loss='mean_squared_error',metrics='accuracy')
        print(model.summary())
        return model

    def writeWeights(self):
        save_model(self.target_network, 'Weights.keras')

    def readWeights(self):
        try:
            self.main_network = load_model('Weights.keras')
            self.target_network = self.main_network
        except Exception as e:
            print(e)

    def printInfoState(self,state,action,reward,total_reward,next_state,time_game):
        actions = {0:'↑',1:'→',2:'↓',3:'←'}
        print(f'Состояние = {state} Действие = {actions[action]} Награда = {reward} Сумм.Награда = {total_reward} След.состояние = {next_state} время вышло = {time_game}')

    def preprocessState(self,state): #Для одного состояния без батча (36,28,1)
        return np.expand_dims(state,axis = 2)
    
    def preprocessStates(self,states): #Несколько состояний (batch_size,36,28,1)
        return np.stack(states)
    
    def updateTargetNetwork(self):
        self.target_network.set_weights(self.main_network.get_weights())

    def trainNetworkStep(self,step):
        batch = self.replay_memory.sample()
        states = self.preprocessStates(batch[:,0])
        actions = np.asarray(batch[:,1],dtype=int)
        rewards = np.asarray(batch[:,2],dtype=int)
        next_state = self.preprocessStates(batch[:,3])
        is_terminals = np.asarray(batch[:,4],dtype=int)
        actions_pred = self.target_network.predict(next_state,verbose=0)
        target_Q = rewards + (1 - is_terminals) * self.gamma * np.max(actions_pred,axis=1)
        Q_values = self.main_network.predict(states,verbose=0)
        Q_values[np.arange(self.batch_size),actions] = self.alpha * target_Q
        self.main_network.fit(states,Q_values,epochs=step,verbose=1,initial_epoch=step-1)

    #Прописать функцию получения награды за выигрыш
    def train(self):
        self.readWeights()
        totals_reward = []
        is_end = False
        try:
            [print(i.shape, i.dtype) for i in self.main_network.inputs]
            iter = 0
            step = 0
            for _ in range(self.number_episods):
                self.env.render()
                is_terminal = False
                total_reward = 0
                state = self.preprocessState(self.env.getStateEnv())
                state_previus_step = self.env.getStatePacman()
                while not is_terminal:
                    iter += 1
                    old_state_pacman = self.env.getStatePacman()
                    action = self.egreedyPolicy(self.preprocessStates([state]),state_previus_step,old_state_pacman)
                    next_state, reward, is_terminal,time_game = self.env.step(action) ## Выполняем действие
                    new_state_pacman = self.env.getStatePacman()
                    state_previus_step = old_state_pacman
                    total_reward += reward
                    pre_next_state = self.preprocessState(next_state)
                    self.replay_memory.store(state,action,reward,pre_next_state,is_terminal,time_game)
                    self.printInfoState(old_state_pacman,action,reward,total_reward,new_state_pacman,time_game)
                    state = pre_next_state
                    if iter >= self.memory_size and iter % self.interval_of_training == 0:
                        step += 1
                        self.trainNetworkStep(step)
                        if iter % self.new_weights == 0:
                            self.updateTargetNetwork()
                            print('copying model to network')
                    # self.env.writeGameBoard()
                    for event in pygame.event.get():
                        if event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_q:
                                is_terminal = True
                                is_end = True
                                break
                totals_reward.append(total_reward)
                print(f'{np.mean(np.array(totals_reward))} mean reward')
                if is_end:
                    break
                self.env.newGame()
                self.env = Game(1,0)
        except Exception as e:
            print(e)
        finally:
            self.writeWeights()
            print('save model')


class ReplayMemory:
    def __init__(self, memory_size, minibatch_size):
        self.memory_size = memory_size
        self.minibatch_size = minibatch_size
        self.experience = np.array([[np.zeros((36,28),dtype=int), 0, 0, np.zeros((36,28),dtype=int), 0,0]]* memory_size,dtype=object)
        self.current_index = 0
        self.size = 0

    def store(self, state, action, reward, next_state, is_terminal, time_game):
        self.experience[self.current_index] = [state, action, reward, next_state, int(is_terminal),float(time_game)]
        self.current_index += 1
        self.size = min(self.size+1, self.memory_size)
        if self.current_index >= self.memory_size:
            self.current_index -= self.memory_size

    def sample(self):
        if self.size <  self.minibatch_size:
            return []
        indexs = np.random.randint(0,self.memory_size,size=self.minibatch_size)
        samples = self.experience[indexs]
        return samples


if __name__ == "__main__":
    agent = DQN(env = Game(1,0),number_episods=100)
    agent.train()
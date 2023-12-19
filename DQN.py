from keras.models import Sequential,load_model,save_model
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D,Dropout
from keras.losses import mean_squared_error
from keras.initializers import RandomUniform
from keras.optimizers import Adam
import numpy as np
import pygame
from Pacman import Game
import os
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class DQN:
    def __init__(self,env,gamma = 0.9, number_episods = 100, epsilon=0.1):
        self.env = env
        self.gamma = gamma
        self.number_episods = number_episods 
        self.epsilon = epsilon
        self.lr = 1e-4
        self.new_weights = 400 # к-во шагов, после к-ых веса копируются в target
        self.interval_of_training = 200 # к-во шагов после к-ых мы корректируем веса
        self.batch_size = 150 #размер батча для обучения
        self.val_batch_size = 50 #валидационная выборка
        self.memory_size = 400 # размер памяти
        self.replay_memory = ReplayMemory(self.memory_size)
        self.size_state = self.env.state_size
        self.count_actions = self.env.pacman.number_actions
        self.main_network = self.createModel()
        self.target_network = self.createModel()
        self.target_network.set_weights(self.main_network.get_weights())
        self.statictics = Statictics()

    def egreedyPolicy(self,state):
        if (np.random.random() < self.epsilon):
            action = np.random.choice(self.count_actions)
            return action
        else:
            actions = self.main_network.predict(state,verbose=0)
            action = np.argmax(actions)   
            return action

    def createModel(self):
        model = Sequential([
            Conv2D(32, kernel_size=(4, 4),strides=2,padding='valid', activation='relu', input_shape=self.size_state),
            MaxPooling2D((2,2)),
            Conv2D(64,kernel_size= (3, 3),strides=1,padding='valid', activation='relu'),
            Flatten(),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(128, activation='relu'),
            Dropout(0.2),
            Dense(self.count_actions, activation='linear')
        ])
        optimiz = Adam(learning_rate=self.lr)
        model.compile(optimizer=optimiz,loss='mean_squared_error',metrics='accuracy')
        print(model.summary())
        return model
    
    def writeWeights(self):
        save_model(self.target_network, 'DQN/Weights.keras')
    def readWeights(self):
        try:
            self.main_network = load_model('DQN/Weights.keras')
            self.target_network = self.main_network
        except Exception as e:
            print(e)

    def writeGameBoard(self,game_board):
        np.savetxt('Data/gameBoard.txt',np.asarray(game_board,dtype=int),fmt='%d',delimiter=' ')
    def printInfoState(self,state,action,reward,total_reward,next_state,time_game):
        actions = {0:'↑',1:'→',2:'↓',3:'←'}
        print(f'Состояние = {state} Действие = {actions[action]} Награда = {reward} Сумм.Награда = {total_reward} След.состояние = {next_state} время вышло = {time_game}')

    def preprocessStates(self,states): #Несколько состояний (batch_size,31,28,2)
        return np.stack(states)
    
    def updateTargetNetwork(self):
        self.target_network.set_weights(self.main_network.get_weights())

    def preBatch(self,batch):
        states = self.preprocessStates(batch[:,0])
        actions = np.asarray(batch[:,1],dtype=int)
        rewards = np.asarray(batch[:,2],dtype=int)
        eps = np.finfo(np.float32).eps.item()
        rewards = (rewards - rewards.mean())/(rewards.std() + eps)
        next_state = self.preprocessStates(batch[:,3])
        is_terminals = np.asarray(batch[:,4],dtype=int)
        actions_pred = self.target_network.predict(next_state,verbose=0)
        target_Q = rewards + (1 - is_terminals) * self.gamma * np.max(actions_pred,axis=1)
        Q_values = self.main_network.predict(states,verbose=0)
        Q_values[np.arange(batch.shape[0]),actions] = target_Q
        return (states,Q_values)
    
    def trainNetworkStep(self,step):
        batch = self.replay_memory.sample(self.batch_size,self.replay_memory.relevance_value)
        train_data = self.preBatch(batch)
        val_batch = self.replay_memory.sample(self.val_batch_size,self.replay_memory.relevance_value - self.batch_size)
        val_data = self.preBatch(val_batch)
        hist = self.main_network.fit(train_data[0],train_data[1],epochs=step,verbose=1,initial_epoch=step-1,validation_data=(val_data[0],val_data[1]) )
        self.statictics.hist_loss = np.append(self.statictics.hist_loss,hist.history['loss'][0])
        self.statictics.hist_val_loss = np.append(self.statictics.hist_val_loss,hist.history['val_loss'][0])

    #Прописать функцию получения награды за выигрыш
    def train(self):
        self.readWeights()
        self.statictics.readTotalReward()
        self.statictics.readHistLoss()
        self.statictics.readHistValLoss()
        is_end = False
        try:
            iter = 0
            step_train = 0
            for episod in range(self.number_episods):
                self.env.newLevel()
                is_terminal = False
                total_reward = 0
                state = self.env.getStateEnv()
                while not is_terminal:
                    iter += 1
                    state_pacman_before_action = self.env.getStatePacman()
                    action = self.egreedyPolicy(self.preprocessStates([state]))
                    next_state, reward, is_terminal,time_game = self.env.step(action) ## Выполняем действие
                    new_state_pacman = self.env.getStatePacman()
                    total_reward += reward
                    self.replay_memory.store(state,action,reward,next_state,is_terminal,time_game)
                    state = next_state
                    if iter >= self.memory_size and iter % self.interval_of_training == 0:
                        step_train += 1
                        self.trainNetworkStep(step_train)
                        if iter % self.new_weights == 0:
                            self.updateTargetNetwork()
                            print('copying MODEL to TARGET NETWORK')
                    for event in pygame.event.get():
                        if event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_q:
                                is_terminal = True
                                is_end = True
                                break
                self.statictics.total_rewards = np.append(self.statictics.total_rewards,total_reward)
                print(f"Награда {int(total_reward)} Средняя {int(self.statictics.total_rewards.mean())} Откл {int(self.statictics.total_rewards.std())}")
                if episod % 50 == 0:
                    self.writeWeights()
                    self.statictics.writeHistLoss()
                    self.statictics.writeHistValLoss()
                    self.statictics.writeTotalReward()
                    print("WRITING")
                if is_end:
                    break
        except Exception as e:
            print(e)
        finally:
            print(f'{np.mean(self.statictics.total_rewards)} mean reward')
            print(f'{np.max(self.statictics.total_rewards)} max reward')
            self.writeWeights()
            self.statictics.writeHistLoss()
            self.statictics.writeHistValLoss()
            self.statictics.writeTotalReward()
            print('save model')

class Statictics:
    def __init__(self):
        self.hist_loss = []
        self.hist_val_loss = []
        self.total_rewards = []
    
    def writeTotalReward(self):
        with open('DQN/total_reward.txt','w') as f:
            np.savetxt(f,self.total_rewards,fmt='%f')
    def readTotalReward(self):
        with open('DQN/total_reward.txt','r') as f:
            self.total_rewards = np.loadtxt(f,dtype=float)
    def writeHistLoss(self):
        with open('DQN/history_loss.txt','w') as f:
            np.savetxt(f,self.hist_loss,fmt='%f')
    def readHistLoss(self):
        with open('DQN/history_loss.txt','r') as f:
            self.hist_loss = np.loadtxt(f,dtype=float)
    def writeHistValLoss(self):
        with open('DQN/history_val_loss.txt','w') as f:
            np.savetxt(f,self.hist_val_loss,fmt='%f')
    def readHistValLoss(self):
        with open('DQN/history_val_loss.txt','r') as f:
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

class ReplayMemory:
    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.experience = np.array([[np.zeros((31,28),dtype=int), 0, 0, np.zeros((31,28),dtype=int), 0,0,0]]* memory_size,dtype=object)
        self.current_index = 0
        self.relevance_value = -1
        self.size = 0

    def store(self, state, action, reward, next_state, is_terminal, time_game):
        self.incrementRelevance()
        self.experience[self.current_index] = [state, action, reward, next_state, int(is_terminal),float(time_game), float(self.relevance_value)]
        self.current_index += 1
        self.size = min(self.size+1, self.memory_size)
        if self.current_index >= self.memory_size:
            self.current_index -= self.memory_size

    def incrementRelevance(self):
        self.relevance_value += 1

    def sample(self, minibatch_size, relevance_value):
        if self.size <  minibatch_size:
            return []
        relevaces = np.arange(relevance_value + 1 - minibatch_size, relevance_value + 1,1) # включительно последнее состояние 
        #indexs = np.random.randint(0,self.memory_size,size=self.minibatch_size)
        samples = self.experience[np.isin(self.experience[:,6],relevaces)]
        return samples


if __name__ == "__main__":
    agent = DQN(env = Game(1,0),number_episods=5000)
    agent.train()
    agent.statictics.oneGraph(agent.statictics.total_rewards,'step','reward','blue')
    agent.statictics.twoGraph(agent.statictics.hist_loss,agent.statictics.hist_val_loss,'step','loss','green','red')
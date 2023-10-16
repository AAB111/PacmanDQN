from keras.models import Sequential,load_model,save_model
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from keras.losses import mean_squared_error
from keras.initializers import RandomUniform
import numpy as np
import pygame
from Pacman import game
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class DQN:
    def __init__(self,env,gamma = 0.9,alpha = 0.4, number_episods = 100, epsilon=0.2):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.number_episods = number_episods 
        self.epsilon = epsilon 
        self.new_weights = 300 # к-во шагов, после к-ых веса копируются в target
        self.interval_of_training = 20 # к-во шагов после к-ых мы корректируем веса
        self.batch_size = 200 #размер батча для обучения
        self.memory_size = 1000 # размер памяти
        self.replay_memory = ReplayMemory(self.memory_size,self.batch_size)
        self.channels = 1 + 1 # 1 изображение + 1 время
        self.size_state = (self.env.number_states['row'],self.env.number_states['col'],self.channels)
        self.count_actions = self.env.pacman.number_actions
        self.main_network = self.createModel()
        self.target_network = self.createModel()
        self.target_network.set_weights(self.main_network.get_weights())
        self.FPS = 15

    def egreedyPolicy(self,state):
        if (np.random.random() < self.epsilon):
            return np.random.choice(self.count_actions)
        else:
            actions = self.main_network.predict(state,verbose=0)
            return np.argmax(actions)

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

    def printInfoState(self,state,action,reward,total_reward,next_state,dir):
        actions = {0:'↑',1:'→',2:'↓',3:'←'}
        print(f'Состояние = {state} Действие = {actions[action]} Dir = {actions[dir]} Награда = {reward} Сумм.Награда = {total_reward} След.состояние = {next_state}')

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

    def train(self):
        self.readWeights()
        clock = pygame.time.Clock()
        totals_reward = []
        is_end = False
        try:
            game.render()
            [print(i.shape, i.dtype) for i in self.main_network.inputs]
            for _ in range(self.number_episods):
                is_terminal = False
                total_reward = 0
                iter = 0
                state = self.preprocessState(self.env.getStateEnv())
                while not is_terminal:
                    iter += 1
                    clock.tick(self.FPS)
                    action = self.egreedyPolicy(self.preprocessStates([state]))
                    old_state_pacman = self.env.getStatePacman()
                    next_state, reward, is_terminal,playtime = self.env.step(action) ## Выполняем действие
                    new_state_pacman = self.env.getStatePacman()
                    total_reward += reward
                    pre_next_state = self.preprocessState(next_state)
                    self.replay_memory.store(state,action,reward,pre_next_state,is_terminal,playtime)
                    self.printInfoState(old_state_pacman,action,reward,total_reward,new_state_pacman,self.env.pacman.dir)
                    state = pre_next_state
                    step = 0
                    if iter >= self.memory_size and iter % self.interval_of_training == 0:
                        step += 1
                        self.trainNetworkStep(step)
                        if iter % self.new_weights == 0:
                            self.updateTargetNetwork()
                            print('copying model to network')
                    self.env.writeGameBoard()
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
        except Exception as e:
            print(e)
        finally:
            self.writeWeights()
            print('save model')


class ReplayMemory:
    def __init__(self, memory_size, minibatch_size):
        self.memory_size = memory_size
        self.minibatch_size = minibatch_size
        self.experience = np.array([[np.zeros((36,28),dtype=int), 0, 0, np.zeros((36,28),dtype=int), 0],0]* memory_size,dtype=object)
        self.current_index = 0
        self.size = 0

    def store(self, state, action, reward, next_state, is_terminal,playtime):
        self.experience[self.current_index] = [state, action, reward, next_state, int(is_terminal),playtime]
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
    agent = DQN(env = game,number_episods=10)
    agent.train()
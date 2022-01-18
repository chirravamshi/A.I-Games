from PROJECT.Snake import Snake
import random
import numpy as np
from keras import Sequential
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
class DQN:
    #TRAINING
    def __init__(self,env,params):
        self.action_space=env.action_space#left,right,up,down 4 actionspaces Decsreate
        self.state_space=env.state_space#
        self.epsilon=params['epsilon']
        self.gamma=params['gamma']
        self.batch_size=params['batch_size']
        self.epsilon_min=params['epsilon_min']
        self.epsilon_decay=params['epsilon_decay']
        self.learning_rate=params['learning_rate']
        self.layer_sizes=params['layer_sizes']
        self.memory=deque(maxlen=2500) #STORAGE FOR STORING PREVIOUS MOMENTS REPLAY
        self.model=self.build_model()
    def build_model(self):
        model=Sequential()
        for i in range(len(self.layer_sizes)):
            if i==0:
                model.add(Dense(self.layer_sizes[i],input_shape=(self.state_space,),activation='relu'))
            else:
                model.add(Dense(self.layer_sizes[i],activation='relu'))
        model.add(Dense(self.action_space, activation='softmax'))
        model.compile(loss='mse',optimizer=Adam(lr=self.learning_rate))
        return model
    #REPLAY MEMORY
    def remember(self,state,action,reward,next_state,done):
        self.memory.append((state,action,reward,next_state,done))
    #ESILON-GREEDY POLOCY=EXPLORATION STRATEGY
    def act(self,state):#for exploration
        if np.random.rand()<=self.epsilon:
            return random.randrange(self.action_space)
        act_values=self.model.predict(state)
        return np.argmax(act_values[0]) #PREDICTED O/P BY NUERAL N/W
    def replay(self):#storing the previous actions and results
        if len(self.memory)<self.batch_size:
            return
        #sample
        minibatch=random.sample(self.memory,self.batch_size)#RANDOMLY SAMAPLING
        states=np.array([i[0] for i in minibatch])
        actions=np.array([i[1] for i in minibatch])
        rewards=np.array([i[2] for i in minibatch])
        next_states=np.array([i[3] for i in minibatch])
        dones=np.array([i[4] for i in minibatch])
        states=np.squeeze(states)
        next_states=np.squeeze(next_states)
        #Bell Man Rule
        targets=rewards+self.gamma*(np.amax(self.model.predict_on_batch(next_states),axis=1)) * (1 - dones)
        targets_full=self.model.predict_on_batch(states)
        ind=np.array([i for i in range(self.batch_size)])
        targets_full[[ind],[actions]]=targets
        self.model.fit(states,targets_full,epochs=1,verbose=0)
        #REDUCING EPSOLON VALUE
        if self.epsilon>self.epsilon_min:
            self.epsilon*= self.epsilon_decay
#TRAINING
def train_dqn(episode,env):
    sum_of_rewards=[]
    agent=DQN(env,params)
    for e in range(episode):
        state=env.reset()
        state=np.reshape(state,(1,env.state_space))
        score=0
        max_steps=10000
        for i in range(max_steps):
            action=agent.act(state)
            prev_state=state
            #CALLING STEP FUCTION IN SNAKE AND PERFORM ACTION
            next_state,reward,done,_=env.step(action)
            score+=reward
            next_state=np.reshape(next_state,(1,env.state_space))
            #ADDING TO MEMORY
            agent.remember(state,action,reward,next_state,done)
            state=next_state
            if params['batch_size']>1:
                agent.replay()
            if done:
                print(f'final state before dying:{str(prev_state)}')
                print(f'episode:{e+1}/{episode},score:{score}')
                break
        sum_of_rewards.append(score)
    return sum_of_rewards
if __name__=='__main__':
    params=dict()
    params['name']=None
    params['epsilon']=1#EPSIOLON MAX
    params['gamma']=.95#DISCOUNT VALUE
    params['batch_size']=500
    params['epsilon_min']=.01#EPSILON MIN
    params['epsilon_decay']=.995
    params['learning_rate']=0.00025
    params['layer_sizes']=[128,128,128]
    results=dict()
    ep=50
    env_infos={'States:only walls':{'state_space':'no body knowledge'},'States: direction 0 or 1': {'state_space': ''}, 'States: coordinates': {'state_space': 'coordinates'},'States: no direction': {'state_space': 'no direction'}}
    env=Snake()
    sum_of_rewards=train_dqn(ep, env)
    results[params['name']]=sum_of_rewards
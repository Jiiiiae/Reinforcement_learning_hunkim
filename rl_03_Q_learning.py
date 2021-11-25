import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register

register(id='FrozenLake-v3',
        entry_point='gym.envs.toy_text:FrozenLakeEnv',
        kwargs={'map_name':'4x4', 'is_slippery':False})

env = gym.make('FrozenLake-v3')

#Initialize table with all zeros
Q = np.zeros([env.observation_space.n, env.action_space.n])

#Set learning parameters
num_episodes = 2000

#Discount factor
dis = 0.9

#Choose e_greedy(0) vs random noise(1)
option = 1

#Create lists to contain total rewards and steps per episode

rList = []

if option == 1:
        
    #Decaying random noise + Discounted future reward
    for i in range(num_episodes):
        #Reset environment and get first new observation
        state = env.reset()
        rAll = 0
        done = False

        while not done:
            #Choose an action by greedily (with noise) picking from Q table
            action = np.argmax(Q[state,:] + np.random.randn(1, env.action_space.n) / (i+1))

            #Get new state and reward from environment
            new_state, reward, done, _ = env.step(action)

            #Updaate Q-table with new knowledge using decay rate
            Q[state,action] = reward + dis * np.max(Q[new_state,:])

            rAll += reward
            state = new_state
        
        rList.append(rAll)

else:
    for i in range(num_episodes):
        e = 1. / ((i//100)+1)
        state = env.reset()
        rAll = 0
        done = False

        while not done:
            if np.random.rand(1) < e:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state,:])
            
            new_state, reward, done, _ = env.step(action)
            Q[state,action] = reward + dis * np.max(Q[new_state,:])

            rAll += reward
            state = new_state

        rList.append(rAll)

print("Success rate:" + str(sum(rList)/num_episodes))
print("Final Q-Table Values")
print(Q)
plt.bar(range(len(rList)), rList, color='b')
plt.show()

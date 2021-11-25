import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v1')

#Initialize table with all zeros
Q = np.zeros([env.observation_space.n, env.action_space.n])

#Set learning parameters
num_episodes = 2000

#Discount factor
dis = 0.99

#learning_rate: more bigger, more faster trainig speed
lr = .85

#Create lists to contain total rewards and steps per episode

rList = []

        
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
        Q[state,action] = (1-lr)*Q[state,action] + \
        lr*(reward + dis * np.max(Q[new_state,:]))

        rAll += reward
        state = new_state

    rList.append(rAll)

print("Success rate:" + str(sum(rList)/num_episodes))
print("Final Q-Table Values")
print(Q)
plt.bar(range(len(rList)), rList, color='b')
plt.show()



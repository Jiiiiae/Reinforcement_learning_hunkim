#!/usr/bin/env python
# coding: utf-8

# In[2]:


import gym
from gym.envs.registration import register
import sys,tty,termios


# In[3]:


class _Getch():
    def __call__(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(3)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch
inkey = _Getch()


# In[4]:


LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3


# In[8]:


arrow_keys = {
    '\x1b[A': UP,
    '\x1b[B': DOWN,
    '\x1b[C': RIGHT,
    '\x1b[D': LEFT
}


# In[14]:


register(id='FrozenLake-v3',
        entry_point='gym.envs.toy_text:FrozenLakeEnv',
        kwargs={'map_name':'4x4', 'is_slippery':False})


# In[15]:


env = gym.make('FrozenLake-v3')
state = env.reset()
env.render()


# In[16]:


while True:
    key = inkey()
    if key not in arrow_keys.keys():
        print("Game aborted!")
        break
        
    action = arrow_keys[key]
    state, reward, done, info = env.step(action)
    env.render()
    print("State: ",state, "Action: ",action,"Reward: ",reward, "Info: ",info)
    
    if done:
        print("Finished with reward",reward)
        break


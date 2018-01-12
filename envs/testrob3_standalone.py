import math
import numpy as np
import gym
from gym import error, spaces, utils
#from gym.utils import seeding

class TestRob3Env:

    def __init__(self):
        
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(np.zeros([8,8,1]), 30.*np.ones([8,8,1]))
        self.state = None
        self.reset()


    def reset(self):
        self.state = np.zeros(self.observation_space.shape)
#        self.state[4,4] = 1.
        shape = self.observation_space.shape

        # create border        
        self.state[0,:,0] = np.ones(shape[1])
        self.state[:,0,0] = np.ones(shape[0])
        self.state[shape[0]-1,:,0] = np.ones(shape[1])
        self.state[:,shape[1]-1,0] = np.ones(shape[0])

        # set agent position
        self.state[np.random.randint(shape[0]),np.random.randint(shape[1])] = 10.
        
        # set ghost position
        while True:
            ii = np.random.randint(shape[0])
            jj = np.random.randint(shape[1])
            if self.state[ii,jj] != 10.:
                self.state[ii,jj] = 20.
                break

        return np.array(self.state)
    
#    def seed(self, seed=None):
#        self.np_random, seed = seeding.np_random(seed)
#        return [seed]

    def step(self, action):
        
        done = 0
        reward = 1
        shape = self.observation_space.shape

        # Get agent and ghost positions. Zero out agent and ghost
        state = self.state
        [ii,jj,kk] = np.nonzero(state == 10.)
        [gii,gjj,gkk] = np.nonzero(state == 20.)
        state[ii,jj,kk] = 0.
        state[gii,gjj,gkk] = 0.
        
        # create border        
        self.state[0,:,0] = np.ones(shape[1])
        self.state[:,0,0] = np.ones(shape[0])
        self.state[shape[0]-1,:,0] = np.ones(shape[1])
        self.state[:,shape[1]-1,0] = np.ones(shape[0])
        
        # Create agent in new position
        if action == 0:
            ii = ii - 1
            if ii < 0:
                ii = 0
        elif action == 1:
            jj = jj + 1
            if jj > self.observation_space.shape[1] - 1:
                jj = self.observation_space.shape[1] - 1
        elif action == 2:
            ii = ii + 1
            if ii > self.observation_space.shape[0] - 1:
                ii = self.observation_space.shape[0] - 1
        elif action == 3:
            jj = jj - 1
            if jj < 0:
                jj = 0
        else:
            print("testrob._step: action out of bounds!")
        state[ii,jj,kk] = 10.

        # Create ghost in new position
        if (np.abs(ii - gii) > np.abs(jj - gjj)):
            if ii < gii:
                gii -= 1
            elif ii > gii:
                gii += 1
        else:
            if jj < gjj:
                gjj -= 1
            elif jj > gjj:
                gjj += 1

        if ((ii == gii) and (jj == gjj)):
            reward = 0
            done = 1
        state[gii,gjj,gkk] = 20.
        self.state = state
        
        if ((reward == 1) and (done == 1)):
            print("error!!!")
            
        return np.array(self.state), reward, done, {}        

    
    def render(self):
        
        print("state:")
        print(str(np.reshape(self.state,np.shape(self.state)[0:2])))

        
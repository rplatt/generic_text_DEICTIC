#
# Derived from testrob3_standalone.py
#
import math
import numpy as np
import gym
from gym import error, spaces, utils
#from gym.utils import seeding

#class MultiGhostEvade:
class StandaloneEnv:

    def __init__(self):
        
        self.action_space = spaces.Discrete(4)
#        self.observation_space = spaces.Box(np.zeros([8,8,1]), 2.*np.ones([8,8,1]))
        self.observation_space = spaces.Box(np.zeros([16,16,1]), 2.*np.ones([16,16,1]))
        self.state = None
        self.num_ghosts = 4 # you can change this parameter
        self.reset()


    def reset(self):
        self.state = np.zeros(self.observation_space.shape)
        shape = self.observation_space.shape

        # set agent position
        self.state[np.random.randint(shape[0]),np.random.randint(shape[1])] = 1.
        
        # set ghost position
        for i in range(self.num_ghosts):
            while True:
                ii = np.random.randint(shape[0])
                jj = np.random.randint(shape[1])
                if self.state[ii,jj] == 0:
                    self.state[ii,jj] = 2.
                    break

        return np.array(self.state)
    
#    def seed(self, seed=None):
#        self.np_random, seed = seeding.np_random(seed)
#        return [seed]

    def step(self, action):
        
        shape = self.observation_space.shape

        # Get agent and ghost positions. Zero out agent and ghost
        state = self.state
        agentPos = np.array(np.nonzero(state == 1.))
        ghostsPos = np.array(np.nonzero(state == 2.))
        state[agentPos[0],agentPos[1],agentPos[2]] = 0.
        for i in range(np.shape(ghostsPos)[1]):
            state[ghostsPos[0,i],ghostsPos[1,i],ghostsPos[2,i]] = 0.
        
#        # create border        
#        self.state[0,:,0] = np.ones(shape[1])
#        self.state[:,0,0] = np.ones(shape[0])
#        self.state[shape[0]-1,:,0] = np.ones(shape[1])
#        self.state[:,shape[1]-1,0] = np.ones(shape[0])
        
        # Create agent in new position
        if action == 0:
            agentPos[0] = agentPos[0] - 1
            if agentPos[0] < 0:
                agentPos[0] = 0
        elif action == 1:
            agentPos[1] = agentPos[1] + 1
            if agentPos[1] > self.observation_space.shape[1] - 1:
                agentPos[1] = self.observation_space.shape[1] - 1
        elif action == 2:
            agentPos[0] = agentPos[0] + 1
            if agentPos[0] > self.observation_space.shape[0] - 1:
                agentPos[0] = self.observation_space.shape[0] - 1
        elif action == 3:
            agentPos[1] = agentPos[1] - 1
            if agentPos[1] < 0:
                agentPos[1] = 0
        else:
            print("testrob._step: action out of bounds!")
        state[agentPos[0],agentPos[1],agentPos[2]] = 1.

        # Create ghosts in new position
        for i in range(np.shape(ghostsPos)[1]): # iterate over all ghosts
            if (np.abs(agentPos[0] - ghostsPos[0,i]) > np.abs(agentPos[1] - ghostsPos[1,i])):
                if agentPos[0] < ghostsPos[0,i]:
                    ghostsPos[0,i] -= 1
                elif agentPos[0] > ghostsPos[0,i]:
                    ghostsPos[0,i] += 1
            else:
                if agentPos[1] < ghostsPos[1,i]:
                    ghostsPos[1,i] -= 1
                elif agentPos[1] > ghostsPos[1,i]:
                    ghostsPos[1,i] += 1

        # check for termination
        done = 0
        reward = 1
        for i in range(np.shape(ghostsPos)[1]):
            if ((agentPos[0] == ghostsPos[0,i]) and (agentPos[1] == ghostsPos[1,i])):
                reward = 0
                done = 1
        
        # re-create ghosts
        for i in range(np.shape(ghostsPos)[1]):
            state[ghostsPos[0,i],ghostsPos[1,i],ghostsPos[2,i]] = 2.
        
        self.state = state
        
        if ((reward == 1) and (done == 1)):
            print("error!!!")
            
        return np.array(self.state), reward, done, {}        

    
    def render(self):
        
        print("state:")
        print(str(np.reshape(self.state,np.shape(self.state)[0:2])))

        
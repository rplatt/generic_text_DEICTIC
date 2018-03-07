#
# Derived from blockarrange1_standalone.py
#
# This version of the code gives reward of +10 when successful. OW zero reward.
#
import math
import numpy as np
import gym
from gym import error, spaces, utils
#from gym.utils import seeding

class BlockArrange:

    def __init__(self):
        
        self.maxSide = 8
        self.num_blocks = 2
        self.num_moves = self.maxSide**2
        
        # 0 -- self.maxSide**2 -> pick from specified location
        # self.maxSide**2 + 1 -- 2*self.maxSide**2 -> place at specified location
        self.action_space = spaces.Discrete(2*self.maxSide**2)

        # Observations:
        # 0: block layout
        # 1: holding (0 (nothing), or block num)
        self.observation_space = spaces.Tuple([spaces.Box(np.zeros([self.maxSide,self.maxSide,1]), 2.*np.ones([self.maxSide,self.maxSide,1])), spaces.Discrete(self.num_blocks)])
            
        self.state = None
#        self.max_episode = 50
        self.max_episode = 10
        
        self.reset()


    def reset(self):

        shape = self.observation_space.spaces[0].shape

        # Initialize state as null
        self.state = []
        
        # self.state[0] encodes block layout
        self.state.append(np.zeros(self.observation_space.spaces[0].shape))
        for i in range(self.num_blocks):
            while True:
                ii = np.random.randint(shape[0])
                jj = np.random.randint(shape[1])
                if self.state[0][ii,jj] == 0:
                    self.state[0][ii,jj] = i+1.
                    break

        # self.state[1] encodes what the robot is holding -- start out holding nothing (0)
        self.state.append(0)
        self.episode_timer = 0
        
        return np.array(self.state)
    
    
    def step(self, action):
        
        posBlocks = -np.ones([self.num_blocks,2])
        
        X,Y = np.meshgrid(range(self.maxSide),range(self.maxSide))
        coords = np.stack([np.reshape(Y,[self.maxSide**2,]), np.reshape(X,[self.maxSide**2,])],axis=0)

        # if PICK
        if action < self.num_moves:
            
            # if not already holding something
            if self.state[1] == 0:
            
                # set holding to contents of action target
                self.state[1] = np.int32(np.copy(np.squeeze(self.state[0][coords[0,action],coords[1,action]])))
                
                # zero out action target on grid
                self.state[0][coords[0,action],coords[1,action]] = 0
            
        # if PLACE
        elif action < 2*self.num_moves:
            
            action -= self.num_moves
            
            # if holding something and spot is free, then place
            if (self.state[1] != 0) and (self.state[0][coords[0,action],coords[1,action]] == 0):

                # place item
                self.state[0][coords[0,action],coords[1,action]] = self.state[1]
    
                # set holding to zero
                self.state[1] = 0
            
        else:
            print("error")

        # locate new block positions
        for i in range(self.num_blocks):
            if np.sum(self.state[0] == i+1) > 0: # if this block exists on the board
                posBlocks[i,:] = np.squeeze(np.nonzero(self.state[0] == i+1))[0:2]

        # check for termination condition
#        reward = -1
        reward = 0
        done = 0
        if posBlocks[0,0] == posBlocks[1,0]:
            if np.abs(posBlocks[0,1] - posBlocks[1,1]) <= 1:
                done = 1
#                reward = 0
                reward = 10
                
        
        if self.episode_timer > self.max_episode:
            self.episode_timer = 0
            done = 1
        self.episode_timer += 1
        
        return self.state, reward, done, {}        

    
    def render(self):
        
        print("state:")
        print(str(np.reshape(self.state,np.shape(self.state)[0:2])))

        
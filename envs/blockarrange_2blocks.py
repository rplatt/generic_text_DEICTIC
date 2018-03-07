#
# Derived from blockarrange2_rewardonsuccess_standalone.py
#
# Two blocks. Reward when they are placed horizontally adjacent.
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
#        self.observation_space = spaces.Tuple([spaces.Box(np.zeros([self.maxSide,self.maxSide,1]), self.num_blocks*np.ones([self.maxSide,self.maxSide,1])), spaces.Discrete(self.num_blocks)])
        self.observation_space = spaces.Tuple([spaces.Box(np.zeros([self.maxSide,self.maxSide,1]), np.ones([self.maxSide,self.maxSide,1])), spaces.Discrete(2)])
            
        self.state = None
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
#                    self.state[0][ii,jj] = i+1.
                    self.state[0][ii,jj] = 1
                    break

        # self.state[1] encodes what the robot is holding -- start out holding nothing (0)
        self.state.append(0)
        self.episode_timer = 0
        
        return np.array(self.state)
    
    
    def step(self, action):
        
#        posBlocks = -np.ones([self.num_blocks,2])
        
        X,Y = np.meshgrid(range(self.maxSide),range(self.maxSide))
        coords = np.stack([np.reshape(Y,[self.maxSide**2,]), np.reshape(X,[self.maxSide**2,])],axis=0)

        # if PICK
        if action < self.num_moves:
            
            # if not holding anything
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

        # check for termination condition
        reward = 0
        done = 0
        
        # locate new block positions
        blockCoords = np.nonzero(self.state[0][:,:,0])
        if np.sum(self.state[0]) == 2: # if two blocks on the board
            if blockCoords[0][0] == blockCoords[0][1]: # if two blocks at same level
                if np.abs(blockCoords[1][0] - blockCoords[1][1]) <= 1: # if two blocks horizontally adjacent
                    done = 1
                    reward = 10
                
#        # three-block adjacency condition
#        if max(posBlocks[:,0]) - min(posBlocks[:,0]) == 0:
#            if max(posBlocks[:,1]) - min(posBlocks[:,1]) <= 2:
#                done = 1
#                reward = 10
        
        if self.episode_timer > self.max_episode:
            self.episode_timer = 0
            done = 1
        self.episode_timer += 1
        
        return self.state, reward, done, {}        

    
    def render(self):
        
        print("grid:")
        print(str(self.state[0][:,:,0]))
        print("holding: " + str(self.state[1]))

        
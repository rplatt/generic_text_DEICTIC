#
# Two mnist numbers.
#
# Derived from blockarrange_3blocks.py
#
# 
#
import math
import numpy as np
import gym
from gym import error, spaces, utils
import tensorflow as tf # neet this for mnist dataset
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class NumbersArrange:

    def getBoundingBox(self,img):
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return rmin, rmax, cmin, cmax
    
    def __init__(self):
        
        mnist = tf.contrib.learn.datasets.load_dataset("mnist")
        train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
        blocksOfInterestIdx = np.nonzero(train_labels==2)[0]
        self.blocks = mnist.train.images[blocksOfInterestIdx]
        
        self.blockSideSize = np.int32(np.sqrt(np.shape(self.blocks)[1])) # edge size of one block
        self.numBlocksWide = 8 # number of blocks that can fit end-to-end on one edge of board
        
        self.gridSideSize = self.numBlocksWide*self.blockSideSize
        self.num_blocks = 2
        self.num_moves = self.numBlocksWide**2
        
        self.action_space = spaces.Discrete(2*self.gridSideSize**2)

        self.observation_space = spaces.Tuple([spaces.Box(np.zeros([self.gridSideSize,self.gridSideSize,1]), np.ones([self.gridSideSize,self.gridSideSize,1])), spaces.Discrete(2)])
        
        self.holdingImage = []
        
        self.state = None
        self.max_episode = 10
        self.numBlocksInRowGoal = 2
#        self.numBlocksInRowGoal = 3
        
        self.reset()

#        plt.imshow(img)
#        plt.show()
    
    def reset(self):

        # Initialize state as null
        self.state = []
        
        # self.state[0] encodes block layout
        self.state.append(np.zeros(self.observation_space.spaces[0].shape))
        for i in range(self.num_blocks):
            while True:
                ii, jj = np.random.randint(self.numBlocksWide,size=[2,])
                iiRange, jjRange = np.meshgrid(range(ii*self.blockSideSize,(ii+1)*self.blockSideSize),range(jj*self.blockSideSize,(jj+1)*self.blockSideSize))

                if True not in (self.state[0][iiRange,jjRange,0] > 0): # if this block empty
                    blockSel = np.random.randint(np.shape(self.blocks)[0])
                    self.state[0][iiRange,jjRange,0] = np.rot90(np.flipud(np.reshape(self.blocks[blockSel],[self.blockSideSize,self.blockSideSize])),3)
#                    self.state[0][jjRange,iiRange,0] = np.reshape(self.blocks[blockSel],[self.blockSideSize,self.blockSideSize])
                    break

        # self.state[1] encodes what the robot is holding -- start out holding nothing (0)
        self.state.append(0)
        self.episode_timer = 0
        
        return np.array(self.state)
    
    
    def step(self, action):
        
        X,Y = np.meshgrid(range(self.numBlocksWide),range(self.numBlocksWide))
        coords = np.stack([np.reshape(Y,[self.numBlocksWide**2,]), np.reshape(X,[self.numBlocksWide**2,])],axis=0)

        # if PICK
        if action < self.num_moves:
            
            # if not holding anything
            if self.state[1] == 0:
                ii = coords[0,action]
                jj = coords[1,action]
                iiRange, jjRange = np.meshgrid(range(ii*self.blockSideSize,(ii+1)*self.blockSideSize),range(jj*self.blockSideSize,(jj+1)*self.blockSideSize))
                
                if True in (self.state[0][iiRange,jjRange,0] > 0): # if this square is NOT empty
                    self.holdingImage = np.copy(self.state[0][iiRange,jjRange,0])
                    self.state[1] = 1 # set holding to contents of action target
                    self.state[0][iiRange,jjRange,0] = np.zeros([self.blockSideSize,self.blockSideSize])

            
        # if PLACE
        elif action < 2*self.num_moves:
            
            action -= self.num_moves
            
            if self.state[1] != 0:
                
                ii = coords[0,action]
                jj = coords[1,action]
                iiRange, jjRange = np.meshgrid(range(ii*self.blockSideSize,(ii+1)*self.blockSideSize),range(jj*self.blockSideSize,(jj+1)*self.blockSideSize))
                
                if True not in (self.state[0][iiRange,jjRange,0] > 0): # if this square is empty
                    self.state[0][iiRange,jjRange,0] = np.copy(self.holdingImage)
                    self.state[1] = 0 # set holding to zero

        else:
            print("error")

        # check for termination condition
        reward = 0
        done = 0
        
        # check for two adjacent blocks
        if self.state[1] == 0: # if both objs on the grid
            bounds = self.getBoundingBox(self.state[0][:,:,0])
            if (bounds[1] - bounds[0]) < (self.blockSideSize*1.5):
                if (bounds[3] - bounds[2]) < (self.blockSideSize*2.5):
                    done = 1
                    reward = 10

        if self.episode_timer > self.max_episode:
            self.episode_timer = 0
            done = 1
        self.episode_timer += 1
        
        return self.state, reward, done, {}        

    
    def render(self):
        print("grid:")
        plt.imshow(np.tile(self.state[0],[1,1,3]))
        plt.show()
#        print(str(self.state[0][:,:,0]))
#        print("holding: " + str(self.state[1]))

        
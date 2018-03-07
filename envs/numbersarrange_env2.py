#
# Two mnist numbers. This version allows for pick/place actions outside of the 8x8 grid.
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
import scipy as sp

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
        
        self.blockSize = 28 # edge size of one block
        self.stride = 28 # num grid cells between adjacent move destinations
#        self.stride = 24 # num grid cells between adjacent move destinations
#        self.stride = 20 # num grid cells between adjacent move destinations
#        self.stride = 14 # num grid cells between adjacent move destinations
#        self.stride = 7 # num grid cells between adjacent move destinations
#        self.stride = 6 # num grid cells between adjacent move destinations
#        self.stride = 3 # num grid cells between adjacent move destinations
#        self.numBlocksWide = 8 # number of blocks that can fit end-to-end on one edge of board
        self.gridSize = 28*8
        self.num_blocks = 2
#        self.num_blocks = 4
        self.observation_space = spaces.Tuple([spaces.Box(np.zeros([self.gridSize,self.gridSize,1]), np.ones([self.gridSize,self.gridSize,1])), spaces.Discrete(2)])
        self.holdingImage = []
        self.state = None
        self.max_episode = 10
        self.gap = 3 # num pixels that need to be clear around a block in order ot move it.
        self.numBlocksInRowGoal = 2

        self.reset()

    
    def reset(self):

        # reset parameters derived from stride
        self.moveCenters = range(self.blockSize/2,self.gridSize-(self.blockSize/2)+1,self.stride)
        self.num_moves = len(self.moveCenters)**2
        self.action_space = spaces.Discrete(2*self.num_moves)

        # Initialize state as null
        self.state = []
        
        halfSide = self.blockSize/2
                
        # self.state[0] encodes block layout
        self.state.append(np.zeros(self.observation_space.spaces[0].shape))
        
        for i in range(self.num_blocks):
            while True:
                
                ii = self.moveCenters[np.random.randint(len(self.moveCenters))]
                jj = self.moveCenters[np.random.randint(len(self.moveCenters))]
                iiRangeInner, jjRangeInner = np.meshgrid(range(ii-halfSide+self.gap,ii+halfSide-self.gap),range(jj-halfSide+self.gap,jj+halfSide-self.gap))
                iiRangeOuter, jjRangeOuter = np.meshgrid(range(ii-halfSide,ii+halfSide),range(jj-halfSide,jj+halfSide))
                
                if True not in (self.state[0][iiRangeOuter,iiRangeOuter,0] > 0): # if this block empty
#                    blockSel = np.random.randint(np.shape(self.blocks)[0])
                    blockSel = 55
                    img = np.rot90(np.flipud(np.reshape(self.blocks[blockSel],[self.blockSize,self.blockSize])),3)
                    imgSmall = sp.misc.imresize(img,[self.blockSize-2*self.gap,self.blockSize-2*self.gap],interp='nearest')
                    imgSmall = np.float32(imgSmall / 255.)
                    self.state[0][iiRangeInner,jjRangeInner,0] = imgSmall
#                    self.state[0][iiRangeOuter,jjRangeOuter,0] = img
                    break

        # self.state[1] encodes what the robot is holding -- start out holding nothing (0)
        self.state.append(0)
        self.episode_timer = 0
        
        return np.array(self.state)
    
    
    def step(self, action):
        
        X,Y = np.meshgrid(self.moveCenters,self.moveCenters)
        coords = np.stack([np.reshape(Y,[-1]), np.reshape(X,[-1])],axis=0)
        halfSide = self.blockSize/2

        # if PICK
        if action < self.num_moves:
            
            # if not holding anything
            if self.state[1] == 0:
                ii = coords[0,action]
                jj = coords[1,action]
                iiRangeInner, jjRangeInner = np.meshgrid(range(ii-halfSide+self.gap,ii+halfSide-self.gap),range(jj-halfSide+self.gap,jj+halfSide-self.gap))
                iiRangeOuter, jjRangeOuter = np.meshgrid(range(ii-halfSide,ii+halfSide),range(jj-halfSide,jj+halfSide))
                
                if np.sum(self.state[0][iiRangeInner,jjRangeInner,0]) > 0:
                    if (np.sum(self.state[0][iiRangeInner,jjRangeInner,0]) - np.sum(self.state[0][iiRangeOuter,jjRangeOuter,0])) == 0:
                        self.holdingImage = np.copy(self.state[0][iiRangeInner,jjRangeInner,0])
                        self.state[1] = 1 # set holding to contents of action target
                        self.state[0][iiRangeInner,jjRangeInner,0] = np.zeros([len(iiRangeInner),len(jjRangeInner)])

            
        # if PLACE
        elif action < 2*self.num_moves:
            
            action -= self.num_moves
            
            if self.state[1] != 0:
                
                ii = coords[0,action]
                jj = coords[1,action]
                iiRangeInner, jjRangeInner = np.meshgrid(range(ii-halfSide+self.gap,ii+halfSide-self.gap),range(jj-halfSide+self.gap,jj+halfSide-self.gap))
                iiRangeOuter, jjRangeOuter = np.meshgrid(range(ii-halfSide,ii+halfSide),range(jj-halfSide,jj+halfSide))
                
                if True not in (self.state[0][iiRangeOuter,jjRangeOuter,0] > 0): # if this square is empty
                    self.state[0][iiRangeInner,jjRangeInner,0] = np.copy(self.holdingImage)
                    self.state[1] = 0 # set holding to zero

        else:
            print("error")

        # check for termination condition
        reward = 0
        done = 0
        
        # check for two adjacent blocks
        if self.state[1] == 0: # if both objs on the grid
            bounds = self.getBoundingBox(self.state[0][:,:,0])
            if (bounds[1] - bounds[0]) < (self.blockSize*1.25):
                if (bounds[3] - bounds[2]) < (self.blockSize*2.0):
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

        
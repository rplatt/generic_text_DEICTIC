#
# Three disks placed in a 224x224 image. Disks placed randomly initially. 
# Reward given when the pucks are placed adjacent. Agent must learn to place
# all three pucks next to each other.
#
# Derived from puckarrange_env2.py
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

class PuckArrange:

    def getBoundingBox(self,img):
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return rmin, rmax, cmin, cmax
    
    def __init__(self,gridSizeBlocks):
        
#        mnist = tf.contrib.learn.datasets.load_dataset("mnist")
#        train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
#        blocksOfInterestIdx = np.nonzero(train_labels==2)[0]
#        self.blocks = mnist.train.images[blocksOfInterestIdx]
        
        self.blockSize = 28 # edge size of one block
        self.stride = 28 # num grid cells between adjacent move destinations
#        self.stride = 24 # num grid cells between adjacent move destinations
#        self.stride = 20 # num grid cells between adjacent move destinations
#        self.stride = 14 # num grid cells between adjacent move destinations
#        self.stride = 7 # num grid cells between adjacent move destinations
#        self.stride = 6 # num grid cells between adjacent move destinations
#        self.stride = 3 # num grid cells between adjacent move destinations
#        self.numBlocksWide = 8 # number of blocks that can fit end-to-end on one edge of board
        self.initStride = self.stride
        self.gridSize = 28*gridSizeBlocks
#        self.gridSize = 28*8
#        self.gridSize = 28*3
#        self.num_blocks = 2
        self.num_blocks = 3
        self.observation_space = spaces.Tuple([spaces.Box(np.zeros([self.gridSize,self.gridSize,1]), np.ones([self.gridSize,self.gridSize,1])), spaces.Discrete(2)])
        self.holdingImage = []
        self.state = None
#        self.max_episode = 10
        self.max_episode = 15
        self.gap = 3 # num pixels that need to be clear around a block in order ot move it.
#        self.gap = 2 # num pixels that need to be clear around a block in order ot move it.
        self.numBlocksInRowGoal = 2
#        self.blockType = 'Numerals'
        self.blockType = 'Disks'
        
        if self.blockType == 'Numerals': # load MNIST numerals
            mnist = tf.contrib.learn.datasets.load_dataset("mnist")
            train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
            blocksOfInterestIdx = np.nonzero(train_labels==2)[0]
            self.blocks = mnist.train.images[blocksOfInterestIdx]
            
        elif self.blockType == 'Disks': # load random radius disks instead
            numBlocks = 10
            minRadius = 8
#            maxRadius = 12
            maxRadius = 10
            self.blocks = []
            for i in range(numBlocks):
                X, Y = np.mgrid[0:self.blockSize,0:self.blockSize]
                halfBlock = int(self.blockSize/2)
                dist2 = (X - halfBlock) ** 2 + (Y - halfBlock) ** 2
                radius = np.random.randint(minRadius,maxRadius)
                im = np.int32(dist2 < radius**2)
                self.blocks.append(np.reshape(im,int(self.blockSize**2)))

        self.reset()

    
    def reset(self):

        # grid used to select actions
        self.moveCenters = range(int(self.blockSize/2),int(self.gridSize-(self.blockSize/2)+1),self.stride)
        self.num_moves = len(self.moveCenters)**2
        self.action_space = spaces.Discrete(2*self.num_moves)

        # grid on which objects are initially placed
#        initMoveCenters = range(int(self.blockSize/2),int(self.gridSize-(self.blockSize/2)+1),1)
#        initMoveCenters = range(int(self.blockSize/2),int(self.gridSize-(self.blockSize/2)+1),14)
        initMoveCenters = range(int(self.blockSize/2),int(self.gridSize-(self.blockSize/2)+1),self.initStride)
#        initMoveCenters = self.moveCenters
        
        # Initialize state as null
        self.state = []
        
        halfSide = self.blockSize/2
                
        # self.state[0] encodes block layout
        self.state.append(np.zeros(self.observation_space.spaces[0].shape))
        
        for i in range(self.num_blocks):
            while True:
                
                ii = initMoveCenters[np.random.randint(len(initMoveCenters))]
                jj = initMoveCenters[np.random.randint(len(initMoveCenters))]
                iiRangeOuter, jjRangeOuter = np.meshgrid(range(ii-halfSide,ii+halfSide),range(jj-halfSide,jj+halfSide))
                
                if True not in (self.state[0][iiRangeOuter,jjRangeOuter,0] > 0): # if this block empty
                    blockSel = np.random.randint(np.shape(self.blocks)[0])
                    img = np.reshape(self.blocks[blockSel],[self.blockSize,self.blockSize])
                    self.state[0][iiRangeOuter,jjRangeOuter,0] = img
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
                if (bounds[3] - bounds[2]) < (self.blockSize*3.25): # three blocks in a row
                    done = 1
                    reward = 10
            if (bounds[1] - bounds[0]) < (self.blockSize*3.25):
                if (bounds[3] - bounds[2]) < (self.blockSize*1.25): # three blocks in a row
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

        
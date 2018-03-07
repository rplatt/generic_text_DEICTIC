#
# 
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
import scipy.ndimage as ndimage

class PuckArrange:

    def getBoundingBox(self,img):
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return rmin, rmax, cmin, cmax
    
    def __init__(self):
        
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
        self.gridSize = 28*8
        self.num_blocks = 2
#        self.num_blocks = 4
        self.observation_space = spaces.Tuple([spaces.Box(np.zeros([self.gridSize,self.gridSize,1]), np.ones([self.gridSize,self.gridSize,1])), spaces.Discrete(2)])
#        self.holdingImage = []
        self.state = None
        self.max_episode = 10
        self.gap = 3 # num pixels that need to be clear around a block in order ot move it.
#        self.gap = 2 # num pixels that need to be clear around a block in order ot move it.
        self.numBlocksInRowGoal = 2
#        self.blockType = 'Numerals'
#        self.blockType = 'Disks'
        self.blockType = 'Blocks'
#        self.num_orientations = 1
#        self.num_orientations = 2
        self.num_orientations = 4 # number of orientations between 0 and pi
        
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
                dist2 = (X - halfBlock) ** 2 + (Y - halfBlock) ** 2 # disk
                radius = np.random.randint(minRadius,maxRadius)
                im = np.int32(dist2 < radius**2)
                self.blocks.append(np.reshape(im,int(self.blockSize**2)))

        elif self.blockType == 'Blocks': # load random blocks instead
            numBlocks = 10
            minRadius = 8
#            maxRadius = 12
            maxRadius = 10
            self.blocks = []
            for i in range(numBlocks):
                Y, X = np.mgrid[0:self.blockSize,0:self.blockSize]
                halfBlock = int(self.blockSize/2)
                dist2 = (Y - halfBlock) ** 2 # vertical rectangle
                randNum = np.random.rand()
#                randNum = np.random.rand() * 0.5
                if randNum < 0.25:
#                    dist2 = dist2+0.01
                    dist2 = ndimage.rotate(dist2+0.01,45,reshape=False)
                elif randNum < 0.5:
#                    dist2 = dist2+0.01
                    dist2 = ndimage.rotate(dist2+0.01,45,reshape=False)
                elif randNum < 0.75:
#                    dist2 = ndimage.rotate(dist2+0.01,90,reshape=False)
                    dist2 = ndimage.rotate(dist2+0.01,135,reshape=False)
                elif randNum < 1.01:
#                    dist2 = ndimage.rotate(dist2+0.01,90,reshape=False)
                    dist2 = ndimage.rotate(dist2+0.01,135,reshape=False)
                else:
                    print("error!")

                dist2 = dist2 + (dist2==0)*np.max(dist2)
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
        initMoveCenters = range(int(self.blockSize/2),int(self.gridSize-(self.blockSize/2)+1),self.initStride)
#        initMoveCenters = self.moveCenters
        
        # Initialize state as null
        self.state = []
        self.holdingImage = np.zeros([self.blockSize,self.blockSize-2*self.gap])
        
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
        if action < self.num_moves * self.num_orientations:
            
            position = action % self.num_moves
            orientation = action / self.num_moves
            if orientation == 0:
                orientation = 1
            if orientation == 2:
                orientation = 3
            
            # if not holding anything
            if self.state[1] == 0:
                ii = coords[0,position]
                jj = coords[1,position]

                jjRangeOuter, iiRangeOuter = np.meshgrid(range(jj-halfSide,jj+halfSide),range(ii-halfSide,ii+halfSide))
                if np.sum(self.state[0][iiRangeOuter,jjRangeOuter,0]) > 0:
                    im = self.state[0][iiRangeOuter,jjRangeOuter,0]
                    
                    if orientation == 0:
                        imRot = np.int32(im>0.5)
                    elif orientation == 1:
                        imRot = np.int32(ndimage.rotate(im, 45, reshape=False)>0.5)
                    elif orientation == 2:
                        imRot = np.int32(ndimage.rotate(im, 90, reshape=False)>0.5)
                    elif orientation == 3:
                        imRot = np.int32(ndimage.rotate(im, 135, reshape=False)>0.5)
                    else:
                        print("error in orientation 1")
                    
#                    if orientation == 1:
##                        imRot = np.int32(ndimage.rotate(im, 90, reshape=False)>0.5)
#                        imRot = np.int32(ndimage.rotate(im, 45, reshape=False)>0.5)
#                    else:
#                        imRot = np.int32(im>0.5)
                        
#                    imRot = im # comment out this line to use rotation
                    shape = np.shape(imRot)
                    jjGap, iiGap = np.meshgrid(np.r_[range(0,self.gap), range(shape[0]-self.gap,shape[0])], range(0,shape[0]))
                    if np.sum(imRot[iiGap,jjGap]) < 1:
                        self.holdingImage = imRot
                        self.state[1] = 1 # set holding to contents of action target
                        self.state[0][iiRangeOuter,jjRangeOuter,0] = np.zeros([np.shape(iiRangeOuter)[0],np.shape(iiRangeOuter)[1]])

        # if PLACE
        elif action < 2 * self.num_moves * self.num_orientations:
            
            action -= self.num_moves * self.num_orientations
            position = action % self.num_moves
            orientation = action / self.num_moves
            if orientation == 0:
                orientation = 1
            if orientation == 2:
                orientation = 3
            
            if self.state[1] != 0:
                
                ii = coords[0,position]
                jj = coords[1,position]
                
                jjRangeOuter, iiRangeOuter = np.meshgrid(range(jj-halfSide,jj+halfSide),range(ii-halfSide,ii+halfSide))
                
                if True not in (self.state[0][iiRangeOuter,jjRangeOuter,0] > 0): # if this square is empty
                    im = self.holdingImage
                    if orientation == 0:
                        imRot = im
                    elif orientation == 1:
                        imRot = ndimage.rotate(im, -45, reshape=False)
                    elif orientation == 2:
                        imRot = ndimage.rotate(im, -90, reshape=False)
                    elif orientation == 3:
                        imRot = ndimage.rotate(im, -135, reshape=False)
                    else:
                        print("error in orientation 2")
                        
#                    imRot = im # comment out this line to use rotation
                    self.state[0][iiRangeOuter,jjRangeOuter,0] = imRot
#                    self.state[0][jjRangeOuter,iiRangeOuter,0] = imRot
                    self.state[1] = 0 # set holding to zero

        else:
            print("error")

        # check for termination condition
        reward = 0
        done = 0
        
        # check for two adjacent blocks
        if self.state[1] == 0: # if both objs on the grid
            
#            for ii in range(2):
            for ii in range(4):
                if ii == 0:
                    im = self.state[0][:,:,0]
                elif ii == 1:
                    im = ndimage.rotate(self.state[0][:,:,0],45,reshape=True)
                elif ii == 2:
                    im = ndimage.rotate(self.state[0][:,:,0],90,reshape=True)
                elif ii == 3:
                    im = ndimage.rotate(self.state[0][:,:,0],135,reshape=True)
                else:
                    print("error in orientation 3")
            
                bounds = self.getBoundingBox(np.int32(im>0.5))
                
                # Check for horizontal adjacency
                if (bounds[1] - bounds[0]) < (self.blockSize*0.9): # require two blocks to be aligned and flat
                    if (bounds[3] - bounds[2]) < (self.blockSize*3.0):
                        done = 1
                        reward = 10
                        break
                    
                # Check for vertical adjacency
                if (bounds[1] - bounds[0]) < (self.blockSize*3.0):
                    if (bounds[3] - bounds[2]) < (self.blockSize*0.9): # require two blocks to be aligned and flat
                        done = 1
                        reward = 10
                        break

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

        
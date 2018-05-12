#
# 
#
# Derived from puckarrange_env15.py
#
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
        
        self.observation_space = spaces.Tuple([spaces.Box(np.zeros([self.gridSize,self.gridSize,1]), np.ones([self.gridSize,self.gridSize,1])), spaces.Discrete(2)])
        self.holdingImage = []
        self.state = None
        self.max_episode = 10
        self.gap = 3 # num pixels that need to be clear around a block in order ot move it.
#        self.gap = 2 # num pixels that need to be clear around a block in order ot move it.
        self.numBlocksInRowGoal = 2

#        self.num_orientations = 4
#        self.blockType = 'Disks'
#        self.blockType = 'Blocks'
        
#        self.reset()
        
    def reset(self):

        if self.num_orientations == 2:
            self.orientationVector = [0,90]
#            self.orientationVector = [45,135]
        elif self.num_orientations == 4:
            self.orientationVector = [0,45,90,135]
        elif self.num_orientations == 8:
            self.orientationVector = [0,22.5,45,67.5,90,112.5,135,157.5]
        elif self.num_orientations == 16:
            self.orientationVector = [0,11.25,22.5,33.75,45,56.25,67.5,78.75,90,101.25,112.5,123.75,135,146.25,157.5,168.75]
        else:
            print('ERROR: wrong number of input orientations')

        if self.blockType == 'Numerals': # load MNIST numerals
            mnist = tf.contrib.learn.datasets.load_dataset("mnist")
            train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
            blocksOfInterestIdx = np.nonzero(train_labels==2)[0]
            self.blocks = mnist.train.images[blocksOfInterestIdx]
            
        elif self.blockType == 'Disks': # load random radius disks instead
            numBlocks = 10
            minRadius = 8
            maxRadius = 10
            self.blocks = []
            for i in range(numBlocks):
                X, Y = np.mgrid[0:self.blockSize,0:self.blockSize]
                halfBlock = int(self.blockSize/2)
                dist2 = (X - halfBlock) ** 2 + (Y - halfBlock) ** 2
                radius = np.random.randint(minRadius,maxRadius)
                im = np.int32(dist2 < radius**2)
                self.blocks.append(np.reshape(im,int(self.blockSize**2)))

        elif self.blockType == 'Blocks': # load random blocks instead
            numBlocks = 10
            minRadius = 8
            maxRadius = 10
            self.blocks = []
            for i in range(numBlocks):
                Y, X = np.mgrid[0:self.blockSize,0:self.blockSize]
                halfBlock = int(self.blockSize/2)
                dist2 = (Y - halfBlock) ** 2 # vertical rectangle

                randNum = np.random.randint(self.num_orientations)
                dist2 = ndimage.rotate(dist2+0.01,self.orientationVector[randNum],reshape=False)
                
#                randNum = np.random.rand()
#                if randNum < 0.25:
#                    dist2 = dist2+0.01
#                elif randNum < 0.5:
#                    dist2 = ndimage.rotate(dist2+0.01,45,reshape=False)
#                elif randNum < 0.75:
#                    dist2 = ndimage.rotate(dist2+0.01,90,reshape=False)
#                elif randNum < 1.01:
#                    dist2 = ndimage.rotate(dist2+0.01,135,reshape=False)
#                else:
#                    print("error!")

                dist2 = dist2 + (dist2==0)*np.max(dist2)
                radius = np.random.randint(minRadius,maxRadius)
                im = np.int32(dist2 < radius**2)

                # Create a boundary of zeros around the object. Needed to break
                # up adjacent objects.
                im[0,:] = 0
                im[:,0] = 0
                im[-1,:] = 0
                im[:,-1] = 0
                
                self.blocks.append(np.reshape(im,int(self.blockSize**2)))
        else:
            print('ERROR: no blockType defined.')

    
#    def reset(self):

        # grid used to select actions
        self.moveCenters = range(int(self.blockSize/2),int(self.gridSize-(self.blockSize/2)+1),self.stride)
        self.num_moves = len(self.moveCenters)**2
        self.action_space = spaces.Discrete(2*self.num_moves)

        # grid on which objects are initially placed
        initMoveCenters = range(int(self.blockSize/2),int(self.gridSize-(self.blockSize/2)+1),self.initStride)
        
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

        position = action % self.num_moves
        pickplace = action / (self.num_moves * self.num_orientations)
        orientation = (action - pickplace * self.num_moves * self.num_orientations) / self.num_moves
#        orientation = action / self.num_moves # DEBUG
        
        # if PICK
        if pickplace == 0:
            
            # if not holding anything
            if self.state[1] == 0:
                ii = coords[0,position]
                jj = coords[1,position]
                jjRangeInner, iiRangeInner = np.meshgrid(range(jj-halfSide+self.gap,jj+halfSide-self.gap),range(ii-halfSide+1,ii+halfSide-1))
                jjRangeOuter, iiRangeOuter = np.meshgrid(range(jj-halfSide,jj+halfSide),range(ii-halfSide,ii+halfSide))

                # if there's something in this spot
                if np.sum(self.state[0][iiRangeOuter,jjRangeOuter,0]) > 0:
                    im = self.state[0][iiRangeOuter,jjRangeOuter,0]

                    # if it's a separate object
                    shape = np.shape(im)
                    jjGap1, iiGap1 = np.meshgrid(range(0,shape[1]), np.r_[range(0,1), range(shape[1]-1,shape[1])])
                    jjGap2, iiGap2 = np.meshgrid(np.r_[range(0,1), range(shape[1]-1,shape[1])],range(0,shape[1]))
                    if (np.sum(im[iiGap1,jjGap1]) < 1) and (np.sum(im[iiGap2,jjGap2]) < 1):

                        # rotate image
                        if orientation >= self.num_orientations:
                            print('ERROR: bad orientation in step')
                        imRot = np.int32(ndimage.rotate(im, self.orientationVector[orientation], reshape=False)>0.5)
                        
#                        if orientation == 0:
#                            imRot = np.int32(im>0.5)
#                        elif orientation == 1:
#                            imRot = np.int32(ndimage.rotate(im, 45, reshape=False)>0.5)
#                        elif orientation == 2:
#                            imRot = np.int32(ndimage.rotate(im, 90, reshape=False)>0.5)
#                        elif orientation == 3:
#                            imRot = np.int32(ndimage.rotate(im, 135, reshape=False)>0.5)
#                        else:
#                            print("error in orientation 1")

                        # if there's a gap for the fingers
                        shape = np.shape(imRot)
                        jjGap, iiGap = np.meshgrid(np.r_[range(0,self.gap), range(shape[1]-self.gap,shape[1])], range(0,shape[0]))
                        if np.sum(imRot[iiGap,jjGap]) < 1:

                            self.holdingImage = imRot
                            self.state[1] = 1 # set holding to contents of action target
                            self.state[0][iiRangeOuter,jjRangeOuter,0] = np.zeros([np.shape(iiRangeOuter)[0],np.shape(jjRangeOuter)[1]])

        # if PLACE
        elif pickplace == 1:
            
            if self.state[1] != 0:
                
                ii = coords[0,position]
                jj = coords[1,position]
                jjRangeOuter, iiRangeOuter = np.meshgrid(range(jj-halfSide,jj+halfSide),range(ii-halfSide,ii+halfSide))

                if True not in (self.state[0][iiRangeOuter,jjRangeOuter,0] > 0): # if this square is empty
                    
                    im = self.holdingImage
                    
                    # rotate image
                    if orientation >= self.num_orientations:
                        print('ERROR: bad orientation in step')
                    imRot = np.int32(ndimage.rotate(im, -self.orientationVector[orientation], reshape=False)>0.5)

#                    if orientation == 0:
#                        imRot = im
#                    elif orientation == 1:
#                        imRot = ndimage.rotate(im, -45, reshape=False)
#                    elif orientation == 2:
#                        imRot = ndimage.rotate(im, -90, reshape=False)
#                    elif orientation == 3:
#                        imRot = ndimage.rotate(im, -135, reshape=False)
#                    else:
#                        print("error in orientation 2")
                    
                    self.state[0][iiRangeOuter,jjRangeOuter,0] = imRot
                    self.state[1] = 0 # set holding to zero

        else:
            print("error: action out of bounds")

        # check for termination condition
        reward = 0
        done = 0
        
        allowedGapFraction = 0.8 # percent of total block size that fits between fingers

        if self.state[1] == 0: # if both objs on the grid

            for ii in range(self.num_orientations):
                
                im = np.int32(ndimage.rotate(self.state[0][:,:,0],self.orientationVector[ii],reshape=True)>0.5)
                bounds = self.getBoundingBox(im)            
                if (bounds[1] - bounds[0]) < (self.blockSize*allowedGapFraction):
                    if (bounds[3] - bounds[2]) < (self.blockSize*3.0):
                        done = 1
                        reward = 10
                        break
                if (bounds[1] - bounds[0]) < (self.blockSize*3.0):
                    if (bounds[3] - bounds[2]) < (self.blockSize*allowedGapFraction):
                        done = 1
                        reward = 10
                        break

        
#        # check for two horizontal, vertical, or diagonally adjacent blocks
#        if self.state[1] == 0: # if both objs on the grid
#            bounds = self.getBoundingBox(self.state[0][:,:,0])
#            
#            if (bounds[1] - bounds[0]) < (self.blockSize*0.7):
#                if (bounds[3] - bounds[2]) < (self.blockSize*3.0):
#                    done = 1
#                    reward = 10
#                    
#            if (bounds[1] - bounds[0]) < (self.blockSize*3.0):
#                if (bounds[3] - bounds[2]) < (self.blockSize*0.7):
#                    done = 1
#                    reward = 10
#                    
#            im = np.int32(ndimage.rotate(self.state[0][:,:,0],45,reshape=True)>0.5)            
#            bounds = self.getBoundingBox(im)
#            
#            if (bounds[1] - bounds[0]) < (self.blockSize*0.9):
#                if (bounds[3] - bounds[2]) < (self.blockSize*3.0):
#                    done = 1
#                    reward = 10
#                    
#            if (bounds[1] - bounds[0]) < (self.blockSize*3.0):
#                if (bounds[3] - bounds[2]) < (self.blockSize*0.9):
#                    done = 1
#                    reward = 10
            

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

        
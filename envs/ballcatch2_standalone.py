import math
import numpy as np
import gym
from gym import error, spaces, utils
#from gym.utils import seeding

class StandaloneEnv:

    def __init__(self):
        
        self.action_space = spaces.Discrete(3) # 0: no change, 1: left, 2: right
        self.observation_space = spaces.Box(np.zeros([8,8,1]), 2.*np.ones([8,8,1]))
        self.state = None
        self.reset()


    def reset(self):
        self.state = np.zeros(self.observation_space.shape)
        shape = self.observation_space.shape

        # set agent position on bottom row
        self.state[shape[0]-1,np.random.randint(shape[1])] = 1.

        # set ball position on top row
        self.state[0,np.random.randint(shape[1])] = 2.
        
        return np.array(self.state)
    
#    def seed(self, seed=None):
#        self.np_random, seed = seeding.np_random(seed)
#        return [seed]

    def step(self, action):
        
        shape = self.observation_space.shape

        # Get agent and ball positions. Zero out agent and ball
        state = self.state
        [ii,jj,kk] = np.nonzero(state == 1.)
        [gii,gjj,gkk] = np.nonzero(state == 2.)
        state[ii,jj,kk] = 0.
        state[gii,gjj,gkk] = 0.
        
        # Create agent in new position. action==0 for no motion
        if action == 1: # go left
            jj = jj - 1
            if jj < 0:
                jj = 0
        elif action == 2: # go right
            jj = jj + 1
            if jj > self.observation_space.shape[1] - 1:
                jj = self.observation_space.shape[1] - 1
        elif action != 0:
            print("testrob.step: action out of bounds!")
        state[ii,jj,kk] = 1.

        # Create ball in new position
        gii += 1
#        self.slopecount += 1
#        if self.slopecount > self.slopeofmotion:
#            if self.directionofmotion == 0: # go left
#                gjj -= 1
#            elif self.directionofmotion == 1: # go right
#                gjj += 1
#            else:
#                print("BallCatch.step: error! Slopeofmotion out of bounds!")
#            if gjj < 0:
#                gjj = 0
#                self.directionofmotion == 1
#            if gjj > self.observation_space.shape[1] - 1:
#                gjj = self.observation_space.shape[1] - 1
#                self.directionofmotion == 0
                
        state[gii,gjj,gkk] = 2.
        
        # if ball on bottom row, +1 reward for catch, -1 reward OW
        done = 0
        reward = 0
        if gii == self.observation_space.shape[0] - 1:
            done = 1
            if jj == gjj:
                reward = +1
            else:
                reward = -1
        
        return np.array(self.state), reward, done, {}

    
    def render(self):
        
        print("state:")
        print(str(np.reshape(self.state,np.shape(self.state)[0:2])))

        
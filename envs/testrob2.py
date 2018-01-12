#
# get-to-the-goal gridworld. reward = -1 on all timesteps except the last.
#
import math
import numpy as np   
import gym
from gym import error, spaces, utils
from gym.utils import seeding


class TestRob2Env(gym.Env):
    metadata = {'render.modes': ['human']}

#    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self):
        
#        self.observation_space = spaces.Box(np.zeros([8,8]), np.ones([8,8]))

        self.action_space = spaces.Discrete(4)
#        self.observation_space = spaces.Box(np.array([0.,0.]), np.array([1.,1.]))
        self.observation_space = spaces.Box(np.zeros([8,8,1]), np.ones([8,8,1]))
        self.state = None
        self.reset()

#        self.min_position = -1.2
#        self.max_position = 0.6
#        self.max_speed = 0.07
#        self.goal_position = 0.5
#
#        self.low = np.array([self.min_position, -self.max_speed])
#        self.high = np.array([self.max_position, self.max_speed])
#
#        self.viewer = None
#
#        self.action_space = spaces.Discrete(3)
#        self.observation_space = spaces.Box(self.low, self.high)
#
#        self._seed()
#        self.reset()

    def _reset(self):
        self.state = np.zeros(self.observation_space.shape)
#        self.state[4,4] = 1.
        shape = self.observation_space.shape
        self.state[np.random.randint(shape[0]),np.random.randint(shape[1])] = 1.
#        self.state = np.array([self.np_random.uniform(low=-0.6, high=-0.4), 0])
#        self.state = np.array([1.,2.])
        return np.array(self.state)
    
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        
        state = self.state
        [ii,jj,kk] = np.nonzero(state>0.5) # find agent
        state[ii,jj,kk] = 0.
        
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
            print("testrob._step: error!")
        
        state[ii,jj,kk] = 1.
        self.state = state

        done = 0
        reward = -1
        if (ii == self.observation_space.shape[0] - 1) and (jj == self.observation_space.shape[1] - 1):
            done = 1
            reward = 0

        return np.array(self.state), reward, done, {}        
#        return self.state
    
    def _render(self, mode='human', close=False):
        
        if close:
            return
        
        if mode == 'human':
            print("state:")
            print(str(np.reshape(self.state,np.shape(self.state)[0:2])))
#            print("state: " + str(self.state))
        
                  
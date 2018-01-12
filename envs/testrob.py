#
# 2D environment
#
import math
import numpy as np   
import gym
from gym import error, spaces, utils
from gym.utils import seeding


class TestRobEnv(gym.Env):
#    metadata = {'render.modes': ['human']}

    def __init__(self):
        
#        self.observation_space = spaces.Box(np.zeros([8,8]), np.ones([8,8]))

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(np.array([0.,0.]), np.array([1.,1.]))
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
#        self.state = np.zeros(env.observation_space.shape)
#        self.state = np.array([self.np_random.uniform(low=-0.6, high=-0.4), 0])
        self.state = np.array([1.,2.])
        return np.array(self.state)
    
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        
        state = self.state
        reward = -1
        
        done = 0
#        if action == 0:
#            state[0] = state[0] - 1
#        elif action == 1:
#            state[1] = state[1] + 1
#        elif action == 2:
#            state[0] = state[0] + 1
#        elif action == 3:
#            state[1] = state[1] - 1
#        else:
#            print("testrob._step: error!")
        

        return np.array(self.state), reward, done, {}        
#        return self.state
    
    def _render(self, mode='human', close=False):
        print("state: " + str(self.state))
        
                  
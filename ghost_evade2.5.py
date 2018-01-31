#
# Simple tabular version of ghost evade using the avg-value deictic method.
# This method should converge to approx avg 60 reward
#
import gym
import numpy as np
#import tensorflow as tf
#import tf_util_rob as U
#import models as models
#import build_graph as build_graph
from replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from schedules import LinearSchedule
#import tempfile
import matplotlib.pyplot as plt
import copy

# I had problems w/ the gym environment, so here I made my own standalone class
#import envs.frozen_lake
import envs.testrob3_standalone as envstandalone

def main():


    env = envstandalone.TestRob3Env()
    
    max_timesteps=40000
    buffer_size=50000
    exploration_fraction=0.2
    exploration_final_eps=0.02
    print_freq=10
    learning_starts=1000
    gamma=.98
    target_network_update_freq=500
    learning_alpha = 0.2
    
    batch_size=64
    train_freq=2

    deicticShape = (3,3,1)
    num_deictic_patches=36

    num_actions = 4
    episode_rewards = [0.0]

    # Create the schedule for exploration starting from 1.
    exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * max_timesteps),
                                 initial_p=1.0,
                                 final_p=exploration_final_eps)

    # same as getDeictic except this one just calculates for the observation
    # input: n x n x channels
    # output: dn x dn x channels
    def getDeicticObs(obs):
        windowLen = deicticShape[0]
        deicticObs = []
        for i in range(np.shape(obs)[0] - windowLen + 1):
            for j in range(np.shape(obs)[1] - windowLen + 1):
                deicticObs.append(obs[i:i+windowLen,j:j+windowLen,:])
        return np.array(deicticObs)

    # input: batch x nxnx1 tensor of observations
    def convertState(observations):
        shape = np.shape(observations)
        observations_small = np.squeeze(observations)
        agent_pos = np.nonzero(observations_small == 10)
        ghost_pos = np.nonzero(observations_small == 20)
        state_numeric = 3*np.ones((4,shape[0]))
        state_numeric[0,agent_pos[0]] = agent_pos[1]
        state_numeric[1,agent_pos[0]] = agent_pos[2]
        state_numeric[2,ghost_pos[0]] = ghost_pos[1]
        state_numeric[3,ghost_pos[0]] = ghost_pos[2]
        return np.int32(state_numeric)

    tabularQ = 100*np.ones([deicticShape[0]+1,deicticShape[1]+1,deicticShape[0]+1,deicticShape[1]+1, num_actions])
    
    obs = env.reset()

    for t in range(max_timesteps):

        # get current q-values
        obsDeictic = getDeicticObs(obs)
        stateCurr = convertState(obsDeictic)
        qCurr = tabularQ[stateCurr[0], stateCurr[1], stateCurr[2], stateCurr[3],:]
        
        # select action
        action = np.argmax(np.max(qCurr,0))
        selPatch = np.argmax(np.max(qCurr,1))
        if np.random.rand() < exploration.value(t):
            action = np.random.randint(env.action_space.n)

        # take action
        new_obs, rew, done, _ = env.step(action)
            
        # get next q-values
        stateNext = convertState(getDeicticObs(new_obs))
        qNext1 = tabularQ[stateNext[0], stateNext[1], stateNext[2], stateNext[3],:]
        
        # perform learning update
        qNextmaxa = np.max(qNext1,1) # this deictic max seems to work better on this problem. why?
#        qNextmaxa = np.max(qNext1) # this is the correct deictic max
        targets = rew + (1-done) * gamma * qNextmaxa
        
        tabularQ[stateCurr[0], stateCurr[1], stateCurr[2], stateCurr[3],action] = \
            (1 - learning_alpha) * tabularQ[stateCurr[0], stateCurr[1], stateCurr[2], stateCurr[3],action] \
            + learning_alpha * targets

        # bookkeeping for storing episode rewards
        episode_rewards[-1] += rew
        if done:
            new_obs = env.reset()
            episode_rewards.append(0.0)
            
        mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
        num_episodes = len(episode_rewards)
        if done and print_freq is not None and len(episode_rewards) % print_freq == 0:
            print("steps: " + str(t) + ", episodes: " + str(num_episodes) + ", mean 100 episode reward: " + str(mean_100ep_reward) + ", % time spent exploring: " + str(int(100 * exploration.value(t))) + ", max q at curr state: " + str(np.max(qCurr)))
        
        obs = new_obs
 

if __name__ == '__main__':
    main()


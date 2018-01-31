#
# Tabular Q-learning w/ no replay buffer.
#
# You should try to get your code working with this simple version before scaling up
# to rob_tabular_replay.py and then to rob_dqn_replay.py
#
#
#
import gym
import numpy as np
import time as time
import tensorflow as tf
import tf_util_rob as U
import models as models
from replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from schedules import LinearSchedule

import envs.frozen_lake


def main():
    
    np.set_printoptions(formatter={'float_kind':lambda x: "%.2f" % x})
    
#    env = gym.make("FrozenLake-v0")
    env = gym.make("FrozenLake8x8-v0")

    # Dictionary-based value function
    q_func_tabular = {}
    defaultQValue = np.ones(env.action_space.n)
    
    # Given an integer, return the corresponding boolean array 
    def getBoolBits(state):
        return [np.unpackbits(np.uint8(state))==1]
    
    # cols of vectorKey must be boolean less than 64 bits long
    def getTabularKeys(vectorKey):
        obsBits = np.packbits(vectorKey,1)
        obsKeys = 0
        for i in range(np.shape(obsBits)[1]):
            # IMPORTANT: the number of bits in the type cast below (UINT64) must be at least as big
            # as the bits required to encode obsBits. If it is too small, we get hash collisions...
            obsKeys = obsKeys + (256**i) * np.uint64(obsBits[:,i])
        return obsKeys
    
    def getTabular(vectorKey):
        keys = getTabularKeys(vectorKey)
        return np.array([q_func_tabular[x] if x in q_func_tabular else defaultQValue for x in keys])
    
#    def trainTabular(vectorKey,qCurrTargets,weights):
    def trainTabular(vectorKey,qCurrTargets):
        keys = getTabularKeys(vectorKey)
        alpha=0.1
        for i in range(len(keys)):
            if keys[i] in q_func_tabular:
                q_func_tabular[keys[i]] = (1-alpha)*q_func_tabular[keys[i]] + alpha*qCurrTargets[i]
#                q_func_tabular[keys[i]] = q_func_tabular[keys[i]] + alpha*weights[i,:]*(qCurrTargets[i] - q_func_tabular[keys[i]]) # (1-alpha)*q_func[keys[i]] + alpha*qCurrTargets[i]
            else:
                q_func_tabular[keys[i]] = qCurrTargets[i]



    max_timesteps=200000
    exploration_fraction=0.3
    exploration_final_eps=0.02
    print_freq=1
    gamma=.98
    num_cpu = 16
        
    episode_rewards = [0.0]
    
    # Create the schedule for exploration starting from 1.
    exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * max_timesteps),
                                 initial_p=1.0,
                                 final_p=exploration_final_eps)

    def make_obs_ph(name):
        return U.BatchInput(env.observation_space.spaces[0].shape, name=name)
    
    sess = U.make_session(num_cpu)
    sess.__enter__()

    state = env.reset()

    episode_rewards = [0.0]
    timerStart = time.time()
    for t in range(max_timesteps):
        
        
        qCurr = getTabular(getBoolBits(state))

        qCurrNoise = qCurr + np.random.random(np.shape(qCurr))*0.01 # add small amount of noise to break ties randomly

        # select action at random
        action = np.argmax(qCurrNoise)
        if np.random.rand() < exploration.value(t):
            action = np.random.randint(env.action_space.n)
        
        # take action
        nextState, rew, done, _ = env.step(action)

        qNext = getTabular(getBoolBits(nextState))

        # Calculate TD target
        qNextmax = np.max(qNext)
        target = rew + (1-done) * gamma * qNextmax


        # Update value function
        qCurrTarget = qCurr
        qCurrTarget[0][action] = target
        trainTabular(getBoolBits(state),qCurrTarget)


        # bookkeeping for storing episode rewards
        episode_rewards[-1] += rew
        if done:
            new_obs = env.reset()
            episode_rewards.append(0.0)
        mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
        num_episodes = len(episode_rewards)
        if done and print_freq is not None and len(episode_rewards) % print_freq == 0:
            timerFinal = time.time()
            print("steps: " + str(t) + ", episodes: " + str(num_episodes) + ", mean 100 episode reward: " + str(mean_100ep_reward) + ", % time spent exploring: " + str(int(100 * exploration.value(t))) + ", time elapsed: " + str(timerFinal - timerStart))
            timerStart = timerFinal
        
        state = np.copy(nextState)




if __name__ == '__main__':
    main()


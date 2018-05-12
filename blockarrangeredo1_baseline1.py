#
# I'm trying to test a statdard deep-q version of this problem as a baseline.
#
# Adapted from blockarrangeredo1.py
#
# Results: this is the first baseline that works. Very sensitive to the size
#          of the grid that we run on. Runs quickly for a 3x3 grid...
#
import gym
import numpy as np
import time as time
import tensorflow as tf
import tf_util_rob as U
import models as models
from replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from schedules import LinearSchedule
import pickle

#import envs.blockarrange3_rewardonsuccess_standalone as envstandalone
import envs.blockarrange_2blocks_baseline as envstandalone

# **** Make tensorflow functions ****


def main():
    
    np.set_printoptions(formatter={'float_kind':lambda x: "%.2f" % x})

    # Dictionary-based value function
    q_func_tabular = {}

#    filehandle = open("saved.pkl","rb")
#    q_func_tabular = pickle.load(filehandle)
#    filehandle.close()

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
#        return np.array([q_func_tabular[x] if x in q_func_tabular else 10*np.ones(num_states) for x in keys])
        return np.array([q_func_tabular[x] if x in q_func_tabular else 10*np.ones(num_actions) for x in keys])
    
#    def trainTabular(vectorKey,qCurrTargets,weights):
    def trainTabular(vectorKey,qCurrTargets):
        keys = getTabularKeys(vectorKey)
        alpha=0.2
#        alpha=0.05
        for i in range(len(keys)):
            if keys[i] in q_func_tabular:
                q_func_tabular[keys[i]] = (1-alpha)*q_func_tabular[keys[i]] + alpha*qCurrTargets[i]
#                q_func_tabular[keys[i]] = q_func_tabular[keys[i]] + alpha*weights[i,:]*(qCurrTargets[i] - q_func_tabular[keys[i]]) # (1-alpha)*q_func[keys[i]] + alpha*qCurrTargets[i]
            else:
                q_func_tabular[keys[i]] = qCurrTargets[i]


    env = envstandalone.BlockArrange()

    max_timesteps=50000
    exploration_fraction=0.3
    exploration_final_eps=0.1
    print_freq=1
    gamma=.90
    num_cpu = 16

    # first two elts of deicticShape must be odd
#    actionShape = (3,3,2)
#    num_states = 2 # either holding or not
    num_patches = env.maxSide**2
    num_actions = 2*num_patches
    num_actions_discrete = 2
    valueFunctionType = "TABULAR"
    actionSelectionStrategy = "UNIFORM_RANDOM" # actions are selected randomly from collection of all actions

    episode_rewards = [0.0]
    
    # Create the schedule for exploration starting from 1.
    exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * max_timesteps),
                                 initial_p=1.0,
                                 final_p=exploration_final_eps)

    def make_obs_ph(name):
        return U.BatchInput(env.observation_space.spaces[0].shape, name=name)

#    getMoveActionDescriptors = build_getMoveActionDescriptors(make_obs_ph=make_obs_ph,deicticShape=actionShape)
    
    sess = U.make_session(num_cpu)
    sess.__enter__()

    obs = env.reset()

    episode_rewards = [0.0]
    timerStart = time.time()
    for t in range(max_timesteps):
        
        # FULLSTATE representation
        stateDescriptorsFlat = np.reshape(obs[0],[-1,env.maxSide**2]) == 1
        stateDescriptorsFlat = np.array([np.concatenate([[obs[1]==1],stateDescriptorsFlat[0]])])
        qCurr = getTabular(stateDescriptorsFlat)[0]
        
        qCurrNoise = qCurr + np.random.random(np.shape(qCurr))*0.01 # add small amount of noise to break ties randomly

        # select action at random
        action = np.argmax(qCurrNoise)
        if np.random.rand() < exploration.value(t):
            action = np.random.randint(num_actions)

        # take action
        new_obs, rew, done, _ = env.step(action)

        # FULLSTATE representation
        stateDescriptorsNextFlat = np.reshape(new_obs[0],[-1,env.maxSide**2]) == 1
        stateDescriptorsNextFlat = np.array([np.concatenate([[new_obs[1]==1],stateDescriptorsNextFlat[0]])])
        qNext = getTabular(stateDescriptorsNextFlat)[0]
        
        # Calculate TD target
        qNextmax = np.max(qNext)
        target = rew + (1-done) * gamma * qNextmax

        # Update value function
        qCurrTarget = np.copy(qCurr)
        qCurrTarget[action] = target # target avg value
        trainTabular(stateDescriptorsFlat,[qCurrTarget])

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
        
        obs = np.copy(new_obs)

#    # save value function
#    filehandle = open("saved.pkl","wb")
#    pickle.dump(q_func_tabular,filehandle)
#    filehandle.close()

    # display value function
    obs = env.reset()
    print(str(obs[0][:,:,0]))
    
    stateDescriptorsFlat = np.reshape(obs[0],[-1,env.maxSide**2]) == 1
    stateDescriptorsFlat = np.array([np.concatenate([[False],stateDescriptorsFlat[0]])])
    qCurrNotHolding = getTabular(stateDescriptorsFlat)[0]

    print("Value function for pick action in hold-nothing state:")
    print(str(np.reshape(qCurrNotHolding[0:env.num_moves],[env.maxSide,env.maxSide])))
    print("Value function for place action in hold-nothing state:")
    print(str(np.reshape(qCurrNotHolding[env.num_moves:2*env.num_moves],[env.maxSide,env.maxSide])))

    # zero out first one to simulate picking
    coords = np.array(np.nonzero(obs[0] == 1))
    obs[0][coords[0,0],coords[1,0],0] = 0
    
    stateDescriptorsFlat = np.reshape(obs[0],[-1,env.maxSide**2]) == 1
    stateDescriptorsFlat = np.array([np.concatenate([[True],stateDescriptorsFlat[0]])])
    qCurrHolding = getTabular(stateDescriptorsFlat)[0]
    
    print("Value function for pick action in hold-1 state:")
    print(str(np.reshape(qCurrHolding[0:env.num_moves],[env.maxSide,env.maxSide])))
    print("Value function for place action in hold-1 state:")
    print(str(np.reshape(qCurrHolding[env.num_moves:2*env.num_moves],[env.maxSide,env.maxSide])))
    


if __name__ == '__main__':
    main()


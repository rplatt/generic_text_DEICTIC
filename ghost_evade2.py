#
# Tabular deictic version of ghost-evade.
# This one uses cascaded value functions to estimate a worst-case value.
# I see an improvement in performance when using the worst-case value vs. the avg-case value
# This code can also be used to evaluate the difference between same-patch next state 
# and any-patch next state. In general, I seem to find that same-patch works better.
# It converges slightly faster and seems to converge to a higher value: 120 vs 85...
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

    tabularQ1 = 100*np.ones([deicticShape[0]+1,deicticShape[1]+1,deicticShape[0]+1,deicticShape[1]+1, num_actions])
    tabularQ2 = 100*np.ones([deicticShape[0]+1,deicticShape[1]+1,deicticShape[0]+1,deicticShape[1]+1, num_actions])
    tabularQ3 = 100*np.ones([deicticShape[0]+1,deicticShape[1]+1,deicticShape[0]+1,deicticShape[1]+1, num_actions])
    tabularQ4 = 100*np.ones([deicticShape[0]+1,deicticShape[1]+1,deicticShape[0]+1,deicticShape[1]+1, num_actions])
    tabularQ5 = 100*np.ones([deicticShape[0]+1,deicticShape[1]+1,deicticShape[0]+1,deicticShape[1]+1, num_actions])
    
    obs = env.reset()
#    OHEnc = np.identity(max_num_groups)


    for t in range(max_timesteps):

        # get current q-values
        obsDeictic = getDeicticObs(obs)
        stateCurr = convertState(obsDeictic)
#        qCurr = tabularQ1[stateCurr[0], stateCurr[1], stateCurr[2], stateCurr[3],:]
        qCurr = tabularQ5[stateCurr[0], stateCurr[1], stateCurr[2], stateCurr[3],:]
        
        # select action
        action = np.argmax(np.max(qCurr,0))
        selPatch = np.argmax(np.max(qCurr,1))
        if np.random.rand() < exploration.value(t):
#            print("Random action!")
            action = np.random.randint(env.action_space.n)

#        if t > max_timesteps * 0.75:
#            print("obs:\n" + str(np.squeeze(obs)))
#            print("patch:\n" + str(np.reshape(obsDeictic[selPatch],(3,3))))
#            print("action: " + str(action) + ", patch: " + str(selPatch))


        # take action
        new_obs, rew, done, _ = env.step(action)
            
        # get next q-values
        stateNext = convertState(getDeicticObs(new_obs))
        qNext5 = tabularQ5[stateNext[0], stateNext[1], stateNext[2], stateNext[3],:]
        
        # same-patch next state (this seems to be better)
        qNextmaxa = np.max(qNext5,1)
        
#        # any-patch next state (this seems to be worse)
#        qNextmaxa = np.repeat(np.max(qNext5),num_deictic_patches)

        targets = rew + (1-done) * gamma * qNextmaxa
        
#        max_negative_td_error = np.max(np.abs(targets - tabularQ[stateCurr[0], stateCurr[1], stateCurr[2], stateCurr[3],action]) * np.int32(targets < tabularQ[stateCurr[0], stateCurr[1], stateCurr[2], stateCurr[3],action]))
#        if max_negative_td_error > 5:
#            max_negative_td_error
#        print("max_td_error: " + str(max_negative_td_error))
#        tabularQ[stateCurr[0], stateCurr[1], stateCurr[2], stateCurr[3],action] = np.minimum(targets, tabularQ[stateCurr[0], stateCurr[1], stateCurr[2], stateCurr[3],action])
        
        target2_mask = targets < tabularQ1[stateCurr[0], stateCurr[1], stateCurr[2], stateCurr[3],action]
        target3_mask = targets < tabularQ2[stateCurr[0], stateCurr[1], stateCurr[2], stateCurr[3],action]
        target4_mask = targets < tabularQ3[stateCurr[0], stateCurr[1], stateCurr[2], stateCurr[3],action]
        target5_mask = targets < tabularQ4[stateCurr[0], stateCurr[1], stateCurr[2], stateCurr[3],action]
        targets1 = targets
        targets2 = target2_mask * targets + (1 - target2_mask) * tabularQ2[stateCurr[0], stateCurr[1], stateCurr[2], stateCurr[3],action]
        targets3 = target3_mask * targets + (1 - target3_mask) * tabularQ3[stateCurr[0], stateCurr[1], stateCurr[2], stateCurr[3],action]
        targets4 = target4_mask * targets + (1 - target4_mask) * tabularQ4[stateCurr[0], stateCurr[1], stateCurr[2], stateCurr[3],action]
        targets5 = target5_mask * targets + (1 - target5_mask) * tabularQ5[stateCurr[0], stateCurr[1], stateCurr[2], stateCurr[3],action]
        
        
        tabularQ1[stateCurr[0], stateCurr[1], stateCurr[2], stateCurr[3],action] = \
            (1 - learning_alpha) * tabularQ1[stateCurr[0], stateCurr[1], stateCurr[2], stateCurr[3],action] \
            + learning_alpha * targets1
        tabularQ2[stateCurr[0], stateCurr[1], stateCurr[2], stateCurr[3],action] = \
            (1 - learning_alpha) * tabularQ2[stateCurr[0], stateCurr[1], stateCurr[2], stateCurr[3],action] \
            + learning_alpha * targets2
        tabularQ3[stateCurr[0], stateCurr[1], stateCurr[2], stateCurr[3],action] = \
            (1 - learning_alpha) * tabularQ3[stateCurr[0], stateCurr[1], stateCurr[2], stateCurr[3],action] \
            + learning_alpha * targets3
        tabularQ4[stateCurr[0], stateCurr[1], stateCurr[2], stateCurr[3],action] = \
            (1 - learning_alpha) * tabularQ4[stateCurr[0], stateCurr[1], stateCurr[2], stateCurr[3],action] \
            + learning_alpha * targets4
        tabularQ5[stateCurr[0], stateCurr[1], stateCurr[2], stateCurr[3],action] = \
            (1 - learning_alpha) * tabularQ5[stateCurr[0], stateCurr[1], stateCurr[2], stateCurr[3],action] \
            + learning_alpha * targets5


#        # Store transition in the replay buffer.
#        replay_buffer.add(obs, action, rew, new_obs, float(done))

        # bookkeeping for storing episode rewards
        episode_rewards[-1] += rew
        if done:
#            print("************************* Episode done! **************************")
            new_obs = env.reset()
            episode_rewards.append(0.0)
        mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
        num_episodes = len(episode_rewards)
        if done and print_freq is not None and len(episode_rewards) % print_freq == 0:
            print("steps: " + str(t) + ", episodes: " + str(num_episodes) + ", mean 100 episode reward: " + str(mean_100ep_reward) + ", % time spent exploring: " + str(int(100 * exploration.value(t))) + ", max q at curr state: " + str(np.max(qCurr)))
        
#            # stop at the end of training
#            if t > max_timesteps * 0.75:
#                np.set_printoptions(precision=1)
#                
#                obsDeicticReshape = np.reshape(obsDeictic,[36,9])
#                qCurr1 = tabularQ1[stateCurr[0], stateCurr[1], stateCurr[2], stateCurr[3],:]
#                qCurr2 = tabularQ2[stateCurr[0], stateCurr[1], stateCurr[2], stateCurr[3],:]
#                qCurr3 = tabularQ3[stateCurr[0], stateCurr[1], stateCurr[2], stateCurr[3],:]
#                qCurr4 = tabularQ4[stateCurr[0], stateCurr[1], stateCurr[2], stateCurr[3],:]
#                qCurr5 = tabularQ5[stateCurr[0], stateCurr[1], stateCurr[2], stateCurr[3],:]
#                todisplay = np.c_[np.max(qCurr1,1), np.max(qCurr2,1), np.max(qCurr3,1), np.max(qCurr4,1), np.max(qCurr5,1), obsDeicticReshape]
#                print("q-values:\n" + str(todisplay))
                
#                print("obs:\n" + str(np.squeeze(obs)))
#                print("patch:\n" + str(np.reshape(obsDeictic[selPatch],(3,3))))
#                print("action: " + str(action) + ", patch: " + str(selPatch))
#                t
            
            
        # *************************************
        # *************************************
#        to do: set break point when there is a decrease in value and study that situation...
        # I noticed the deitic representations are wierd when 10 and 20 are vertically separated by one empty row...
        # env.step came back w/ rew=1 and done=true. that shouldn't happen!
        # *************************************
        # *************************************
            
        obs = new_obs

    t

        

if __name__ == '__main__':
    main()


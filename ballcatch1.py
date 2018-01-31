#
# Tabular deictic version of ballcatch. I explored a few different versions
# of this. They generally all work, although they don't learn quite as fast
# as I would have expected.
#
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
import envs.ballcatch1_standalone as envstandalone

def main():


    env = envstandalone.BallCatch()
    
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

    num_actions = 3
    episode_rewards = [0.0]

    # Create the schedule for exploration starting from 1.
    exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * max_timesteps),
                                 initial_p=1.0,
                                 final_p=exploration_final_eps)

    # Extract deictic patches for an input obs. Each deictic patch has a low level
    # and a foveated view.
    # input: n x n x 1
    # output: dn x dn x 4
    def getDeicticObs(obs):
        windowLen = deicticShape[0]
        obsShape = np.shape(obs)
        obsPadded = np.zeros((obsShape[0]+2*windowLen,obsShape[1]+2*windowLen))
        obsPadded[windowLen:windowLen+obsShape[0],windowLen:windowLen+obsShape[1]] = obs[:,:,0]
        deicticObsThis = np.zeros((windowLen,windowLen,4)) # channel1: zoomin window; channel2: agent in zoomout window; channel3: ball in zoomout window
        deicticObs = []
        for i in range(obsShape[0] - windowLen + 1):
            for j in range(obsShape[1] - windowLen + 1):
                deicticObsThis[:,:,0] = obs[i:i+windowLen,j:j+windowLen,0] == 1 # agent zoomin
                deicticObsThis[:,:,1] = obs[i:i+windowLen,j:j+windowLen,0] == 2 # ball zoomin
                patch = obsPadded[i:i+3*windowLen,j:j+3*windowLen]
                for k in range(1,3):
# THE VERSION BELOW USES A FIXED VIEW
#                    deicticObsThis[:,:,k+1] = [[(k in obs[0:3,0:3,0]), (k in obs[0:3,3:5]), (k in obs[0:3,5:8,0])], 
#                                 [(k in obs[3:5,0:3,0]), (k in obs[3:5,3:5,0]), (k in obs[3:5,5:8,0])],
#                                 [(k in obs[5:8,0:3,0]), (k in obs[5:8,3:5,0]), (k in obs[5:8,5:8,0])]]
# THE VERSION BELOW USES A WIDE VIEW W/ 2 UNITS IN EACH CELL
                    deicticObsThis[:,:,k+1] = [[(k in patch[1:3,1:3]), (k in patch[1:3,3:5]), (k in patch[1:3,5:7])], 
                                 [(k in patch[3:5,1:3]), (k in patch[3:5,3:5]), (k in patch[3:5,5:7])], 
                                 [(k in patch[5:7,1:3]), (k in patch[5:7,3:5]), (k in patch[5:7,5:7])]]
# THE VERSION BELOW USES A WIDE VIEW W/ 3 UNITS IN EACH CELL
#                    deicticObsThis[:,:,k+1] = [[(k in patch[0:3,0:3]), (k in patch[0:3,3:6]), (k in patch[0:3,6:9])], 
#                                 [(k in patch[3:6,0:3]), (k in patch[3:6,3:6]), (k in patch[3:6,6:9])], 
#                                 [(k in patch[6:9,0:3]), (k in patch[6:9,3:6]), (k in patch[6:9,6:9])]]
                deicticObs.append(deicticObsThis.copy()) # CAREFUL WITH APPENDING REFERENCES VS APPENDING COPIES!!! THIS WAS A BUG BEFORE I CORRECTED IT...

        return np.array(deicticObs)

    # input: batch x nxnx1 tensor of observations
    # output: 8 x batch matrix of deictic observations
    def convertState(observations):
        
        # Reshape to batch x flatimage x channel.
        # Channel1 = zoomin agent, channel2 = zoomin ball
        # Channel3 = zoomout agent, channel4 = zoomout ball
        obs = np.zeros((36,9,4))
        for i in range(4):
            obs[:,:,i] = np.reshape(observations[:,:,:,i],[36,9])

        # state_numeric: 4 x batch.
        # row0: pos of agent in zoomin, row1: pos of ball in zoomin
        # row2: pos of agent in zoomout, row3: pos of ball in zoomout
        shape = np.shape(obs)
        state_numeric = 9*np.ones((4,shape[0])) # 9 indicates agent/ball does not appear at this zoom in this glance
        pos = np.nonzero(obs == 1)
        for i in range(4):
            idx = np.nonzero(pos[2]==i)[0]
            state_numeric[i,pos[0][idx]] = pos[1][idx]
#            state_numeric[i,pos[0][pos[2] == i]] = pos[1][pos[2] == i]
        
        return np.int32(state_numeric)

    dimSize = deicticShape[0]*deicticShape[1] + 1
#    tabularQ = 100*np.ones([dimSize, dimSize, dimSize, dimSize, num_actions])
    tabularQ1 = 100*np.ones([dimSize, dimSize, dimSize, dimSize, num_actions])
    tabularQ2 = 100*np.ones([dimSize, dimSize, dimSize, dimSize, num_actions])
    tabularQ3 = 100*np.ones([dimSize, dimSize, dimSize, dimSize, num_actions])
    tabularQ4 = 100*np.ones([dimSize, dimSize, dimSize, dimSize, num_actions])
    tabularQ5 = 100*np.ones([dimSize, dimSize, dimSize, dimSize, num_actions])
    
    obs = env.reset()
#    OHEnc = np.identity(max_num_groups)


    for t in range(max_timesteps):

        # get current q-values
        obsDeictic = getDeicticObs(obs)
        stateCurr = convertState(obsDeictic)

#        # do a couple of spot checks to verify that obsDeictic is correct
#        num2check = 17
#        print(str(obsDeictic[num2check,:,:,0] + obsDeictic[num2check,:,:,1]))
#        print(str(obsDeictic[num2check,:,:,2] + obsDeictic[num2check,:,:,3]))

        qCurr = tabularQ5[stateCurr[0], stateCurr[1], stateCurr[2], stateCurr[3],:]
#        qCurr = tabularQ[stateCurr[0], stateCurr[1], stateCurr[2], stateCurr[3],:]
        
        # select action
        qCurrNoise = qCurr + np.random.random()*0.01 # add small amount of noise to break ties randomly
        action = np.argmax(np.max(qCurrNoise,0))
        selPatch = np.argmax(np.max(qCurrNoise,1))
        if np.random.rand() < exploration.value(t):
            action = np.random.randint(env.action_space.n)


#        env.render()
#        print("action: " + str(action))

        # take action
        new_obs, rew, done, _ = env.step(action)
        new_obs
        
#        if done == 1:
#            print("action: " + str(action) + ", patch: " + str(selPatch) + ", reward: " + str(rew))
#            action
            
        if t > max_timesteps * 1.05:
            print("obs:\n" + str(np.squeeze(obs)))
            print("qCurr:\n" + str(qCurr))
            print("action: " + str(action) + ", patch: " + str(selPatch))
            print("close:\n" + str(obsDeictic[selPatch,:,:,0] + obsDeictic[selPatch,:,:,1]))
            print("far:\n" + str(obsDeictic[selPatch,:,:,2] + obsDeictic[selPatch,:,:,3]))
            action
            
        # get next q-values
        stateNext = convertState(getDeicticObs(new_obs))
        qNext5 = tabularQ5[stateNext[0], stateNext[1], stateNext[2], stateNext[3],:]
#        qNext = tabularQ[stateNext[0], stateNext[1], stateNext[2], stateNext[3],:]
        
        # perform learning update
#        qNextmaxa = np.max(qNext5,1)
        qNextmaxa = np.max(qNext5)
        targets = rew + (1-done) * gamma * qNextmaxa
        
#        max_negative_td_error = np.max(np.abs(targets - tabularQ[stateCurr[0], stateCurr[1], stateCurr[2], stateCurr[3],action]) * np.int32(targets < tabularQ[stateCurr[0], stateCurr[1], stateCurr[2], stateCurr[3],action]))
#        if max_negative_td_error > 5:
#            max_negative_td_error
#        print("max_td_error: " + str(max_negative_td_error))
#        print("curr tabularQ:\n" + str(tabularQ[stateCurr[0], stateCurr[1], stateCurr[2], stateCurr[3],action]))
#        print("targets:\n" + str(targets))        
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
#            print("steps: " + str(t) + ", episodes: " + str(num_episodes) + ", mean 100 episode reward: " + str(mean_100ep_reward) + ", % time spent exploring: " + str(int(100 * exploration.value(t))) + ", max q at curr state: " + str(np.max(qCurr)))
            print("steps: " + str(t) + ", episodes: " + str(num_episodes) + ", mean 100 episode reward: " + str(mean_100ep_reward) + ", % time spent exploring: " + str(int(100 * exploration.value(t))))
        
        obs = new_obs
        
        # stop at the end of training
        if t > max_timesteps * 1.1:
#            np.set_printoptions(precision=1)
#            np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
            np.set_printoptions(formatter={'float_kind':lambda x: "%.1f" % x})
            
#            qCurr1 = tabularQ1[stateCurr[0], stateCurr[1], stateCurr[2], stateCurr[3],:]
#            qCurr2 = tabularQ2[stateCurr[0], stateCurr[1], stateCurr[2], stateCurr[3],:]
#            qCurr3 = tabularQ3[stateCurr[0], stateCurr[1], stateCurr[2], stateCurr[3],:]
#            qCurr4 = tabularQ4[stateCurr[0], stateCurr[1], stateCurr[2], stateCurr[3],:]
#            qCurr5 = tabularQ5[stateCurr[0], stateCurr[1], stateCurr[2], stateCurr[3],:]
#            todisplay = np.c_[np.max(qCurr1,1), np.max(qCurr2,1), np.max(qCurr3,1), np.max(qCurr4,1), np.max(qCurr5,1), obsDeicticReshape]
#            todisplay = np.c_[qCurr5,np.transpose(stateCurr)]

            print("obs:\n" + str(np.squeeze(obs)))

#            todisplay = np.c_[np.max(qCurr5,1),np.transpose(stateCurr)]
#            print("q-values:\n" + str(todisplay))
#
#            print("close:\n" + str(obsDeictic[selPatch,:,:,0] + obsDeictic[selPatch,:,:,1]))
#            print("far:\n" + str(obsDeictic[selPatch,:,:,2] + obsDeictic[selPatch,:,:,3]))
#            print("action: " + str(action) + ", patch: " + str(selPatch))
            action

#                print("obs:\n" + str(np.squeeze(obs)))
#                print("patch:\n" + str(np.reshape(obsDeictic[selPatch],(3,3))))
#                print("action: " + str(action) + ", patch: " + str(selPatch))
#                t
            
            

    t

        

if __name__ == '__main__':
    main()


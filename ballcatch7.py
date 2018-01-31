#
# In this version, I hope to implement a neural network version of the 
# cascaded value function. Should be comparable to ballcatch6.py
#
# I never got this version to work for some reason... restarting at ballcatch8 from ballbatch6 
#
#
import gym
import numpy as np
import tensorflow as tf
import tf_util_rob as U
import models as models
import build_graph_rob as build_graph
from replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from schedules import LinearSchedule
import matplotlib.pyplot as plt


# I had problems w/ the gym environment, so here I made my own standalone class
#import envs.frozen_lake
import envs.ballcatch1_standalone as envstandalone

def main():

    env = envstandalone.BallCatch()
    
    max_timesteps=20000
#    learning_starts=1000
    learning_starts=64
    buffer_size=50000
#    buffer_size=1000
#    buffer_size=64
    exploration_fraction=0.2
    exploration_final_eps=0.02
    print_freq=10
    gamma=.98
#    target_network_update_freq=500
    target_network_update_freq=100
    learning_alpha = 0.2
    
    batch_size=32
#    batch_size=2
    train_freq=1
#    train_freq=8

#    deicticShape = (3,3,1)
    deicticShape = (3,3,4) # four channels for agent, ball at two resolutions...
    num_deictic_patches=36

    num_actions = 3
    episode_rewards = [0.0]
    num_cpu=16
    num_cascade = 5
    

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
#                    deicticObsThis[:,:,k+1] = [[(k in patch[1:3,1:3]), (k in patch[1:3,3:5]), (k in patch[1:3,5:7])], 
#                                 [(k in patch[3:5,1:3]), (k in patch[3:5,3:5]), (k in patch[3:5,5:7])], 
#                                 [(k in patch[5:7,1:3]), (k in patch[5:7,3:5]), (k in patch[5:7,5:7])]]
# THE VERSION BELOW USES A WIDE VIEW W/ 3 UNITS IN EACH CELL
                    deicticObsThis[:,:,k+1] = [[(k in patch[0:3,0:3]), (k in patch[0:3,3:6]), (k in patch[0:3,6:9])], 
                                 [(k in patch[3:6,0:3]), (k in patch[3:6,3:6]), (k in patch[3:6,6:9])], 
                                 [(k in patch[6:9,0:3]), (k in patch[6:9,3:6]), (k in patch[6:9,6:9])]]
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


    def convertStateBatch(observations):
        shape = np.shape(observations)
        state_numeric_batch = []
        for batch in range(shape[0]):
            state_numeric_batch.append(convertState(observations[batch]))
        return(np.array(state_numeric_batch))


    # Same as getDeicticObs, but it operates on a batch rather than a single obs
    # input: obs -> batches x glances x 3 x 3 x 4
    def getDeicticObsBatch(obs):
        obsShape = np.shape(obs)
        deicticObsBatch = []
        for batch in range(obsShape[0]):
            deicticObsBatch.append(getDeicticObs(obs[batch]))
        return(np.array(deicticObsBatch))

    # conv model parameters: (num_outputs, kernel_size, stride)
    model = models.cnn_to_mlp(
#        convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)], # used in pong
#        hiddens=[256],  # used in pong
#        convs=[(8,4,1)], # used for non-deictic TestRob3-v0
#        convs=[(8,3,1)], # used for deictic TestRob3-v0
        convs=[(16,3,1)], # used for deictic TestRob3-v0
#        convs=[(4,3,1)], # used for deictic TestRob3-v0
#        convs=[(16,3,1)], # used for deictic TestRob3-v0
#        convs=[(8,2,1)], # used for deictic TestRob3-v0
        hiddens=[16],
        dueling=True
    )

    q_func=model
    lr=1e-3
    
    def make_obs_ph(name):
        return U.BatchInput(deicticShape, name=name)

    def make_target_ph(name):
        return U.BatchInput([num_cascade], name=name)

    def make_actions_ph(name):
        return U.BatchInput([num_cascade], dtype=tf.int32, name=name)

    sess = U.make_session(num_cpu)
    sess.__enter__()

    getq, getq_target, targetTrain, update_target = build_graph.build_train_cascaded(
        make_obs_ph=make_obs_ph,
        make_target_ph=make_target_ph,
        make_actions_ph=make_actions_ph,
        q_func=q_func,
        num_cascade=num_cascade,
        num_actions=env.action_space.n,
        batch_size=batch_size,
        num_deictic_patches=num_deictic_patches,
        optimizer=tf.train.AdamOptimizer(learning_rate=lr),
        gamma=gamma,
        grad_norm_clipping=10,
        double_q=False
    )

    # Initialize the parameters and copy them to the target network.
    U.initialize()
    update_target()

    dimSize = deicticShape[0]*deicticShape[1] + 1
    tabularQ = 100*np.ones((dimSize, dimSize, dimSize, dimSize, num_cascade, num_actions))
    
    replay_buffer = ReplayBuffer(buffer_size)
    obs = env.reset()


    for t in range(max_timesteps):
        
        obsDeictic = getDeicticObs(obs)
        
        # Get current q-values: tabular version        
#        stateCurr = convertState(obsDeictic)
#        qCurr = tabularQ[stateCurr[0], stateCurr[1], stateCurr[2], stateCurr[3],:,:]

        # Get current q-values: neural network version        
        qCurr = np.reshape(getq(np.array(obsDeictic)),[num_deictic_patches,num_cascade,num_actions])

        # Select action using last elt in cascade, qCurr[:,-1,:]
        qCurrNoise = qCurr[:,-1,:] + np.random.random((num_deictic_patches,num_actions))*0.01 # add small amount of noise to break ties randomly
        action = np.argmax(np.max(qCurrNoise,0))
        selPatch = np.argmax(np.max(qCurrNoise,1))
        if np.random.rand() < exploration.value(t):
            action = np.random.randint(env.action_space.n)

        # take action
        new_obs, rew, done, _ = env.step(action)
        replay_buffer.add(obs, action, rew, new_obs, float(done))
        
        # debug
        if t > max_timesteps * 1.1:
            print("obs:\n" + str(np.squeeze(obs)))
            print("qCurr:\n" + str(qCurr))
            print("action: " + str(action) + ", patch: " + str(selPatch))
            print("close:\n" + str(obsDeictic[selPatch,:,:,0] + obsDeictic[selPatch,:,:,1]))
            print("far:\n" + str(obsDeictic[selPatch,:,:,2] + obsDeictic[selPatch,:,:,3]))
            action
            
        # sample from replay buffer and train
        if t > learning_starts and t % train_freq == 0:

            obs_resize_to_network = [batch_size*num_deictic_patches,deicticShape[0],deicticShape[1],deicticShape[2]]
            q_reseize_from_network = [batch_size,num_deictic_patches,num_cascade,num_actions]
            obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(batch_size)
            obses_t_deic = getDeicticObsBatch(obses_t)
            obses_tp1_deic = getDeicticObsBatch(obses_tp1)
            
            # Get curr, next values: tabular version
            stateCurr = convertStateBatch(getDeicticObsBatch(obses_t))
            stateNext = convertStateBatch(getDeicticObsBatch(obses_tp1))
            qCurr = tabularQ[stateCurr[:,0,:], stateCurr[:,1,:], stateCurr[:,2,:], stateCurr[:,3,:],:,:]
            qNext = tabularQ[stateNext[:,0,:], stateNext[:,1,:], stateNext[:,2,:], stateNext[:,3,:],:,:]
            
            # Get curr, next values: neural network version
#            qCurr = np.reshape(getq(np.reshape(obses_t_deic,obs_resize_to_network)),q_reseize_from_network)
#            qNext = np.reshape(getq_target(np.reshape(obses_tp1_deic,obs_resize_to_network)),q_reseize_from_network)
#            qNext = np.reshape(getq(np.reshape(obses_tp1_deic,obs_resize_to_network)),q_reseize_from_network)
            
            qNextmax = np.max(np.max(qNext[:,:,-1,:],2),1)
            targetsRaw = rewards + (1-dones) * gamma * qNextmax
            targetsTiled = np.tile(np.reshape(targetsRaw,[batch_size,1]),[1,num_deictic_patches])
            actionsTiled = np.tile(np.reshape(actions,[batch_size,1]),[1,num_deictic_patches])
            actionsTiled = np.tile(actionsTiled,[num_cascade,1,1])
            actionsTiled = np.rollaxis(actionsTiled,0,3)
            
            # Value of action taken
            # qCurrActionSelect has dimensions batch_size x num_deictic_patches x num_cascade
            qCurrActionSelect = np.zeros((batch_size,num_deictic_patches,num_cascade))
            for i in range(num_actions):
                qCurrActionSelect += (actionsTiled == i) * qCurr[:,:,:,i]
            
            targetMask = np.tile(np.reshape(targetsTiled,[batch_size,num_deictic_patches,1]),[1,1,num_cascade]) < qCurrActionSelect
            targets = np.zeros((batch_size,num_deictic_patches,num_cascade))
            targets[:,:,0] = targetsTiled
            targets[:,:,1] = targetMask[:,:,0] * targetsTiled + (1 - targetMask[:,:,0]) * qCurrActionSelect[:,:,1]
            targets[:,:,2] = targetMask[:,:,1] * targetsTiled + (1 - targetMask[:,:,1]) * qCurrActionSelect[:,:,2]
            targets[:,:,3] = targetMask[:,:,2] * targetsTiled + (1 - targetMask[:,:,2]) * qCurrActionSelect[:,:,3]
            targets[:,:,4] = targetMask[:,:,3] * targetsTiled + (1 - targetMask[:,:,3]) * qCurrActionSelect[:,:,4]

            # Update values: tabular version            
            tabularQ[stateCurr[:,0,:], stateCurr[:,1,:], stateCurr[:,2,:], stateCurr[:,3,:],:,actionsTiled[:,:,0]] = \
                (1 - learning_alpha) * tabularQ[stateCurr[:,0,:], stateCurr[:,1,:], stateCurr[:,2,:], stateCurr[:,3,:],:,actionsTiled[:,:,0]] \
                + learning_alpha * targets

#            targets_resize_to_network = [batch_size*num_deictic_patches, num_cascade]
#            td_error_out, q_t_action_select_out, targets_out = targetTrain(
#                    np.reshape(obses_t_deic,obs_resize_to_network),
#                    np.reshape(actionsTiled,targets_resize_to_network),
#                    np.reshape(targets,targets_resize_to_network)
#                    )
#            targets_resize_to_network
                
        # Update target network periodically.
        if t > learning_starts and t % target_network_update_freq == 0:
            update_target()


        # bookkeeping for storing episode rewards
        episode_rewards[-1] += rew
        if done:
            new_obs = env.reset()
            episode_rewards.append(0.0)
        mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
        num_episodes = len(episode_rewards)
        if done and print_freq is not None and len(episode_rewards) % print_freq == 0:
#            print("steps: " + str(t) + ", episodes: " + str(num_episodes) + ", mean 100 episode reward: " + str(mean_100ep_reward) + ", % time spent exploring: " + str(int(100 * exploration.value(t))) + ", max q at curr state: " + str(np.max(qCurr)))
            print("steps: " + str(t) + ", episodes: " + str(num_episodes) + ", mean 100 episode reward: " + str(mean_100ep_reward) + ", % time spent exploring: " + str(int(100 * exploration.value(t))))
        
        obs = new_obs
        


        

if __name__ == '__main__':
    main()


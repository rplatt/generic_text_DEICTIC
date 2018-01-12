#
# Adapted from generic_text8.py. This is a simple tabular version of the algorithm
# that uses a dictionary-based tabular value function.
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

import envs.multi_ghost_evade1_standalone as envstandalone
#import envs.ghost_evade1_standalone as envstandalone
#import envs.ballcatch2_standalone as envstandalone

# **** Make tensorflow functions ****

def build_getDeic_Foc(make_obs_ph,deicticShape):
    observations_ph = U.ensure_tf_input(make_obs_ph("observation"))
    patches = tf.extract_image_patches(
            observations_ph.get(),
            ksizes=[1, deicticShape[0], deicticShape[1], 1], 
            strides=[1, 1, 1, 1], 
            rates=[1, 1, 1, 1], 
            padding='VALID')
    patchesShape = tf.shape(patches)
    patchesTiled = tf.reshape(patches,[patchesShape[0]*patchesShape[1]*patchesShape[2],deicticShape[0],deicticShape[1]])
    patchesTiledStacked = tf.stack([tf.equal(patchesTiled,1), tf.equal(patchesTiled,2)],axis=-1)
    getDeic = U.function(inputs=[observations_ph], outputs=patchesTiledStacked)
    return getDeic


# Both focused and coarse representation. Assumes one-channel image
def build_getDeic_FocCoarse(make_obs_ph,deicticShape):
    
    observations_ph = U.ensure_tf_input(make_obs_ph("observation"))

    # create padded image
    obs = observations_ph.get()
    shape = tf.shape(obs)
    obsZeroPadded = tf.image.resize_image_with_crop_or_pad(obs,shape[1]+2*deicticShape[0],shape[2]+2*deicticShape[0])

    # extract large patches from padded image
    patchesLarge = tf.extract_image_patches(
            obsZeroPadded,
            ksizes=[1, 3*deicticShape[0], 3*deicticShape[1], 1], 
            strides=[1, 1, 1, 1], 
            rates=[1, 1, 1, 1], 
            padding='VALID')

    # reshape into focused and large images
    patchesShape = tf.shape(patchesLarge)
    patchesTiledLarge = tf.reshape(patchesLarge,[patchesShape[0]*patchesShape[1]*patchesShape[2],3*deicticShape[0],3*deicticShape[1],1])
    patchesTiledFocused = patchesTiledLarge[:,deicticShape[0]:2*deicticShape[0],deicticShape[1]:2*deicticShape[1],0]

    # get two coarse images: one for agent and one for the ghost (might make this more efficient by doing the resize only once...)
    coarseAgent = tf.image.resize_area(tf.cast(tf.equal(patchesTiledLarge,1.),tf.int32), deicticShape[0:2])[:,:,:,0] > 0
    coarseGhost = tf.image.resize_area(tf.cast(tf.equal(patchesTiledLarge,2.),tf.int32), deicticShape[0:2])[:,:,:,0] > 0

    patchesTiledStacked = tf.stack([tf.equal(patchesTiledFocused,1), tf.equal(patchesTiledFocused,2), coarseAgent, coarseGhost],axis=-1)
    
    getDeic = U.function(inputs=[observations_ph], outputs=patchesTiledStacked)
    return getDeic


def main():

    env = envstandalone.MultiGhostEvade()
#    env = envstandalone.GhostEvade()
#    env = envstandalone.BallCatch()
    
#    max_timesteps=40000
    max_timesteps=80000
    learning_starts=1000
    buffer_size=50000
#    exploration_fraction=0.2
    exploration_fraction=0.4
    exploration_final_eps=0.02
    print_freq=10
    gamma=.98
#    target_network_update_freq=500
#    target_network_update_freq=100
#    target_network_update_freq=10
    target_network_update_freq=1
    learning_alpha = 0.2
    
    batch_size=32
#    batch_size=64
#    batch_size=1024
    train_freq=1

#    obsShape = (8,8,1)
    obsShape = env.observation_space.shape
#    deicticShape = (3,3,2)
#    deicticShape = (3,3,4)
#    deicticShape = (4,4,2)
#    deicticShape = (4,4,4)
    deicticShape = (5,5,2)
#    deicticShape = (6,6,2)
#    deicticShape = (8,8,2)
#    num_deictic_patches = 36
#    num_deictic_patches = 25
    num_deictic_patches = 16
#    num_deictic_patches = 9
#    num_deictic_patches = 1

    num_cascade = 5
    num_actions = env.action_space.n

    episode_rewards = [0.0]
    num_cpu=16
    
    # Create the schedule for exploration starting from 1.
    exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * max_timesteps),
                                 initial_p=1.0,
                                 final_p=exploration_final_eps)

    # Dictionary-based value function
    q_func = {}
    
    def make_obs_ph(name):
        return U.BatchInput(obsShape, name=name)

    def getTabularKeys(obsDeictic):
        obsDeicticTiled = np.reshape(obsDeictic,[-1,deicticShape[0]*deicticShape[1]*deicticShape[2]])
        obsBits = np.packbits(obsDeicticTiled,1)
        obsKeys = 0
        for i in range(np.shape(obsBits)[1]):
            # IMPORTANT: the type cast below (UINT64) must be large enough to support the size of obsBits
            # if it is too small, we get hash collisions...
            obsKeys = obsKeys + (256**i) * np.uint64(obsBits[:,i])
        return obsKeys
    
    def getTabular(obsDeictic):
        keys = getTabularKeys(obsDeictic)
        return np.array([q_func[x] if x in q_func else 1000*np.ones([num_cascade,num_actions]) for x in keys])
    
    def trainTabular(obsDeictic,qCurrTargets):
        keys = getTabularKeys(obsDeictic)
        alpha=0.5
        for i in range(len(keys)):
            if keys[i] in q_func:
                q_func[keys[i]] = (1-alpha)*q_func[keys[i]] + alpha*qCurrTargets[i]
            else:
                q_func[keys[i]] = qCurrTargets[i]


    sess = U.make_session(num_cpu)
    sess.__enter__()

    getDeic = build_getDeic_Foc(make_obs_ph=make_obs_ph,deicticShape=deicticShape)
    
    # Initialize the parameters and copy them to the target network.
    U.initialize()
    
    obs = env.reset()

    timerStart = time.time()
    for t in range(max_timesteps):

        # Get current obervations
        obsDeictic = getDeic([obs])
        qCurr = getTabular(obsDeictic)

        # select action
        qCurrNoise = qCurr + np.random.random(np.shape(qCurr))*0.01 # add small amount of noise to break ties randomly
        action = np.argmax(np.max(qCurrNoise[:,-1,:],0)) # USE CASCADE
        if np.random.rand() < exploration.value(t):
            action = np.random.randint(env.action_space.n)

        # take action
        new_obs, rew, done, _ = env.step(action)

        # Get next obervations
        obsNextDeictic = getDeic([new_obs])
        qNext = getTabular(obsNextDeictic)

        # Calculate TD target
        qNextmax = np.max(qNext[:,-1,:],1) # USE CASCADE
        targets = rew + (1-done) * gamma * qNextmax

        # Update dictionary value function
        qCurrTargets = np.copy(qCurr)

        # Copy into cascade with pruning.
        qCurrTargets[:,0,action] = targets
        for i in range(num_cascade-1):
            mask = targets < qCurr[:,i,action]
            qCurrTargets[:,i+1,action] = \
                mask*targets + \
                (1-mask)*qCurr[:,i+1,action]
        
#        qCurrTargets[:,action] = np.minimum(targets,qCurrTargets[:,action])
        
        
        trainTabular(obsDeictic,qCurrTargets)

        if t > 3000:
            obsDeictic

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
        
        obs = new_obs



if __name__ == '__main__':
    main()


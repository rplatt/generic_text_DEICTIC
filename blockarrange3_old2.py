#
# Batched version of tabular deictic RL.
#
# Adapted from blockarrange2.py
#
# Result: I notice that convergence is much faster with a smaller buffer size. However,
#         it eventually converges for any buffer size...
#
import gym
import numpy as np
import time as time
import tensorflow as tf
import tf_util_rob as U
import models as models
from replay_buffer2 import ReplayBuffer, PrioritizedReplayBuffer
#from replay_buffer3 import ReplayBuffer, PrioritizedReplayBuffer
from schedules import LinearSchedule

#import envs.blockarrange1_standalone as envstandalone
import envs.blockarrange1_unitstepcost_standalone as envstandalone

# **** Make tensorflow functions ****

def build_getMoveActionDescriptors(make_obs_ph,deicticShape):
    
    if (deicticShape[0] % 2 == 0) or (deicticShape[1] % 2 == 0):
        print("build_getActionDescriptors ERROR: first two elts of deicticShape must by odd")
        
    observations_ph = U.ensure_tf_input(make_obs_ph("observation"))
    obs = observations_ph.get()
    shape = tf.shape(obs)
    deicticPad = np.floor(np.array(deicticShape)-1)
    obsZeroPadded = tf.image.resize_image_with_crop_or_pad(obs,shape[1]+deicticPad[0],shape[2]+deicticPad[1])
    patches = tf.extract_image_patches(
#            observations_ph.get(),
            obsZeroPadded,
            ksizes=[1, deicticShape[0], deicticShape[1], 1], 
            strides=[1, 1, 1, 1], 
            rates=[1, 1, 1, 1], 
            padding='VALID')
    patchesShape = tf.shape(patches)
    patchesTiled = tf.reshape(patches,[patchesShape[0]*patchesShape[1]*patchesShape[2],deicticShape[0],deicticShape[1]])
    patchesTiledStacked = tf.stack([tf.equal(patchesTiled,1), tf.equal(patchesTiled,2)],axis=-1)
    X,Y = tf.meshgrid(tf.range(shape[1]),tf.range(shape[2]))
    moveActions = tf.stack([tf.reshape(Y,[shape[1]*shape[2],]), tf.reshape(X,[shape[1]*shape[2],])],axis=0)
    getMoveActionDescriptors = U.function(inputs=[observations_ph], outputs=patchesTiledStacked)
#    getMoveActionDescriptors = U.function(inputs=[observations_ph], outputs=[patchesTiledStacked, moveActions])
    return getMoveActionDescriptors



def main():

    # Define environment
    env = envstandalone.BlockArrange()

    # Dictionary-based value function
    q_func = {}

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
        return np.array([q_func[x] if x in q_func else 0*np.ones(num_states) for x in keys])
    
    def trainTabular(vectorKey,qCurrTargets):
        keys = getTabularKeys(vectorKey)
        alpha=1.0
        for i in range(len(keys)):
            if keys[i] in q_func:
                q_func[keys[i]] = (1-alpha)*q_func[keys[i]] + alpha*qCurrTargets[i]
            else:
                q_func[keys[i]] = qCurrTargets[i]


    # Standard DQN parameters
    max_timesteps=40000
#    learning_starts=1000
    learning_starts=10
#    buffer_size=50000
#    buffer_size=10000
#    buffer_size=1000
#    buffer_size=100
    buffer_size=1
    exploration_fraction=0.2
    exploration_final_eps=0.02
    print_freq=1
    gamma=.98
    target_network_update_freq=1
#    batch_size=32
    batch_size=1
    train_freq=1
    num_cpu = 16
    exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * max_timesteps),
                                 initial_p=1.0,
                                 final_p=exploration_final_eps)
    replay_buffer = ReplayBuffer(buffer_size)

    # Deictic state/action parameters
    deicticShape = (3,3,2) # IMPORTANT: first two elts of deicticShape must be odd
    deicticActionShape = (3,3,4)
    num_cascade = 5
    num_states = env.num_blocks + 1 # one more state than blocks to account for not holding anything
    num_patches = env.maxSide**2
    num_actions = 2*num_patches
    num_actions_discrete = 2

    
    def make_obs_ph(name):
        return U.BatchInput(env.observation_space.spaces[0].shape, name=name)
    
    getMoveActionDescriptors = build_getMoveActionDescriptors(make_obs_ph=make_obs_ph,deicticShape=deicticShape)

    # Start tensorflow session
    sess = U.make_session(num_cpu)
    sess.__enter__()

    episode_rewards = [0.0]
    timerStart = time.time()
    obs = env.reset()
    for t in range(max_timesteps):
        
        # Get state: in range(0,env.num_blocks)
        stateDeictic = obs[1] # holding

        # Get action set: <num_patches> pick actions followed by <num_patches> place actions
        moveDescriptors = getMoveActionDescriptors([obs[0]])
        actionsPickDescriptors = np.concatenate([moveDescriptors, np.zeros(np.shape(moveDescriptors))],axis=3)
        actionsPlaceDescriptors = np.concatenate([np.zeros(np.shape(moveDescriptors)), moveDescriptors],axis=3)
        actionDescriptors = np.r_[actionsPickDescriptors,actionsPlaceDescriptors]
        actionDescriptors = np.reshape(actionDescriptors,[-1,deicticActionShape[0]*deicticActionShape[1]*deicticActionShape[2]]) == 1

        # Get q-values
        qCurr = getTabular(actionDescriptors)
        
        # select action
        qCurrNoise = qCurr + np.random.random(np.shape(qCurr))*0.01 # add small amount of noise to break ties randomly
        action = np.argmax(qCurrNoise[:,stateDeictic])
        if np.random.rand() < exploration.value(t):
            action = np.random.randint(num_actions)

        # take action
        new_obs, rew, done, _ = env.step(action)
        replay_buffer.add(stateDeictic, actionDescriptors[action,:], rew, new_obs, float(done))

        if t > learning_starts and t % train_freq == 0:

            states_t, actions, rewards, images_tp1, states_tp1, dones = replay_buffer.sample(batch_size)
            
            moveDescriptorsNext1 = getMoveActionDescriptors(images_tp1)
            actionsPickDescriptorsNext1 = np.concatenate([moveDescriptorsNext1, np.zeros(np.shape(moveDescriptorsNext1))],axis=3)
            actionsPlaceDescriptorsNext1 = np.concatenate([np.zeros(np.shape(moveDescriptorsNext1)), moveDescriptorsNext1],axis=3)
            actionDescriptorsNext1 = np.stack([actionsPickDescriptorsNext1, actionsPlaceDescriptorsNext1], axis=0)
            actionDescriptorsNextFlat1 = np.reshape(actionDescriptorsNext1,[batch_size*num_patches*num_actions_discrete,-1]) == 1

            # Get action set: <num_patches> pick actions followed by <num_patches> place actions
            moveDescriptorsNext = getMoveActionDescriptors([new_obs[0]])
            actionsPickDescriptorsNext = np.concatenate([moveDescriptorsNext, np.zeros(np.shape(moveDescriptorsNext))],axis=3)
            actionsPlaceDescriptorsNext = np.concatenate([np.zeros(np.shape(moveDescriptorsNext)), moveDescriptorsNext],axis=3)
            actionDescriptorsNext = np.r_[actionsPickDescriptorsNext,actionsPlaceDescriptorsNext]
            actionDescriptorsNext = np.reshape(actionDescriptorsNext,[-1,deicticActionShape[0]*deicticActionShape[1]*deicticActionShape[2]]) == 1

            qNextFlat1 = getTabular(actionDescriptorsNextFlat1)
            qNext1 = np.reshape(qNextFlat1,[batch_size,num_patches,num_actions_discrete,num_states])
            qNextmax1 = np.max(np.max(qNext1[range(batch_size),:,:,states_tp1],2),1)
            targets1 = rewards + (1-dones) * gamma * qNextmax1


            # Get state: in range(0,env.num_blocks)
            stateDeicticNext = new_obs[1] # holding
    
            # Calculate TD target
            qNext = getTabular(actionDescriptorsNext)
            qNextmax = np.max(qNext[:,stateDeicticNext])
            target = rew + (1-done) * gamma * qNextmax

    
            qCurrTarget1 = getTabular(actions)
            qCurrTarget1[range(batch_size),states_t] = targets1
            trainTabular(actions, qCurrTarget1)


            # Update dictionary value function
            qCurrTarget = qCurr[action,:]
            qCurrTarget[stateDeictic] = target # target avg value
##            trainTabular([actionDescriptors[action,:]],[qCurrTarget])
#
            if np.sum(qCurrTarget != qCurrTarget1) > 0:
                print("ERROR: qCurrTarget")


#        # sample from replay buffer and train
#        if t > learning_starts and t % train_freq == 0:
#
#            states_t, actionDescriptors, rewards, images_tp1, states_tp1, dones = replay_buffer.sample(batch_size)
#
#            moveDescriptorsNext = getMoveActionDescriptors(images_tp1)
#            actionsPickDescriptorsNext = np.concatenate([moveDescriptorsNext, np.zeros(np.shape(moveDescriptorsNext))],axis=3)
#            actionsPlaceDescriptorsNext = np.concatenate([np.zeros(np.shape(moveDescriptorsNext)), moveDescriptorsNext],axis=3)
#            actionDescriptorsNext = np.stack([actionsPickDescriptorsNext, actionsPlaceDescriptorsNext], axis=1)
#
#            actionDescriptorsNextFlat = np.reshape(actionDescriptorsNext,[batch_size*num_patches*num_actions_discrete,-1]) == 1
#            qNextFlat = getTabular(actionDescriptorsNextFlat)
#            qNext = np.reshape(qNextFlat,[batch_size,num_patches,num_actions_discrete,num_states])
#            
##            states_tp1Full = np.repeat(states_tp1,2*num_patches)
##            qNextTiled = np.reshape(qNext1[range(2*batch_size*num_patches),states_tp1Full],[batch_size,2,num_patches,-1])
##            qNextmax = np.max(np.max(np.max(qNextTiled,3),2),1)
#            qNextmax = np.max(np.max(qNext[range(batch_size),:,:,states_tp1],2),1)
#            
#            targets = rewards + (1-dones) * gamma * qNextmax
#
#            actionDescriptorsFlat = np.reshape(actionDescriptors,[batch_size,-1]) == 1
#            qCurrTarget = getTabular(actionDescriptorsFlat)
#            
##            qCurrTarget[range(batch_size),states_tp1] = np.minimum(qCurrTarget[range(batch_size),states_tp1], targets)
#            qCurrTarget[range(batch_size),states_tp1] = targets
#            trainTabular(actionDescriptorsFlat, qCurrTarget)



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


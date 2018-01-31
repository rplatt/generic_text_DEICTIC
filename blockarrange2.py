#
# Same as blockarrange1.py (simple tabular min-value RL), but with a different 
# hash fn encoding (the encoding used for the DQN). Converges
# to the optimal two-step policy after approx 8k steps.
#
# Adapted from blockarrange1.py
#
# Results:
#         -- I displayed the value function learned by this code. I notice that 
#            it typically learns to pick *either* the 1 or the 2 block and then
#            to always place it in a particular place wrt the other block. This
#            suggests that it gets stuck in one of the several ways of solving the
#            problem.
#
import gym
import numpy as np
import time as time
import tensorflow as tf
import tf_util_rob as U
import models as models
from replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from schedules import LinearSchedule

#import envs.blockarrange1_standalone as envstandalone
import envs.blockarrange1_rewardonsuccess_standalone as envstandalone
#import envs.blockarrange1_unitstepcost_standalone as envstandalone

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

    np.set_printoptions(formatter={'float_kind':lambda x: "%.2f" % x}, threshold=np.nan)
#    np.set_printoptions(threshold=np.nan)
    
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
#        return np.array([q_func[x] if x in q_func else 10*np.ones(num_states) for x in keys])
        return np.array([q_func[x] if x in q_func else 0*np.ones(num_states) for x in keys])
    
    def trainTabular(vectorKey,qCurrTargets):
        keys = getTabularKeys(vectorKey)
#        alpha=1.0
        alpha=0.2
        for i in range(len(keys)):
            if keys[i] in q_func:
                q_func[keys[i]] = (1-alpha)*q_func[keys[i]] + alpha*qCurrTargets[i]
            else:
                q_func[keys[i]] = qCurrTargets[i]


    env = envstandalone.BlockArrange()

#    max_timesteps=40000
    max_timesteps=10000
    learning_starts=1000
    buffer_size=50000
#    exploration_fraction=0.2
#    exploration_fraction=0.4
    exploration_fraction=0.6
    exploration_final_eps=0.02
    print_freq=1
    gamma=.98
    target_network_update_freq=1
    learning_alpha = 0.2
    
    batch_size=32
    train_freq=1
    num_cpu = 16
    
    gridShape = env.observation_space.spaces[0].shape
    
    # first two elts of deicticShape must be odd
    deicticShape = (3,3,2)
    deicticActionShape = (3,3,4)
    num_deictic_patches = 64
    
    num_cascade = 5
    
    num_blocks = 2
    num_states = num_blocks+1

    episode_rewards = [0.0]
    
    # Create the schedule for exploration starting from 1.
    exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * max_timesteps),
                                 initial_p=1.0,
                                 final_p=exploration_final_eps)

    def make_obs_ph(name):
        return U.BatchInput(gridShape, name=name)

    sess = U.make_session(num_cpu)
    sess.__enter__()

    getMoveActionDescriptors = build_getMoveActionDescriptors(make_obs_ph=make_obs_ph,deicticShape=deicticShape)
    
    obs = env.reset()
    
    num_patches = env.maxSide**2
    num_actions = 2*num_patches

    episode_rewards = [0.0]
    timerStart = time.time()
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

        # Get state: in range(0,env.num_blocks)
        stateDeicticNext = new_obs[1] # holding

        # Get action set: <num_patches> pick actions followed by <num_patches> place actions
        moveDescriptorsNext = getMoveActionDescriptors([new_obs[0]])
        actionsPickDescriptorsNext = np.concatenate([moveDescriptorsNext, np.zeros(np.shape(moveDescriptorsNext))],axis=3)
        actionsPlaceDescriptorsNext = np.concatenate([np.zeros(np.shape(moveDescriptorsNext)), moveDescriptorsNext],axis=3)
        actionDescriptorsNext = np.r_[actionsPickDescriptorsNext,actionsPlaceDescriptorsNext]
        actionDescriptorsNext = np.reshape(actionDescriptorsNext,[-1,deicticActionShape[0]*deicticActionShape[1]*deicticActionShape[2]]) == 1

        # Calculate TD target
        qNext = getTabular(actionDescriptorsNext)
        qNextmax = np.max(qNext[:,stateDeicticNext])
        target = rew + (1-done) * gamma * qNextmax

        # Update dictionary value function
        qCurrTarget = qCurr[action,:]
#        qCurrTarget[stateDeictic] = np.minimum(qCurrTarget[stateDeictic], target) # target min value
        qCurrTarget[stateDeictic] = target # target avg value
        trainTabular([actionDescriptors[action,:]],[qCurrTarget])

#        print(t)
        

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
    
    
    # display value function
    obs = env.reset()
    moveDescriptors = getMoveActionDescriptors([obs[0]])
    actionsPickDescriptors = np.concatenate([moveDescriptors, np.zeros(np.shape(moveDescriptors))],axis=3)
    actionsPlaceDescriptors = np.concatenate([np.zeros(np.shape(moveDescriptors)), moveDescriptors],axis=3)
    
    print(str(obs[0][:,:,0]))
    
#    qPick = getq(actionsPickDescriptors)
    qPick = getTabular(np.reshape(actionsPickDescriptors,[num_patches,-1])==1)
    print("Value function for pick action in hold-nothing state:")
    print(str(np.reshape(qPick[:,0],[8,8])))

#    qPlace = getq(actionsPlaceDescriptors)
    qPlace = getTabular(np.reshape(actionsPlaceDescriptors,[num_patches,-1])==1)
    print("Value function for place action in hold-1 state:")
    print(str(np.reshape(qPlace[:,1],[8,8])))
    
    print("Value function for place action in hold-2 state:")
    print(str(np.reshape(qPlace[:,2],[8,8])))

#    print("Pick descriptors:")
#    print(str(np.reshape(actionsPickDescriptors,[num_patches,-1])))
#    
#    print("Place descriptors:")
#    print(str(np.reshape(actionsPlaceDescriptors,[num_patches,-1])))
#    
#    qPlace
    

if __name__ == '__main__':
    main()


#
# This version of simple tabular deictic converges within 2k iterations.
#
# Adapted from blockarrange1.py w/ elements taken from blockarrange8.py
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

import envs.blockarrange3_rewardonsuccess_standalone as envstandalone

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
            obsZeroPadded,
            ksizes=[1, deicticShape[0], deicticShape[1], 1], 
            strides=[1, 1, 1, 1], 
            rates=[1, 1, 1, 1], 
            padding='VALID')
    patchesShape = tf.shape(patches)
    patchesTiled = tf.reshape(patches,[patchesShape[0]*patchesShape[1]*patchesShape[2],deicticShape[0],deicticShape[1]])
    getMoveActionDescriptors = U.function(inputs=[observations_ph], outputs=patchesTiled)
    return getMoveActionDescriptors




def main():
    
    np.set_printoptions(formatter={'float_kind':lambda x: "%.2f" % x})

    # Dictionary-based value function
    q_func_tabular = {}

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
        return np.array([q_func_tabular[x] if x in q_func_tabular else 10*np.ones(num_states) for x in keys])
    
#    def trainTabular(vectorKey,qCurrTargets,weights):
    def trainTabular(vectorKey,qCurrTargets):
        keys = getTabularKeys(vectorKey)
        alpha=0.2
        for i in range(len(keys)):
            if keys[i] in q_func_tabular:
                q_func_tabular[keys[i]] = (1-alpha)*q_func_tabular[keys[i]] + alpha*qCurrTargets[i]
#                q_func_tabular[keys[i]] = q_func_tabular[keys[i]] + alpha*weights[i,:]*(qCurrTargets[i] - q_func_tabular[keys[i]]) # (1-alpha)*q_func[keys[i]] + alpha*qCurrTargets[i]
            else:
                q_func_tabular[keys[i]] = qCurrTargets[i]


    env = envstandalone.BlockArrange()

    max_timesteps=4000
    exploration_fraction=0.3
    exploration_final_eps=0.1
    print_freq=1
    gamma=.90
    num_cpu = 16
        
    # first two elts of deicticShape must be odd
    actionShape = (3,3,2)
    num_states = 2 # either holding or not
    num_patches = env.maxSide**2
    num_actions = 2*num_patches
    num_actions_discrete = 2
    valueFunctionType = "TABULAR"
#    valueFunctionType = "DQN"
#    actionSelectionStrategy = "UNIFORM_RANDOM" # actions are selected randomly from collection of all actions
    actionSelectionStrategy = "RANDOM_UNIQUE" # each unique action descriptor has equal chance of being selected

    episode_rewards = [0.0]
    
    # Create the schedule for exploration starting from 1.
    exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * max_timesteps),
                                 initial_p=1.0,
                                 final_p=exploration_final_eps)

    def make_obs_ph(name):
        return U.BatchInput(env.observation_space.spaces[0].shape, name=name)

    getMoveActionDescriptors = build_getMoveActionDescriptors(make_obs_ph=make_obs_ph,deicticShape=actionShape)
    
    sess = U.make_session(num_cpu)
    sess.__enter__()

    obs = env.reset()

    episode_rewards = [0.0]
    timerStart = time.time()
    for t in range(max_timesteps):
        
        # Get action set: <num_patches> pick actions followed by <num_patches> place actions
        moveDescriptors = getMoveActionDescriptors([obs[0]])
        moveDescriptors = moveDescriptors*2-1
        actionsPickDescriptors = np.stack([moveDescriptors, np.zeros(np.shape(moveDescriptors))],axis=3)
        actionsPlaceDescriptors = np.stack([np.zeros(np.shape(moveDescriptors)),moveDescriptors],axis=3)
        actionDescriptors = np.r_[actionsPickDescriptors,actionsPlaceDescriptors]

        # Get qCurr values
        actionDescriptorsFlat = np.reshape(actionDescriptors,[-1,actionShape[0]*actionShape[1]*actionShape[2]]) == 1
        qCurr = getTabular(actionDescriptorsFlat)

        qCurrNoise = qCurr + np.random.random(np.shape(qCurr))*0.01 # add small amount of noise to break ties randomly

        # select action at random
        if actionSelectionStrategy == "UNIFORM_RANDOM":
            action = np.argmax(qCurrNoise[:,obs[1]])
            if np.random.rand() < exploration.value(t):
                action = np.random.randint(num_actions)
        elif actionSelectionStrategy == "RANDOM_UNIQUE":
            _,idx,inv = np.unique(actionDescriptors,axis=0,return_index=True,return_inverse=True)
            actionIdx = np.argmax(qCurrNoise[idx,obs[1]])
            if np.random.rand() < exploration.value(t):
                actionIdx = np.random.randint(len(idx))
            actionsSelected = np.nonzero(inv==actionIdx)[0]
            action = actionsSelected[np.random.randint(len(actionsSelected))]
        else:
            print("Error...")

#        env.render()
        
        # take action
        new_obs, rew, done, _ = env.step(action)

#        print("action: " + str(action) + ", reward: " + str(rew) + ", done: " + str(done))
#        print("action patch:\n" + str(actionDescriptors[action,:]))
#        if done:
#            print("*** done ***")

        # Get action set: <num_patches> pick actions followed by <num_patches> place actions
        moveDescriptorsNext = getMoveActionDescriptors([new_obs[0]])
        moveDescriptorsNext = moveDescriptorsNext*2-1
        
        actionsPickDescriptorsNext = np.stack([moveDescriptorsNext, np.zeros(np.shape(moveDescriptorsNext))],axis=3)
        actionsPlaceDescriptorsNext = np.stack([np.zeros(np.shape(moveDescriptorsNext)), moveDescriptorsNext],axis=3)
        actionDescriptorsNext = np.stack([actionsPickDescriptorsNext, actionsPlaceDescriptorsNext], axis=0)
        actionDescriptorsNext = np.reshape(actionDescriptorsNext,[num_patches*num_actions_discrete,actionShape[0],actionShape[1],actionShape[2]])
        
        actionDescriptorsNextFlat = np.reshape(actionDescriptorsNext,[num_patches*num_actions_discrete,-1]) == 1
        qNext = getTabular(actionDescriptorsNextFlat)

        # Calculate TD target
        qNextmax = np.max(qNext[:,new_obs[1]])
        target = rew + (1-done) * gamma * qNextmax


        # Update value function
        qCurrTarget = qCurr[action,:]
        qCurrTarget[obs[1]] = target # target avg value
        trainTabular([actionDescriptorsFlat[action,:]],[qCurrTarget])


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


    # display value function
    obs = env.reset()
    moveDescriptors = getMoveActionDescriptors([obs[0]])
    moveDescriptors = moveDescriptors*2-1

    actionsPickDescriptors = np.stack([moveDescriptors, np.zeros(np.shape(moveDescriptors))],axis=3)
    actionsPlaceDescriptors = np.stack([np.zeros(np.shape(moveDescriptors)), moveDescriptors],axis=3)
    
    print(str(obs[0][:,:,0]))
    
#    qPickNotHolding = getqNotHolding(actionsPickDescriptors)
#    qPickHolding = getqHolding(actionsPickDescriptors)
#    qPick = np.concatenate([qPickNotHolding,qPickHolding],axis=1)
    
    qPick = getTabular(np.reshape(actionsPickDescriptors,[num_patches,-1])==1)
    print("Value function for pick action in hold-nothing state:")
    print(str(np.reshape(qPick[:,0],[8,8])))
    print("Value function for pick action in hold-1 state:")
    print(str(np.reshape(qPick[:,1],[8,8])))

#    qPlaceNotHolding = getqNotHolding(actionsPlaceDescriptors)
#    qPlaceHolding = getqHolding(actionsPlaceDescriptors)
#    qPlace = np.concatenate([qPlaceNotHolding,qPlaceHolding],axis=1)    

    qPlace = getTabular(np.reshape(actionsPlaceDescriptors,[num_patches,-1])==1)
    print("Value function for place action in hold-nothing state:")
    print(str(np.reshape(qPlace[:,0],[8,8])))
    print("Value function for place action in hold-1 state:")
    print(str(np.reshape(qPlace[:,1],[8,8])))
    


if __name__ == '__main__':
    main()


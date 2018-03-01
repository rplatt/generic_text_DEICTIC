#
# In this version of the code, I'm going to try to implement a deictic version of 
# block arrange that creates look and grasp as two separate actions that must
# be sequenced in order to act successfully.
#
# I never got this code to work. For now, I am abandoning in favor of something
# based more closely on blockarrangeredo 2 or 3...
#
import gym
import numpy as np
import time as time
import tensorflow as tf
import tf_util_rob as U
import models as models
#from replay_buffer2 import ReplayBuffer, PrioritizedReplayBuffer
from replay_buffer6 import ReplayBuffer, PrioritizedReplayBuffer
from schedules import LinearSchedule

import envs.blockarrange_deictic as envstandalone

# **** Make tensorflow functions ****

def build_getq(make_deic_ph, q_func, scope="deepq", qscope="q_func", reuse=None):

    with tf.variable_scope(scope, reuse=reuse):
        actions_ph = U.ensure_tf_input(make_deic_ph("stateaction"))
        q_values = q_func(actions_ph.get(), 1, scope=qscope)
        getq = U.function(inputs=[actions_ph], outputs=q_values)
        return getq


def build_targetTrain(make_deic_ph,
                        make_target_ph,
                        make_weight_ph,
                        q_func,
                        optimizer,
                        scope="deepq", 
                        qscope="q_func",
                        grad_norm_clipping=None,
                        reuse=None):

    with tf.variable_scope(scope, reuse=reuse):
        
        # set up placeholders
        obs_t_input = U.ensure_tf_input(make_deic_ph("action_t_deic"))
        target_input = U.ensure_tf_input(make_target_ph("target"))
        importance_weights_ph = U.ensure_tf_input(make_weight_ph("target"))
    
        # get variables
        q_func_vars = U.scope_vars(U.absolute_scope_name(qscope))
    
        # q values for all actions
        q_t_raw = q_func(obs_t_input.get(), 1, scope=qscope, reuse=True)
        targetTiled = tf.reshape(target_input.get(), shape=(-1,1))
        
        # calculate error
        td_error = q_t_raw - tf.stop_gradient(targetTiled)
        errors = importance_weights_ph.get() * U.huber_loss(td_error)

        # compute optimization op (potentially with gradient clipping)
        if grad_norm_clipping is not None:
            optimize_expr = U.minimize_and_clip(optimizer,
                                                errors,
                                                var_list=q_func_vars,
                                                clip_val=grad_norm_clipping)
        else:
            optimize_expr = optimizer.minimize(errors, var_list=q_func_vars)
        
        targetTrain = U.function(
            inputs=[
                obs_t_input,
                target_input,
                importance_weights_ph
            ],
            outputs=[td_error, obs_t_input.get(), target_input.get()],
            updates=[optimize_expr]
        )
    
        return targetTrain


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

    env = envstandalone.BlockArrange()

    # Standard q-learning parameters
    max_timesteps=50000
    exploration_fraction=0.3
    exploration_final_eps=0.1
    gamma=.90
    num_cpu = 16

    # Used by buffering and DQN
    learning_starts=10
    buffer_size=1
    batch_size=1
    target_network_update_freq=1
    train_freq=1
    print_freq=1
    lr=0.0003

    # first two elts of deicticShape must be odd
#    actionShape = (3,3,2)
    patchShape = (3,3,1)
    lookstackShape = (3,3,2)
    lookShape = (3,3,3)
    ppShape = (3,3,2)
#    num_states = 2 # either holding or not
    num_patches = env.maxSide**2
    num_actions_discrete = 2
    num_actions = num_patches + num_actions_discrete
    valueFunctionType = "DQN"
    actionSelectionStrategy = "UNIFORM_RANDOM" # actions are selected randomly from collection of all actions
#    actionSelectionStrategy = "RANDOM_UNIQUE" # each unique action descriptor has equal chance of being selected

    episode_rewards = [0.0]
    
    # Create the schedule for exploration starting from 1.
    exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * max_timesteps),
                                 initial_p=1.0,
                                 final_p=exploration_final_eps)

#    prioritized_replay=True
    prioritized_replay=False
#    prioritized_replay_alpha=1.0
    prioritized_replay_alpha=0.6
    prioritized_replay_beta0=0.4
    prioritized_replay_beta_iters=None
#    prioritized_replay_beta_iters=20000
    prioritized_replay_eps=1e-6
    if prioritized_replay:
        replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha=prioritized_replay_alpha)
        if prioritized_replay_beta_iters is None:
            prioritized_replay_beta_iters = max_timesteps
        beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                       initial_p=prioritized_replay_beta0,
                                       final_p=1.0)
    else:
        replay_buffer = ReplayBuffer(buffer_size)
        beta_schedule = None
    beta = 1

    q_func = models.cnn_to_mlp(
#    q_func = models.cnn_to_mlp_2pathways(
#        convs=[(16,3,1), (32,3,1)],
#        hiddens=[48],
        convs=[(32,3,1)],
        hiddens=[48],
#        convs=[(48,3,1)],
#        hiddens=[48],
        dueling=True
    )

    def displayLookStack(lookStack):
        np.set_printoptions(formatter={'float_kind':lambda x: "%.2f" % x})
        lookStack1 = str(lookStack[:,:,0])
        lookStack1 = np.core.defchararray.replace(lookStack1,".00","")
        lookStack1 = np.core.defchararray.replace(lookStack1,".","")
        lookStack1 = np.core.defchararray.replace(lookStack1,"0",".")
        lookStack2 = str(lookStack[:,:,1])
        lookStack2 = np.core.defchararray.replace(lookStack2,".00","")
        lookStack2 = np.core.defchararray.replace(lookStack2,".","")
        lookStack2 = np.core.defchararray.replace(lookStack2,"0",".")
        print("lookStack:")
        print(lookStack1)
        print(lookStack2)

    def make_obs_ph(name):
        return U.BatchInput(env.observation_space.spaces[0].shape, name=name)

    def make_lookDeic_ph(name):
        return U.BatchInput(lookShape, name=name)

    def make_ppDeic_ph(name):
        return U.BatchInput(ppShape, name=name)

    def make_target_ph(name):
        return U.BatchInput([1], name=name)

    def make_weight_ph(name):
        return U.BatchInput([1], name=name)

    getMoveActionDescriptors = build_getMoveActionDescriptors(make_obs_ph=make_obs_ph,deicticShape=lookShape)
    
    getqLookNotHolding = build_getq(
            make_deic_ph=make_lookDeic_ph,
            q_func=q_func,
            scope="deepq",
            qscope="q_func_LookNotHolding"
            )
    getqLookHolding = build_getq(
            make_deic_ph=make_lookDeic_ph,
            q_func=q_func,
            scope="deepq",
            qscope="q_func_LookHolding"
            )
    getqPPNotHolding = build_getq(
            make_deic_ph=make_ppDeic_ph,
            q_func=q_func,
            scope="deepq",
            qscope="q_func_PPNotHolding"
            )
    getqPPHolding = build_getq(
            make_deic_ph=make_ppDeic_ph,
            q_func=q_func,
            scope="deepq",
            qscope="q_func_PPHolding"
            )
    
    targetTrainLookNotHolding = build_targetTrain(
        make_deic_ph=make_lookDeic_ph,
        make_target_ph=make_target_ph,
        make_weight_ph=make_weight_ph,
        q_func=q_func,
        optimizer=tf.train.AdamOptimizer(learning_rate=lr),
        scope="deepq", 
        qscope="q_func_LookNotHolding",
        grad_norm_clipping=1.
    )
    targetTrainLookHolding = build_targetTrain(
        make_deic_ph=make_lookDeic_ph,
        make_target_ph=make_target_ph,
        make_weight_ph=make_weight_ph,
        q_func=q_func,
        optimizer=tf.train.AdamOptimizer(learning_rate=lr),
        scope="deepq", 
        qscope="q_func_LookHolding",
        grad_norm_clipping=1.
    )
    targetTrainPPNotHolding = build_targetTrain(
        make_deic_ph=make_ppDeic_ph,
        make_target_ph=make_target_ph,
        make_weight_ph=make_weight_ph,
        q_func=q_func,
        optimizer=tf.train.AdamOptimizer(learning_rate=lr),
        scope="deepq", 
        qscope="q_func_PPNotHolding",
        grad_norm_clipping=1.
    )
    targetTrainPPHolding = build_targetTrain(
        make_deic_ph=make_ppDeic_ph,
        make_target_ph=make_target_ph,
        make_weight_ph=make_weight_ph,
        q_func=q_func,
        optimizer=tf.train.AdamOptimizer(learning_rate=lr),
        scope="deepq", 
        qscope="q_func_PPHolding",
        grad_norm_clipping=1.
    )
        
    sess = U.make_session(num_cpu)
    sess.__enter__()

    obs = env.reset()
    lookStack = np.zeros(lookstackShape)
    lookStackNext = np.zeros(lookstackShape)
    
    episode_rewards = [0.0]
    td_errors = [0.0]
    timerStart = time.time()
    U.initialize()
    for t in range(max_timesteps):
        
        # Get action set: <num_patches> pick actions followed by <num_patches> place actions
        moveDescriptors = getMoveActionDescriptors([obs[0]])
        moveDescriptors = moveDescriptors*2-1
        moveDescriptors = np.reshape(moveDescriptors,[num_patches,patchShape[0],patchShape[1],patchShape[2]])
        looksStackTiled = np.tile(lookStack,[num_patches,1,1,1])
        lookDescriptors = np.concatenate([moveDescriptors,looksStackTiled],axis=3)
        
        if obs[1] == 0: # not holding
            qCurrLook = getqLookNotHolding(lookDescriptors)
            qCurrPP = np.r_[getqPPNotHolding([lookStack]),[[0]]]
        else: # holding
            qCurrLook = getqLookHolding(lookDescriptors)
            qCurrPP = np.r_[[[0]],getqPPHolding([lookStack])]
        qCurr = np.concatenate([qCurrLook,qCurrPP],axis=0)

        # select action at random
        qCurrNoise = qCurr + np.random.random(np.shape(qCurr))*0.01 # add small amount of noise to break ties randomly
        if actionSelectionStrategy == "UNIFORM_RANDOM":
            action = np.argmax(qCurrNoise)
            if np.random.rand() < exploration.value(t):
                actionClass = np.random.randint(3)
                if actionClass == 0:
                    action = np.random.randint(num_patches)
                else:
                    action = np.random.randint(num_patches,num_patches+2)
#                action = np.random.randint(num_actions)
        elif actionSelectionStrategy == "RANDOM_UNIQUE":
            _,idx,inv = np.unique(lookDescriptors,axis=0,return_index=True,return_inverse=True)
            idx = np.r_[idx,num_patches,num_patches+1]
            actionIdx = np.argmax(qCurrNoise[idx])
            if np.random.rand() < exploration.value(t):
                actionIdx = np.random.randint(len(idx))
            if actionIdx < len(idx)-2:
                actionsSelected = np.nonzero(inv==actionIdx)[0]
                action = actionsSelected[np.random.randint(len(actionsSelected))]
            else:
                action = idx[actionIdx]
        else:
            print("Error...")


        # take action
        new_obs, rew, done, _ = env.step(action)
        
        # If look action, then update look stack
        if action < num_patches:
            lookStackNext[:,:,1] = np.copy(lookStack[:,:,0])
            lookStackNext[:,:,0] = np.copy(moveDescriptors[action][:,:,0])
            lookAction = moveDescriptors[action]
            discreteAction = 0
        else:
            lookAction = np.zeros(patchShape)
            discreteAction = action - num_patches
        
        print("action: " + str(action))
        env.render()
        print("Reward: " + str(rew) + ", done: " + str(done))
        displayLookStack(lookStackNext)
        
        # discrete state, look state, discrete action, look action, reward, discrete next state, look next state, done
        replay_buffer.add(obs[1], lookStack, discreteAction, lookAction, rew, new_obs[1], lookStackNext, new_obs[0], float(done))
        
        lookStack = np.copy(lookStackNext)
        
        if t > learning_starts and t % train_freq == 0:

            # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
            if prioritized_replay:
                beta=beta_schedule.value(t)
                states_t, actionPatches, rewards, images_tp1, states_tp1, dones, weights, batch_idxes = replay_buffer.sample(batch_size, beta)
            else:
                statesHolding_t, statesLookStack_t, actionsDiscrete, lookActions, rewards, statesHolding_tp1, statesLookStack_tp1, observations_tp1, dones = replay_buffer.sample(batch_size)
                weights, batch_idxes = np.ones_like(rewards), None

            moveDescriptorsNext = getMoveActionDescriptors(observations_tp1)
            moveDescriptorsNext = moveDescriptorsNext*2-1
            moveDescriptorsNext = np.reshape(moveDescriptorsNext,[-1,patchShape[0],patchShape[1],patchShape[2]])
            looksStackNextTiled = np.repeat(statesLookStack_tp1,num_patches,axis=0)
            lookDescriptorsNext = np.concatenate([moveDescriptorsNext,looksStackNextTiled],axis=3)

            # calculate qNext
            qNextLookNotHolding = np.max(np.reshape(getqLookNotHolding(lookDescriptorsNext),[batch_size,num_patches,1]),axis=1)
            qNextLookHolding = np.max(np.reshape(getqLookHolding(lookDescriptorsNext),[batch_size,num_patches,1]),axis=1)
            qNextPPNotHolding = getqPPNotHolding(statesLookStack_tp1)
            qNextPPHolding = getqPPHolding(statesLookStack_tp1)
            qNextNotHolding = np.max(np.c_[qNextLookNotHolding,qNextPPNotHolding],axis=1)
            qNextHolding = np.max(np.c_[qNextLookHolding,qNextPPHolding],axis=1)
            qNext = np.stack([qNextNotHolding,qNextHolding],axis=1)

            targets = rewards + (1-dones) * gamma * qNext[range(batch_size),statesHolding_tp1]
            
            # Calculate qCurrTarget
            lookDescriptors = np.concatenate([lookActions,statesLookStack_t],axis=3)
            qCurrLookNotHoldingT = getqLookNotHolding(lookDescriptors)
            qCurrLookHoldingT = getqLookHolding(lookDescriptors)
            qCurrPPNotHoldingT = getqPPNotHolding(statesLookStack_t)
            qCurrPPHoldingT = getqPPHolding(statesLookStack_t)
            qCurrT = np.c_[qCurrLookNotHoldingT,qCurrPPNotHoldingT,qCurrLookHoldingT,qCurrPPHoldingT]
            
            td_error = qCurrT[range(batch_size),np.int32(actionsDiscrete > 0) + (2*statesHolding_t)] - targets
            qCurrT[range(batch_size),np.int32(actionsDiscrete > 0) + (2*statesHolding_t)] = targets

            targetTrainLookNotHolding(lookDescriptors,  np.reshape(qCurrT[:,0],[batch_size,1]), np.reshape(weights,[batch_size,1]))
            targetTrainPPNotHolding(statesLookStack_t, np.reshape(qCurrT[:,1],[batch_size,1]), np.reshape(weights,[batch_size,1]))
            targetTrainLookHolding(lookDescriptors, np.reshape(qCurrT[:,2],[batch_size,1]), np.reshape(weights,[batch_size,1]))
            targetTrainPPHolding(statesLookStack_t, np.reshape(qCurrT[:,3],[batch_size,1]), np.reshape(weights,[batch_size,1]))

            if prioritized_replay:
                new_priorities = np.abs(td_error) + prioritized_replay_eps
                replay_buffer.update_priorities(batch_idxes, new_priorities)

            td_errors[-1] += td_error


        # bookkeeping for storing episode rewards
        episode_rewards[-1] += rew
        if done:
            new_obs = env.reset()
            episode_rewards.append(0.0)
            td_errors.append(0.0)
        mean_100ep_reward = round(np.mean(episode_rewards[-51:-1]), 1)
        mean_100ep_tderror = round(np.mean(td_errors[-51:-1]), 1)
        num_episodes = len(episode_rewards)
        if done and print_freq is not None and len(episode_rewards) % print_freq == 0:
            timerFinal = time.time()
            print("steps: " + str(t) + ", episodes: " + str(num_episodes) + ", mean 100 episode reward: " + str(mean_100ep_reward) + ", % time spent exploring: " + str(int(100 * exploration.value(t))) + ", time elapsed: " + str(timerFinal - timerStart) + ", tderror: " + str(mean_100ep_tderror))
            timerStart = timerFinal
        
        obs = np.copy(new_obs)


    # display value function
    obs = env.reset()
    moveDescriptors = getMoveActionDescriptors([obs[0]])
    moveDescriptors = moveDescriptors*2-1

    actionsPickDescriptors = np.stack([moveDescriptors, np.zeros(np.shape(moveDescriptors))],axis=3)
    actionsPlaceDescriptors = np.stack([np.zeros(np.shape(moveDescriptors)), moveDescriptors],axis=3)
    
    print(str(obs[0][:,:,0]))
    
    if valueFunctionType == "TABULAR":
        qPick = getTabular(np.reshape(actionsPickDescriptors,[num_patches,-1])==1)
    else:
        qPickNotHolding = getqNotHolding(actionsPickDescriptors)
        qPickHolding = getqHolding(actionsPickDescriptors)
        qPick = np.concatenate([qPickNotHolding,qPickHolding],axis=1)
    print("Value function for pick action in hold-nothing state:")
    print(str(np.reshape(qPick[:,0],[8,8])))
    print("Value function for pick action in hold-1 state:")
    print(str(np.reshape(qPick[:,1],[8,8])))

    if valueFunctionType == "TABULAR":
        qPlace = getTabular(np.reshape(actionsPlaceDescriptors,[num_patches,-1])==1)
    else:
        qPlaceNotHolding = getqNotHolding(actionsPlaceDescriptors)
        qPlaceHolding = getqHolding(actionsPlaceDescriptors)
        qPlace = np.concatenate([qPlaceNotHolding,qPlaceHolding],axis=1)    
    print("Value function for place action in hold-nothing state:")
    print(str(np.reshape(qPlace[:,0],[8,8])))
    print("Value function for place action in hold-1 state:")
    print(str(np.reshape(qPlace[:,1],[8,8])))
    


if __name__ == '__main__':
    main()


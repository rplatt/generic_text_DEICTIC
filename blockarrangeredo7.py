#
# This modifies blockarrangeredo6 by adding individually cascaded networks.
#
# Adapted from blockarrangeredo6.py
#
# Results: I'm not sure if this works -- experimenting w/ uniform random actions...
#
#
#
import gym
import numpy as np
import time as time
import tensorflow as tf
import tf_util_rob as U
import models as models
from replay_buffer7 import ReplayBuffer, PrioritizedReplayBuffer
from schedules import LinearSchedule
import copy as copy

#import envs.blockarrange_3blocks as envstandalone
import envs.blockarrange_look2 as envstandalone

# **** Make tensorflow functions ****

def build_getq(make_deic_ph, q_func, num_states, num_cascade, scope="deepq", qscope="q_func", reuse=None):

    with tf.variable_scope(scope, reuse=reuse):

        actions_ph = U.ensure_tf_input(make_deic_ph("actions"))
        q_values = q_func(actions_ph.get(), 1, scope=qscope)
#        q_values = q_func(actions_ph.get(), num_cascade, scope=qscope)
        getq = U.function(inputs=[actions_ph], outputs=q_values)
        return getq


def build_targetTrain(make_deic_ph,
                        make_target_ph,
                        make_weight_ph,
                        q_func,
                        num_states,
                        num_cascade,
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
#        q_t_raw = q_func(obs_t_input.get(), num_cascade, scope=qscope, reuse=True)
#        targetTiled = tf.reshape(target_input.get(), shape=(-1,num_cascade))
        
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
    max_timesteps=800
    exploration_fraction=0.3
    exploration_final_eps=0.1
    gamma=.90
    num_cpu = 16

    # Used by buffering and DQN
    learning_starts=100
    buffer_size=1000
    batch_size=10
    target_network_update_freq=1
    train_freq=1
    print_freq=1
    lr=0.0003

    # first two elts of deicticShape must be odd
    actionShape = (3,3,3)
    memoryShape = (3,3,3)
    stateActionShape = (3,3,6) # includes place memory
    num_states = 2 # either holding or not
    num_patches = env.maxSide**2
    num_actions_discrete = 3 # pick/place/look
    num_actions = num_actions_discrete*num_patches
    num_cascade = 3
#    valueFunctionType = "TABULAR"
    valueFunctionType = "DQN"
#    actionSelectionStrategy = "UNIFORM_RANDOM" # actions are selected randomly from collection of all actions
    actionSelectionStrategy = "RANDOM_UNIQUE" # each unique action descriptor has equal chance of being selected

    DEBUG = False
#    DEBUG = True

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

    def make_obs_ph(name):
        return U.BatchInput(env.observation_space.spaces[0].shape, name=name)

    def make_deic_ph(name):
        return U.BatchInput(stateActionShape, name=name)

    def make_target_ph(name):
        return U.BatchInput([1], name=name)
#        return U.BatchInput([num_cascade], name=name)

    def make_weight_ph(name):
        return U.BatchInput([1], name=name)
#        return U.BatchInput([num_cascade], name=name)

    getMoveActionDescriptors = build_getMoveActionDescriptors(make_obs_ph=make_obs_ph,deicticShape=actionShape)
    
    if valueFunctionType == 'DQN':
        getqNotHolding1 = build_getq(make_deic_ph=make_deic_ph,q_func=q_func,num_states=num_states,num_cascade=num_cascade,scope="deepq",qscope="q_func_notholding")
        getqHolding1 = build_getq(make_deic_ph=make_deic_ph,q_func=q_func,num_states=num_states,num_cascade=num_cascade,scope="deepq",qscope="q_func_holding")
        targetTrainNotHolding1 = build_targetTrain(make_deic_ph=make_deic_ph,make_target_ph=make_target_ph,make_weight_ph=make_weight_ph,q_func=q_func,num_states=num_states,num_cascade=num_cascade,optimizer=tf.train.AdamOptimizer(learning_rate=lr),scope="deepq", qscope="q_func_notholding",grad_norm_clipping=1.)
        targetTrainHolding1 = build_targetTrain(make_deic_ph=make_deic_ph,make_target_ph=make_target_ph,make_weight_ph=make_weight_ph,q_func=q_func,num_states=num_states,num_cascade=num_cascade,optimizer=tf.train.AdamOptimizer(learning_rate=lr),scope="deepq",qscope="q_func_holding",grad_norm_clipping=1.)
        
#        getqNotHolding2 = build_getq(make_deic_ph=make_deic_ph,q_func=q_func,num_states=num_states,num_cascade=num_cascade,scope="deepq2",qscope="q_func_notholding2")
#        getqHolding2 = build_getq(make_deic_ph=make_deic_ph,q_func=q_func,num_states=num_states,num_cascade=num_cascade,scope="deepq2",qscope="q_func_holding2")
#        targetTrainNotHolding2 = build_targetTrain(make_deic_ph=make_deic_ph,make_target_ph=make_target_ph,make_weight_ph=make_weight_ph,q_func=q_func,num_states=num_states,num_cascade=num_cascade,optimizer=tf.train.AdamOptimizer(learning_rate=lr),scope="deepq2", qscope="q_func_notholding2",grad_norm_clipping=1.)
#        targetTrainHolding2 = build_targetTrain(make_deic_ph=make_deic_ph,make_target_ph=make_target_ph,make_weight_ph=make_weight_ph,q_func=q_func,num_states=num_states,num_cascade=num_cascade,optimizer=tf.train.AdamOptimizer(learning_rate=lr),scope="deepq2",qscope="q_func_holding2",grad_norm_clipping=1.)
#        
#        getqNotHolding3 = build_getq(make_deic_ph=make_deic_ph,q_func=q_func,num_states=num_states,num_cascade=num_cascade,scope="deepq3",qscope="q_func_notholding3")
#        getqHolding3 = build_getq(make_deic_ph=make_deic_ph,q_func=q_func,num_states=num_states,num_cascade=num_cascade,scope="deepq3",qscope="q_func_holding3")
#        targetTrainNotHolding3 = build_targetTrain(make_deic_ph=make_deic_ph,make_target_ph=make_target_ph,make_weight_ph=make_weight_ph,q_func=q_func,num_states=num_states,num_cascade=num_cascade,optimizer=tf.train.AdamOptimizer(learning_rate=lr),scope="deepq3", qscope="q_func_notholding3",grad_norm_clipping=1.)
#        targetTrainHolding3 = build_targetTrain(make_deic_ph=make_deic_ph,make_target_ph=make_target_ph,make_weight_ph=make_weight_ph,q_func=q_func,num_states=num_states,num_cascade=num_cascade,optimizer=tf.train.AdamOptimizer(learning_rate=lr),scope="deepq3",qscope="q_func_holding3",grad_norm_clipping=1.)
        
    sess = U.make_session(num_cpu)
    sess.__enter__()

    obs = copy.deepcopy(env.reset())
    grid_t = obs[0]
#    grid_t = np.int32(obs[0]>0)
    stateHolding_t = np.int32(obs[1] > 0)
    memory_t = np.zeros([1, memoryShape[0], memoryShape[1], memoryShape[2]]) # first col is pick, second is place, third is look
#    memory_t[0,:,:,2] = (env.pickBlockGoal + 2) * np.ones([memoryShape[1], memoryShape[2]]) # DEBUG
    
    episode_rewards = [0.0]
    timerStart = time.time()
    U.initialize()

    if DEBUG:
        saver = tf.train.Saver()
        saver.restore(sess, "./temp")

    for t in range(max_timesteps):

        # Get state/action descriptors
        moveDescriptors = getMoveActionDescriptors([grid_t])
        moveDescriptors[moveDescriptors == 0] = -1
        actionsPickDescriptors = np.stack([moveDescriptors, np.zeros(np.shape(moveDescriptors)), np.zeros(np.shape(moveDescriptors))],axis=3)
        actionsPlaceDescriptors = np.stack([np.zeros(np.shape(moveDescriptors)),moveDescriptors, np.zeros(np.shape(moveDescriptors))],axis=3)
        actionsLookDescriptors = np.stack([np.zeros(np.shape(moveDescriptors)), np.zeros(np.shape(moveDescriptors)), moveDescriptors],axis=3)
        actionDescriptors = np.r_[actionsPickDescriptors,actionsPlaceDescriptors,actionsLookDescriptors]
        memoryTiled = np.repeat(memory_t,num_patches*num_actions_discrete,axis=0)
        stateActionDescriptors = np.concatenate([actionDescriptors, memoryTiled],axis=3)

        # Get current values
        qCurrNotHolding = getqNotHolding1(stateActionDescriptors)
        qCurrHolding = getqHolding1(stateActionDescriptors)
#        qCurrNotHolding = getqNotHolding3(stateActionDescriptors)
#        qCurrHolding = getqHolding3(stateActionDescriptors)
        qCurr = np.concatenate([qCurrNotHolding,qCurrHolding],axis=1)

        # Select action
        qCurrNoise = qCurr + np.random.random(np.shape(qCurr))*0.01 # add small amount of noise to break ties randomly
        if actionSelectionStrategy == "UNIFORM_RANDOM":
            action = np.argmax(qCurrNoise[:,stateHolding_t])
            if np.random.rand() < exploration.value(t):
                action = np.random.randint(num_actions)
        elif actionSelectionStrategy == "RANDOM_UNIQUE":
            _,idx,inv = np.unique(actionDescriptors,axis=0,return_index=True,return_inverse=True)
            actionIdx = np.argmax(qCurrNoise[idx,stateHolding_t])
            
#            if not DEBUG:
#                if np.random.rand() < exploration.value(t):
            actionIdx = np.random.randint(len(idx))
            
            actionsSelected = np.nonzero(inv==actionIdx)[0]
            action = actionsSelected[np.random.randint(len(actionsSelected))]
        else:
            print("Error...")

        # Take action
        new_obs, rew, done, _ = env.step(action)
        
        # Update state and memory
        grid_tp1 = new_obs[0]
#        grid_tp1 = np.int32(new_obs[0]>0)
        stateHolding_tp1= np.int32(new_obs[1] > 0)
        memory_tp1 = np.copy(memory_t)
        if (stateHolding_t == 0) and (stateHolding_tp1 != 0): # if a block has been picked
            memory_tp1[:,:,:,0] = np.reshape(stateActionDescriptors[action][:,:,0],[1,stateActionShape[0],stateActionShape[1]])
        if (stateHolding_t > 0) and (stateHolding_tp1 == 0): # if a block has just been placed
            memory_tp1[:,:,:,1] = np.reshape(stateActionDescriptors[action][:,:,1],[1,stateActionShape[0],stateActionShape[1]])
        if action > num_patches*2: # if this is a look action
#            memory_tp1[:,:,:,2] = np.reshape(stateActionDescriptors[action][:,:,2],[1,stateActionShape[0],stateActionShape[1]])
#            memory_tp1[0,:,:,2] = (env.pickBlockGoal + 2) * np.ones([memoryShape[1], memoryShape[2]]) # DEBUG
            if (env.pickBlockGoal + 2) in stateActionDescriptors[action][:,:,2]:
                memory_tp1[0,:,:,2] = (env.pickBlockGoal + 2) * np.ones([memoryShape[1], memoryShape[2]])

        if DEBUG:
            env.render()
            print("memory: ")
            print(str(memory_tp1))
            print("action: " + str(action))
            print("action descriptor:")
            if action < num_patches:
                print(stateActionDescriptors[action][:,:,0])
            elif action < 2*num_patches:
                print(stateActionDescriptors[action][:,:,1])
            else:
                print(stateActionDescriptors[action][:,:,2])

#        memory_tp1[0,:,:,2] = (env.pickBlockGoal + 2) * np.ones([memoryShape[1], memoryShape[2]]) # DEBUG

        # Add to replay buffer
        replay_buffer.add(stateHolding_t, stateActionDescriptors[action,:], rew, stateHolding_tp1, grid_tp1, memory_tp1[0], done)
        
        # handle end of episode
        if done:
            new_obs = env.reset()
            grid_tp1 = new_obs[0]
            stateHolding_tp1= np.int32(new_obs[1] > 0)
            memory_tp1 = np.zeros([1, memoryShape[0], memoryShape[1], memoryShape[2]])
#            memory_tp1[0,:,:,2] = (env.pickBlockGoal + 2) * np.ones([memoryShape[1], memoryShape[2]]) # DEBUG

        # Set tp1 equal to t
        stateHolding_t = stateHolding_tp1
        grid_t = grid_tp1
        memory_t = memory_tp1
        
        
        if t > learning_starts and t % train_freq == 0:

            # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
            if prioritized_replay:
                beta=beta_schedule.value(t)
                states_t, actionPatches, rewards, images_tp1, states_tp1, placeMemory_tp1, dones, weights, batch_idxes = replay_buffer.sample(batch_size, beta)
            else:
                statesDiscrete_t, stateActionsImage_t, rewards, statesDiscrete_tp1, grids_tp1, memories_tp1, dones = replay_buffer.sample(batch_size)
                weights, batch_idxes = np.ones_like(rewards), None

            moveDescriptorsNext = getMoveActionDescriptors(grids_tp1)
            moveDescriptorsNext[moveDescriptorsNext == 0] = -1
            
            actionsPickDescriptorsNext = np.stack([moveDescriptorsNext, np.zeros(np.shape(moveDescriptorsNext)), np.zeros(np.shape(moveDescriptorsNext))],axis=3)
            actionsPlaceDescriptorsNext = np.stack([np.zeros(np.shape(moveDescriptorsNext)), moveDescriptorsNext, np.zeros(np.shape(moveDescriptorsNext))],axis=3)
            actionsLookDescriptorsNext = np.stack([np.zeros(np.shape(moveDescriptorsNext)), np.zeros(np.shape(moveDescriptorsNext)), moveDescriptorsNext],axis=3)
            actionDescriptorsNext = np.stack([actionsPickDescriptorsNext, actionsPlaceDescriptorsNext, actionsLookDescriptorsNext], axis=1) # I sometimes get this axis parameter wrong... pay attention!
            actionDescriptorsNext = np.reshape(actionDescriptorsNext,[batch_size*num_patches*num_actions_discrete,actionShape[0],actionShape[1],actionShape[2]])

            # Augment with state, i.e. place memory
            placeMemory_tp1_expanded = np.repeat(memories_tp1,num_patches*num_actions_discrete,axis=0)
            actionDescriptorsNext = np.concatenate([actionDescriptorsNext, placeMemory_tp1_expanded],axis=3)
            
            qNextNotHolding = getqNotHolding1(actionDescriptorsNext)
            qNextHolding = getqHolding1(actionDescriptorsNext)
            qNextFlat = np.concatenate([qNextNotHolding,qNextHolding],axis=1)

            qNext = np.reshape(qNextFlat,[batch_size,num_patches,num_actions_discrete,num_states])
            qNextmax = np.max(np.max(qNext[range(batch_size),:,:,statesDiscrete_tp1],2),1)
            targets = rewards + (1-dones) * gamma * qNextmax
            
            if any(targets > 11):
                targets
            
            if t > 750:
                qNext

            # avg value
            qCurrTargetNotHolding = getqNotHolding1(stateActionsImage_t)
            qCurrTargetHolding = getqHolding1(stateActionsImage_t)
            qCurrTarget = np.concatenate([qCurrTargetNotHolding,qCurrTargetHolding],axis=1)
            qCurrTarget[range(batch_size),statesDiscrete_t] = targets
            targetTrainNotHolding1(stateActionsImage_t, np.reshape(qCurrTarget[:,0],[batch_size,1]), np.reshape(weights,[batch_size,1]))
            targetTrainHolding1(stateActionsImage_t, np.reshape(qCurrTarget[:,1],[batch_size,1]), np.reshape(weights,[batch_size,1]))

#            # cascaded value
#            qCurrTargetNotHolding1 = getqNotHolding1(stateActionsImage_t)
#            qCurrTargetHolding1 = getqHolding1(stateActionsImage_t)
#            qCurrTarget1 = np.concatenate([qCurrTargetNotHolding1,qCurrTargetHolding1],axis=1)
#            qCurrTargetNotHolding2 = getqNotHolding2(stateActionsImage_t)
#            qCurrTargetHolding2 = getqHolding2(stateActionsImage_t)
#            qCurrTarget2 = np.concatenate([qCurrTargetNotHolding2,qCurrTargetHolding2],axis=1)
#            qCurrTargetNotHolding3 = getqNotHolding3(stateActionsImage_t)
#            qCurrTargetHolding3 = getqHolding3(stateActionsImage_t)
#            qCurrTarget3 = np.concatenate([qCurrTargetNotHolding3,qCurrTargetHolding3],axis=1)
#            
#            mask2Idx = np.nonzero(targets < qCurrTarget1[range(batch_size),statesDiscrete_t])[0]
#            mask3Idx = np.nonzero(targets < qCurrTarget2[range(batch_size),statesDiscrete_t])[0]
#            qCurrTarget1[range(batch_size),statesDiscrete_t] = targets
#            qCurrTarget2[mask2Idx,statesDiscrete_t[mask2Idx]] = targets[mask2Idx]
#            qCurrTarget3[mask3Idx,statesDiscrete_t[mask3Idx]] = targets[mask3Idx]
#            
#            targetTrainNotHolding1(stateActionsImage_t, np.reshape(qCurrTarget1[:,0],[batch_size,1]), np.ones([batch_size,1]))
#            targetTrainHolding1(stateActionsImage_t, np.reshape(qCurrTarget1[:,1],[batch_size,1]), np.ones([batch_size,1]))
#            targetTrainNotHolding2(stateActionsImage_t, np.reshape(qCurrTarget2[:,0],[batch_size,1]), np.ones([batch_size,1]))
#            targetTrainHolding2(stateActionsImage_t, np.reshape(qCurrTarget2[:,1],[batch_size,1]), np.ones([batch_size,1]))
#            targetTrainNotHolding3(stateActionsImage_t, np.reshape(qCurrTarget3[:,0],[batch_size,1]), np.ones([batch_size,1]))
#            targetTrainHolding3(stateActionsImage_t, np.reshape(qCurrTarget3[:,1],[batch_size,1]), np.ones([batch_size,1]))

            if prioritized_replay:
                new_priorities = np.abs(td_error) + prioritized_replay_eps
                replay_buffer.update_priorities(batch_idxes, new_priorities)



        # bookkeeping for storing episode rewards
        episode_rewards[-1] += rew
        if done:
#            new_obs = env.reset()
            episode_rewards.append(0.0)
        mean_100ep_reward = round(np.mean(episode_rewards[-51:-1]), 1)
        num_episodes = len(episode_rewards)
        if done and print_freq is not None and len(episode_rewards) % print_freq == 0:
            timerFinal = time.time()
            print("steps: " + str(t) + ", episodes: " + str(num_episodes) + ", mean 100 episode reward: " + str(mean_100ep_reward) + ", % time spent exploring: " + str(int(100 * exploration.value(t))) + ", time elapsed: " + str(timerFinal - timerStart))
            timerStart = timerFinal
        
        obs = new_obs

    saver = tf.train.Saver()
    saver.save(sess, "./temp")

    # display value function
    obs = env.reset()
    moveDescriptors = getMoveActionDescriptors([obs[0]])
    moveDescriptors[moveDescriptors == 0] = -1
    actionsPickDescriptorsOrig = np.stack([moveDescriptors, np.zeros(np.shape(moveDescriptors)), np.zeros(np.shape(moveDescriptors))],axis=3)
    actionsLookDescriptorsOrig = np.stack([np.zeros(np.shape(moveDescriptors)), np.zeros(np.shape(moveDescriptors)), moveDescriptors],axis=3)
    
    memoryZeros = np.zeros([1, memoryShape[0], memoryShape[1], memoryShape[2]])
    memoryLooked3 = np.zeros([1, memoryShape[0], memoryShape[1], memoryShape[2]])
    memoryLooked3[0,:,:,2] = 3*np.ones([stateActionShape[0], stateActionShape[1]])
    memoryLooked4 = np.zeros([1, memoryShape[0], memoryShape[1], memoryShape[2]])
    memoryLooked4[0,:,:,2] = 4*np.ones([stateActionShape[0], stateActionShape[1]])
    
    print("\nGrid configuration:")
    print(str(obs[0][:,:,0]))
        
    for i in range(3):
        
        if i == 0:
            placeMemory = memoryZeros
            print("\nMemory has zeros:")
        elif i==1:
            placeMemory = memoryLooked3
            print("\nMemory encodes look=3:")
        else:
            placeMemory = memoryLooked4
            print("\nMemory encodes look=4:")
            
        placeMemoryTiled = np.repeat(placeMemory,num_patches,axis=0)
        actionsPickDescriptors = np.concatenate([actionsPickDescriptorsOrig, placeMemoryTiled],axis=3)
        actionsLookDescriptors = np.concatenate([actionsLookDescriptorsOrig, placeMemoryTiled],axis=3)
    
        qPickNotHolding1 = getqNotHolding1(actionsPickDescriptors)
        qLookNotHolding1 = getqNotHolding1(actionsLookDescriptors)
#        qPickNotHolding2 = getqNotHolding2(actionsPickDescriptors)
#        qLookNotHolding2 = getqNotHolding2(actionsLookDescriptors)
#        qPickNotHolding3 = getqNotHolding3(actionsPickDescriptors)
#        qLookNotHolding3 = getqNotHolding3(actionsLookDescriptors)
        
        print("\nValue function for pick action in hold-nothing state:")
        print(str(np.reshape(qPickNotHolding1,[8,8])))
#        print("***")
#        print(str(np.reshape(qPickNotHolding2,[8,8])))
#        print("***")
#        print(str(np.reshape(qPickNotHolding3,[8,8])))
    
        print("\nValue function for look action in hold-nothing state:")
        print(str(np.reshape(qLookNotHolding1,[8,8])))
#        print("***")
#        print(str(np.reshape(qLookNotHolding2,[8,8])))
#        print("***")
#        print(str(np.reshape(qLookNotHolding3,[8,8])))
        


if __name__ == '__main__':
    main()


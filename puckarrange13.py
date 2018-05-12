#
# This version uses a cascaded value function for NoRot
#
# Adapted from puckarrange12.py
#
# Results: I don't think the cascade helps. I'm still trying to verify this,
#          but, I think just a single level (num_cascade=1) works best...
#
#
#
import sys as sys
import gym
import numpy as np
import time as time
import tensorflow as tf
import tf_util_rob as U
import models as models
#from replay_buffer8 import ReplayBuffer, PrioritizedReplayBuffer
from replay_buffer9 import ReplayBuffer, PrioritizedReplayBuffer
from schedules import LinearSchedule
import matplotlib.pyplot as plt
import copy as cp

# Two disks placed in a 224x224 image. Disks placed randomly initially. 
# Reward given when the pucks are placed adjacent. Agent must learn to pick
# up one of the disks and place it next to the other.
import envs.puckarrange_env13 as envstandalone


# **** Make tensorflow functions ****

# Evaluate the q function for a given input.
def build_getq(make_actionDeic_ph, q_func, num_states, num_cascade, scope="deepq", qscope="q_func", reuse=None):

    with tf.variable_scope(scope, reuse=reuse):
        actions_ph = U.ensure_tf_input(make_actionDeic_ph("actions"))
#        q_values = q_func(actions_ph.get(), 1, scope=qscope)
#        getq = U.function(inputs=[actions_ph], outputs=q_values)
        q_values = q_func(actions_ph.get(), num_cascade, scope=qscope)
        q_valuesTiled = tf.reshape(q_values,[-1,num_cascade])
        getq = U.function(inputs=[actions_ph], outputs=q_valuesTiled)
        return getq



# Train q-function
def build_targetTrain(make_actionDeic_ph,
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
        obs_t_input = U.ensure_tf_input(make_actionDeic_ph("action_t_deic"))
        target_input = U.ensure_tf_input(make_target_ph("target"))
        importance_weights_ph = U.ensure_tf_input(make_weight_ph("target"))

        # get variables
        q_func_vars = U.scope_vars(U.absolute_scope_name(qscope))
    
        # q values for all actions
#        q_t_raw = q_func(obs_t_input.get(), 1, scope=qscope, reuse=True)
#        targetTiled = tf.reshape(target_input.get(), shape=(-1,1))
        q_t_raw = q_func(obs_t_input.get(), num_cascade, scope=qscope, reuse=True)
        targetTiled = tf.reshape(target_input.get(), shape=(-1,num_cascade))

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

# Get candidate move descriptors given an input image. Candidates are found by
# sliding a window over the image with the given stride (same stride applied both 
# in x and y)
def build_getMoveActionDescriptors(make_obs_ph,actionShape,actionShapeSmall,stride):
    
    observations_ph = U.ensure_tf_input(make_obs_ph("observation"))
    obs = observations_ph.get()
    shape = tf.shape(obs)
    deicticPad = np.int32(2*np.floor(np.array(actionShape)/3))
    obsZeroPadded = tf.image.resize_image_with_crop_or_pad(obs,shape[1]+deicticPad[0],shape[2]+deicticPad[1])
    patches = tf.extract_image_patches(
            obsZeroPadded,
            ksizes=[1, actionShape[0], actionShape[1], 1],
            strides=[1, stride, stride, 1], 
            rates=[1, 1, 1, 1], 
            padding='VALID')
    patchesShape = tf.shape(patches)
    patchesTiled = tf.reshape(patches,[patchesShape[0]*patchesShape[1]*patchesShape[2],actionShape[0],actionShape[1],1])
    patchesTiledSmall = tf.image.resize_images(patchesTiled, [actionShapeSmall[0], actionShapeSmall[1]])
    patchesTiledSmall = tf.reshape(patchesTiledSmall,[-1,actionShapeSmall[0],actionShapeSmall[1]])

    getMoveActionDescriptors = U.function(inputs=[observations_ph], outputs=patchesTiledSmall)
    return getMoveActionDescriptors


# Get candidate move descriptors given an input image. Candidates are found by
# sliding a window over the image with the given stride (same stride applied both 
# in x and y)
def build_getMoveActionDescriptorsRot(make_obs_ph,actionShape,actionShapeSmall,stride):
    
    observations_ph = U.ensure_tf_input(make_obs_ph("observation"))
    obs = observations_ph.get()
    shape = tf.shape(obs)
    deicticPad = np.int32(2*np.floor(np.array(actionShape)/3))
    obsZeroPadded = tf.image.resize_image_with_crop_or_pad(obs,shape[1]+deicticPad[0],shape[2]+deicticPad[1])
    patches = tf.extract_image_patches(
            obsZeroPadded,
            ksizes=[1, actionShape[0], actionShape[1], 1],
            strides=[1, stride, stride, 1], 
            rates=[1, 1, 1, 1], 
            padding='VALID')
    patchesShape = tf.shape(patches)
    patchesTiled = tf.reshape(patches,[patchesShape[0]*patchesShape[1]*patchesShape[2],actionShape[0],actionShape[1],1])
    
    patchesTiledRot0 = patchesTiled
    patchesTiledRot1 = tf.contrib.image.rotate(patchesTiled,np.pi/4)
    patchesTiledRot2 = tf.contrib.image.rotate(patchesTiled,2*np.pi/4)
    patchesTiledRot3 = tf.contrib.image.rotate(patchesTiled,3*np.pi/4)

#    patchesTiledAll = tf.concat([patchesTiled,patchesTiled,patchesTiled,patchesTiled],axis=0)
#    patchesTiledAll = tf.concat([patchesTiledRot0,patchesTiledRot0,patchesTiledRot2,patchesTiledRot2],axis=0)
#    patchesTiledAll = tf.concat([patchesTiledRot1,patchesTiledRot1,patchesTiledRot3,patchesTiledRot3],axis=0)
    patchesTiledAll = tf.concat([patchesTiledRot0,patchesTiledRot1,patchesTiledRot2,patchesTiledRot3],axis=0)
    
    patchesTiledSmall = tf.image.resize_images(patchesTiledAll, [actionShapeSmall[0], actionShapeSmall[1]])
    patchesTiledSmall = tf.reshape(patchesTiledSmall,[-1,actionShapeSmall[0],actionShapeSmall[1]])

    getMoveActionDescriptors = U.function(inputs=[observations_ph], outputs=patchesTiledSmall)
    return getMoveActionDescriptors



def main(initEnvStride, envStride, fileIn, fileOut, inputmaxtimesteps, vispolicy):

    np.set_printoptions(formatter={'float_kind':lambda x: "%.2f" % x})

    # Create environment and set stride parameters for this problem instance.
    # Most of the time, these two stride parameters will be equal. However,
    # one might use a smaller stride for initial placement and a larger stride
    # for action specification in order to speed things up. Unfortunately, this
    # could cause the problem to be infeasible: no grasp might work for a given
    # initial setup.
    env = envstandalone.PuckArrange()
    env.initStride = initEnvStride # stride for initial puck placement
    env.stride = envStride # stride for action specification
    
    # Standard q-learning parameters
    reuseModels = None
    max_timesteps=inputmaxtimesteps
    exploration_fraction=0.5
    exploration_final_eps=0.1
    gamma=.90
    num_cpu = 16

    # Used by buffering and DQN
    learning_starts=60
    buffer_size=1000
#    batch_size=32
    batch_size=10
    target_network_update_freq=1
    train_freq=1
    print_freq=1
    lr=0.0003

#    useHierarchy = False
    useHierarchy = True
#    num_cascade = 3 # cascade used to get max value for NoRot q-function
    num_cascade = 1 # no cascade

    # Set parameters related to shape of the patch and the number of patches
    descriptorShape = (env.blockSize*3,env.blockSize*3,2)
#    descriptorShapeSmall = (10,10,2)
#    descriptorShapeSmall = (15,15,2)
    descriptorShapeSmall = (20,20,2)
    num_states = 2 # either holding or not
    num_patches = len(env.moveCenters)**2
    num_actions = 2*num_patches*env.num_orientations
    
    # Create the schedule for exploration starting from 1.
    exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * max_timesteps),
                                 initial_p=1.0,
                                 final_p=exploration_final_eps)
#    exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * max_timesteps),
#                                 initial_p=exploration_final_eps,
#                                 final_p=exploration_final_eps)

    # Set parameters for prioritized replay. You  can turn this off just by 
    # setting the line below to False
#    prioritized_replay=True
    prioritized_replay=False
    prioritized_replay_alpha=0.6
    prioritized_replay_beta0=0.4
    prioritized_replay_beta_iters=None
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

    # Create neural network
    q_func = models.cnn_to_mlp(
        convs=[(16,3,1)],
        hiddens=[32],
        dueling=True
    )

    # Build tensorflow functions
    def make_obs_ph(name):
        return U.BatchInput(env.observation_space.spaces[0].shape, name=name)
    def make_actionDeic_ph(name):
        return U.BatchInput(descriptorShapeSmall, name=name)
    def make_target_ph1(name):
        return U.BatchInput([1], name=name)
    def make_weight_ph1(name):
        return U.BatchInput([1], name=name)
    def make_target_phCascade(name):
        return U.BatchInput([num_cascade], name=name)
    def make_weight_phCascade(name):
        return U.BatchInput([num_cascade], name=name)

    getMoveActionDescriptorsNoRot = build_getMoveActionDescriptors(make_obs_ph=make_obs_ph,actionShape=descriptorShape,actionShapeSmall=descriptorShapeSmall,stride=env.stride)
    getMoveActionDescriptorsRot = build_getMoveActionDescriptorsRot(make_obs_ph=make_obs_ph,actionShape=descriptorShape,actionShapeSmall=descriptorShapeSmall,stride=env.stride)
    
    getqNotHoldingRot = build_getq(
            make_actionDeic_ph=make_actionDeic_ph,
            q_func=q_func,
            num_states=num_states,
            num_cascade=1,
            scope="deepq",
            qscope="q_func_notholding_rot",
            reuse=reuseModels
            )
    getqHoldingRot = build_getq(
            make_actionDeic_ph=make_actionDeic_ph,
            q_func=q_func,
            num_states=num_states,
            num_cascade=1,
            scope="deepq",
            qscope="q_func_holding_rot",
            reuse=reuseModels
            )

    targetTrainNotHoldingRot = build_targetTrain(
        make_actionDeic_ph=make_actionDeic_ph,
        make_target_ph=make_target_ph1,
        make_weight_ph=make_weight_ph1,
        q_func=q_func,
        num_states=num_states,
        num_cascade=1,
        optimizer=tf.train.AdamOptimizer(learning_rate=lr),
        scope="deepq", 
        qscope="q_func_notholding_rot",
        grad_norm_clipping=1.,
        reuse=reuseModels
    )

    targetTrainHoldingRot = build_targetTrain(
        make_actionDeic_ph=make_actionDeic_ph,
        make_target_ph=make_target_ph1,
        make_weight_ph=make_weight_ph1,
        q_func=q_func,
        num_states=num_states,
        num_cascade=1,
        optimizer=tf.train.AdamOptimizer(learning_rate=lr),
        scope="deepq", 
        qscope="q_func_holding_rot",
        grad_norm_clipping=1.,
        reuse=reuseModels
    )
    
    getqNotHoldingNoRot = build_getq(
            make_actionDeic_ph=make_actionDeic_ph,
            q_func=q_func,
            num_states=num_states,
            num_cascade=num_cascade,
            scope="deepq",
            qscope="q_func_notholding_norot",
            reuse=reuseModels
            )
    getqHoldingNoRot = build_getq(
            make_actionDeic_ph=make_actionDeic_ph,
            q_func=q_func,
            num_states=num_states,
            num_cascade=num_cascade,
            scope="deepq",
            qscope="q_func_holding_norot",
            reuse=reuseModels
            )

    targetTrainNotHoldingNoRot = build_targetTrain(
        make_actionDeic_ph=make_actionDeic_ph,
        make_target_ph=make_target_phCascade,
        make_weight_ph=make_weight_phCascade,
        q_func=q_func,
        num_states=num_states,
        num_cascade=num_cascade,
        optimizer=tf.train.AdamOptimizer(learning_rate=lr),
        scope="deepq", 
        qscope="q_func_notholding_norot",
        grad_norm_clipping=1.,
        reuse=reuseModels
    )

    targetTrainHoldingNoRot = build_targetTrain(
        make_actionDeic_ph=make_actionDeic_ph,
        make_target_ph=make_target_phCascade,
        make_weight_ph=make_weight_phCascade,
        q_func=q_func,
        num_states=num_states,
        num_cascade=num_cascade,
        optimizer=tf.train.AdamOptimizer(learning_rate=lr),
        scope="deepq", 
        qscope="q_func_holding_norot",
        grad_norm_clipping=1.,
        reuse=reuseModels
    )

    # Initialize tabular state-value function. There are only two states (holding, not holding), so this is very easy.
    lrState = 0.1
    V = np.zeros([2,])
    
    # Start tensorflow session
    sess = U.make_session(num_cpu)
    sess.__enter__()

    # Initialize things
    obs = env.reset()
    episode_rewards = [0.0]
    timerStart = time.time()
    U.initialize()
    
    # Load neural network model if one was specified.
    if fileIn != "None":
        saver = tf.train.Saver()
        saver.restore(sess, fileIn)
        fileInV = fileIn + 'V.npy'
        V = np.load(fileInV)

    # Iterate over time steps
    for t in range(max_timesteps):
        
        # Use hierarchy to get candidate actions
        if useHierarchy:
        
            # Get NoRot descriptors
            moveDescriptorsNoRot = getMoveActionDescriptorsNoRot([obs[0]])
            moveDescriptorsNoRot = moveDescriptorsNoRot*2-1
            actionsPickDescriptorsNoRot = np.stack([moveDescriptorsNoRot, np.zeros(np.shape(moveDescriptorsNoRot))],axis=3)
            actionsPlaceDescriptorsNoRot = np.stack([np.zeros(np.shape(moveDescriptorsNoRot)),moveDescriptorsNoRot],axis=3)
            actionDescriptorsNoRot = np.r_[actionsPickDescriptorsNoRot,actionsPlaceDescriptorsNoRot]
    
            # Get NoRot values
            if obs[1] == 0:
                qCurrPick = getqNotHoldingNoRot(actionsPickDescriptorsNoRot)[:,-1]
                qCurrPlace = getqNotHoldingNoRot(actionsPlaceDescriptorsNoRot)[:,-1]
#                qCurrPick = getqNotHoldingNoRot(actionsPickDescriptorsNoRot)[:,0]
#                qCurrPlace = getqNotHoldingNoRot(actionsPlaceDescriptorsNoRot)[:,0]
            elif obs[1] == 1:
                qCurrPick = getqHoldingNoRot(actionsPickDescriptorsNoRot)[:,-1]
                qCurrPlace = getqHoldingNoRot(actionsPlaceDescriptorsNoRot)[:,-1]
#                qCurrPick = getqHoldingNoRot(actionsPickDescriptorsNoRot)[:,0]
#                qCurrPlace = getqHoldingNoRot(actionsPlaceDescriptorsNoRot)[:,0]
            else:
                print("error: state out of bounds")
            qCurrNoRot = np.squeeze(np.r_[qCurrPick,qCurrPlace])
    
            # Get Rot actions corresponding to top k% NoRot actions
            k=0.2 # top k% of NoRot actions
            valsNoRot = qCurrNoRot
            topKactionsNoRot = np.argsort(valsNoRot)[-np.int32(np.shape(valsNoRot)[0]*k):]
            topKpositionsNoRot = topKactionsNoRot % env.num_moves
            topKpickplaceNoRot = topKactionsNoRot / env.num_moves
            actionsCandidates = []
            for ii in range(2):
                eltsPos = topKpositionsNoRot[topKpickplaceNoRot==ii]
                for jj in range(env.num_orientations):
                    actionsCandidates = np.r_[actionsCandidates,eltsPos + jj*env.num_moves + ii*(env.num_moves*env.num_orientations)]
            actionsCandidates = np.int32(actionsCandidates)
        
        # No hierarchy
        else:
            actionsCandidates = range(2*env.num_moves*env.num_orientations)
            
        # Get Rot descriptors
        moveDescriptorsRot = getMoveActionDescriptorsRot([obs[0]])
        moveDescriptorsRot = moveDescriptorsRot*2-1
        actionsPickDescriptorsRot = np.stack([moveDescriptorsRot, np.zeros(np.shape(moveDescriptorsRot))],axis=3)
        actionsPlaceDescriptorsRot = np.stack([np.zeros(np.shape(moveDescriptorsRot)),moveDescriptorsRot],axis=3)
        actionDescriptorsRot = np.r_[actionsPickDescriptorsRot,actionsPlaceDescriptorsRot]

        # Get qCurr using actionCandidates
        actionDescriptorsRotReduced = actionDescriptorsRot[actionsCandidates]
        if obs[1] == 0:
            qCurrReduced = np.squeeze(getqNotHoldingRot(actionDescriptorsRotReduced))
        elif obs[1] == 1:
            qCurrReduced = np.squeeze(getqHoldingRot(actionDescriptorsRotReduced))
        else:
            print("error: state out of bounds")
        qCurr = -100*np.ones(np.shape(actionDescriptorsRot)[0])
        qCurr[actionsCandidates] = np.copy(qCurrReduced)

        # Update tabular state-value function using V(s) = max_a Q(s,a)
        thisStateValues = np.max(qCurr)
        V[obs[1]] = (1-lrState) * V[obs[1]] + lrState * thisStateValues

        # Select e-greedy action to execute
        qCurrNoise = qCurr + np.random.random(np.shape(qCurr))*0.01 # add small amount of noise to break ties randomly
        action = np.argmax(qCurrNoise)
        if (np.random.rand() < exploration.value(t)) and not vispolicy:
            action = np.random.randint(num_actions)

        if vispolicy:
            print("action: " + str(action))
            plt.subplot(1,2,1)
            plt.imshow(env.state[0][:,:,0])

        # Execute action
        new_obs, rew, done, _ = env.step(action)
        
        position = action % env.num_moves
        pickplace = action / (env.num_moves * env.num_orientations)
        actionNoRot = position+pickplace*env.num_moves
        
        if useHierarchy:
            # store both NoRot and Rot descriptors
            replay_buffer.add(cp.copy(obs[1]), np.copy(actionDescriptorsNoRot[actionNoRot,:]), np.copy(actionDescriptorsRot[action,:]), cp.copy(rew), cp.copy(new_obs[1]), cp.copy(float(done)))
        else:
            # store only Rot descriptor
            replay_buffer.add(cp.copy(obs[1]), np.copy(actionDescriptorsRot[action,:]), np.copy(actionDescriptorsRot[action,:]), cp.copy(rew), cp.copy(new_obs[1]), cp.copy(float(done)))
            

        if vispolicy:
            print("rew: " + str(rew))
            print("done: " + str(done))
            plt.subplot(1,2,2)
            plt.imshow(env.state[0][:,:,0])
            plt.show()

        if t > learning_starts and t % train_freq == 0:

            # Get batch
            if prioritized_replay:
                beta=beta_schedule.value(t)
                states_t, actionPatchesNoRot, actionPatchesRot, rewards, states_tp1, dones, weights, batch_idxes = replay_buffer.sample(batch_size, beta)
            else:
                states_t, actionPatchesNoRot, actionPatchesRot, rewards, states_tp1, dones = replay_buffer.sample(batch_size)
                weights, batch_idxes = np.ones_like(rewards), None

            # Calculate target
            targets = rewards + (1-dones) * gamma * V[states_tp1]
            
            # Get current q-values and calculate td error and q-value targets
            qCurrTargetNotHolding = getqNotHoldingRot(actionPatchesRot)
            qCurrTargetHolding = getqHoldingRot(actionPatchesRot)
            qCurrTarget = np.concatenate([qCurrTargetNotHolding,qCurrTargetHolding],axis=1)
            td_error = qCurrTarget[range(batch_size),states_t] - targets
            qCurrTarget[range(batch_size),states_t] = targets

            # Train
            targetTrainNotHoldingRot(actionPatchesRot, np.reshape(qCurrTarget[:,0],[batch_size,1]), np.reshape(weights,[batch_size,1]))
            targetTrainHoldingRot(actionPatchesRot, np.reshape(qCurrTarget[:,1],[batch_size,1]), np.reshape(weights,[batch_size,1]))

            # Only train NoRot if we're doing the hierarchy
            if useHierarchy:

                # Get qCurr for NoRot
                qCurrTargetNotHoldingNoRot = getqNotHoldingNoRot(actionPatchesNoRot)
                qCurrTargetHoldingNoRot = getqHoldingNoRot(actionPatchesNoRot)
                qCurrTargetNoRot = np.stack([qCurrTargetNotHoldingNoRot,qCurrTargetHoldingNoRot],axis=2)
                qCurrTargetNoRotInit = np.copy(qCurrTargetNoRot)
                
                # Copy into cascade with pruning.
                qCurrTargetNoRot[range(batch_size),0,states_t] = targets
                for i in range(num_cascade-1):
                    mask = targets > qCurrTargetNoRotInit[range(batch_size),i,states_t]
                    qCurrTargetNoRot[range(batch_size),i+1,states_t] = \
                        mask*targets + \
                        (1-mask)*qCurrTargetNoRot[range(batch_size),i+1,states_t]

                targetTrainNotHoldingNoRot(actionPatchesNoRot, qCurrTargetNoRot[:,:,0], np.tile(np.reshape(weights,[batch_size,1]),[1,num_cascade]))
                targetTrainHoldingNoRot(actionPatchesNoRot, qCurrTargetNoRot[:,:,1], np.tile(np.reshape(weights,[batch_size,1]),[1,num_cascade]))

#                targetTrainNotHoldingNoRot(actionPatchesNoRot, np.reshape(qCurrTarget[:,0],[batch_size,1]), np.reshape(weights,[batch_size,1]))
#                targetTrainHoldingNoRot(actionPatchesNoRot, np.reshape(qCurrTarget[:,1],[batch_size,1]), np.reshape(weights,[batch_size,1]))

            # Update replay priorities using td_error
            if prioritized_replay:
                new_priorities = np.abs(td_error) + prioritized_replay_eps
                replay_buffer.update_priorities(batch_idxes, new_priorities)


        # bookkeeping for storing episode rewards
        episode_rewards[-1] += rew
        if done:
            new_obs = env.reset()
            episode_rewards.append(0.0)
        mean_100ep_reward = round(np.mean(episode_rewards[-51:-1]), 1)
        num_episodes = len(episode_rewards)
        if done and print_freq is not None and len(episode_rewards) % print_freq == 0:
            timerFinal = time.time()
#            print("steps: " + str(t) + ", episodes: " + str(num_episodes) + ", mean 100 episode reward: " + str(mean_100ep_reward) + ", % time spent exploring: " + str(int(100 * exploration.value(t))) + ", time elapsed: " + str(timerFinal - timerStart) + ", tderror: " + str(mean_100ep_tderror))
            print("steps: " + str(t) + ", episodes: " + str(num_episodes) + ", mean 100 episode reward: " + str(mean_100ep_reward) + ", % time spent exploring: " + str(int(100 * exploration.value(t))) + ", time elapsed: " + str(timerFinal - timerStart))
#            print("steps: " + str(t) + ", episodes: " + str(num_episodes) + ", mean 100 episode reward: " + str(mean_100ep_reward) + ", % time spent exploring: " + str(int(100 * exploration.value(t))))
#            print("time to do training: " + str(timerFinal - timerStart))
            timerStart = timerFinal
        
        obs = np.copy(new_obs)

    # save what we learned
    if fileOut != "None":
        saver = tf.train.Saver()
        saver.save(sess, fileOut)
        fileOutV = fileOut + 'V'
        print("fileOutV: " + fileOutV)
        np.save(fileOutV,V)

    # display value function
    obs = env.reset()
    
    moveDescriptorsNoRot = getMoveActionDescriptorsNoRot([obs[0]])
    moveDescriptorsNoRot = moveDescriptorsNoRot*2-1
    actionsPickDescriptors = np.stack([moveDescriptorsNoRot, np.zeros(np.shape(moveDescriptorsNoRot))],axis=3)
    actionsPlaceDescriptors = np.stack([np.zeros(np.shape(moveDescriptorsNoRot)), moveDescriptorsNoRot],axis=3)
    qPickNotHoldingNoRot = getqNotHoldingNoRot(actionsPickDescriptors)
    qPickHoldingNoRot = getqHoldingNoRot(actionsPickDescriptors)
    qPickNoRotFirst = np.stack([qPickNotHoldingNoRot[:,0],qPickHoldingNoRot[:,0]],axis=1)
    qPickNoRotLast = np.stack([qPickNotHoldingNoRot[:,-1],qPickHoldingNoRot[:,-1]],axis=1)
    qPlaceNotHoldingNoRot = getqNotHoldingNoRot(actionsPlaceDescriptors)
    qPlaceHoldingNoRot = getqHoldingNoRot(actionsPlaceDescriptors)
    qPlaceNoRotFirst = np.stack([qPlaceNotHoldingNoRot[:,0],qPlaceHoldingNoRot[:,0]],axis=1)
    qPlaceNoRotLast = np.stack([qPlaceNotHoldingNoRot[:,-1],qPlaceHoldingNoRot[:,-1]],axis=1)
    
    moveDescriptors = getMoveActionDescriptorsRot([obs[0]])
    moveDescriptors = moveDescriptors*2-1
    actionsPickDescriptors = np.stack([moveDescriptors, np.zeros(np.shape(moveDescriptors))],axis=3)
    actionsPlaceDescriptors = np.stack([np.zeros(np.shape(moveDescriptors)), moveDescriptors],axis=3)
    qPickNotHolding = getqNotHoldingRot(actionsPickDescriptors)
    qPickHolding = getqHoldingRot(actionsPickDescriptors)
    qPick = np.concatenate([qPickNotHolding,qPickHolding],axis=1)
    qPlaceNotHolding = getqNotHoldingRot(actionsPlaceDescriptors)
    qPlaceHolding = getqHoldingRot(actionsPlaceDescriptors)
    qPlace = np.concatenate([qPlaceNotHolding,qPlaceHolding],axis=1)

    gridSize = len(env.moveCenters)
    print("Value function for pick action in hold-0 state:")
    print(str(np.reshape(qPickNoRotLast[:gridSize**2,0],[gridSize,gridSize])))
    print("Value function for pick action for rot0 in hold-0 state:")
    print(str(np.reshape(qPick[:gridSize**2,0],[gridSize,gridSize])))
    print("Value function for pick action for rot1 in hold-0 state:")
    print(str(np.reshape(qPick[gridSize**2:2*gridSize**2,0],[gridSize,gridSize])))
    print("Value function for pick action for rot2 in hold-0 state:")
    print(str(np.reshape(qPick[2*gridSize**2:3*gridSize**2,0],[gridSize,gridSize])))
    print("Value function for pick action for rot3 in hold-0 state:")
    print(str(np.reshape(qPick[3*gridSize**2:4*gridSize**2,0],[gridSize,gridSize])))
        
    print("Value function for place action in hold-1 state:")
    print(str(np.reshape(qPlaceNoRotLast[:gridSize**2,1],[gridSize,gridSize])))
    print("Value function for place action for rot0 in hold-1 state:")
    print(str(np.reshape(qPlace[:gridSize**2,1],[gridSize,gridSize])))
    print("Value function for place action for rot1 in hold-1 state:")
    print(str(np.reshape(qPlace[gridSize**2:2*gridSize**2,1],[gridSize,gridSize])))
    print("Value function for place action for rot2 in hold-1 state:")
    print(str(np.reshape(qPlace[2*gridSize**2:3*gridSize**2,1],[gridSize,gridSize])))
    print("Value function for place action for rot3 in hold-1 state:")
    print(str(np.reshape(qPlace[3*gridSize**2:4*gridSize**2,1],[gridSize,gridSize])))    

    plt.subplot(2,7,1)
    plt.imshow(np.tile(env.state[0],[1,1,3]),interpolation=None)
    plt.subplot(2,7,2)
    plt.imshow(np.reshape(qPick[:gridSize**2,0],[gridSize,gridSize]),vmin=5,vmax=12)
    plt.subplot(2,7,3)
    plt.imshow(np.reshape(qPick[gridSize**2:2*gridSize**2,0],[gridSize,gridSize]),vmin=5,vmax=12)
    plt.subplot(2,7,4)
    plt.imshow(np.reshape(qPick[2*gridSize**2:3*gridSize**2,0],[gridSize,gridSize]),vmin=5,vmax=12)
    plt.subplot(2,7,5)
    plt.imshow(np.reshape(qPick[3*gridSize**2:4*gridSize**2,0],[gridSize,gridSize]),vmin=5,vmax=12)
    plt.subplot(2,7,6)
    plt.imshow(np.reshape(qPickNoRotFirst[:gridSize**2,0],[gridSize,gridSize]),vmin=5,vmax=12)
    plt.subplot(2,7,7)
    plt.imshow(np.reshape(qPickNoRotLast[:gridSize**2,0],[gridSize,gridSize]),vmin=5,vmax=12)
    plt.subplot(2,7,9)
    plt.imshow(np.reshape(qPlace[:gridSize**2,1],[gridSize,gridSize]),vmin=5,vmax=12)
    plt.subplot(2,7,10)
    plt.imshow(np.reshape(qPlace[gridSize**2:2*gridSize**2,1],[gridSize,gridSize]),vmin=5,vmax=12)
    plt.subplot(2,7,11)
    plt.imshow(np.reshape(qPlace[2*gridSize**2:3*gridSize**2,1],[gridSize,gridSize]),vmin=5,vmax=12)
    plt.subplot(2,7,12)
    plt.imshow(np.reshape(qPlace[3*gridSize**2:4*gridSize**2,1],[gridSize,gridSize]),vmin=5,vmax=12)
    plt.subplot(2,7,13)
    plt.imshow(np.reshape(qPlaceNoRotFirst[:gridSize**2,1],[gridSize,gridSize]),vmin=5,vmax=12)
    plt.subplot(2,7,14)
    plt.imshow(np.reshape(qPlaceNoRotLast[:gridSize**2,1],[gridSize,gridSize]),vmin=5,vmax=12)
    plt.show()


#    plt.subplot(1,3,1)
#    plt.imshow(np.tile(env.state[0],[1,1,3]),vmin=5,vmax=12)
#    plt.subplot(1,3,2)
#    plt.imshow(np.reshape(qPick[:,0],[gridSize,gridSize]),vmin=5,vmax=12)
#    plt.subplot(1,3,3)
#    plt.imshow(np.reshape(qPlace[:,1],[gridSize,gridSize]),vmin=5,vmax=12)
#    plt.show()

if len(sys.argv) == 7:
    initEnvStride = np.int32(sys.argv[1])
    envStride = np.int32(sys.argv[2])
    fileIn = sys.argv[3]
    fileOut = sys.argv[4]
    inputmaxtimesteps = np.int32(sys.argv[5])
    vispolicy = np.int32(sys.argv[6])
else:
    initEnvStride = 28
    envStride = 28
#    fileIn = 'None'
    fileIn = './disk_28'
    fileOut = 'None'
#    fileOut = './whatilearned28'
#    inputmaxtimesteps = 2000
    inputmaxtimesteps = 50
    vispolicy = False
    

main(initEnvStride, envStride, fileIn, fileOut, inputmaxtimesteps, vispolicy)
    
#if __name__ == '__main__':
#    main()
# 
# Disk/Block arrangement in SE(2) as of 6/27/2018. 
#
# Features of this version:
#  -- option for two-stage rotation hierarchy
#  -- option for filtering out of positions with associated zero-patches
#  -- can handle either blocks or disks
#  -- 32 possible orientations
#  -- only a single bit state (holding/notholding). For a version of the code
#          that allows for history-as-state, see puckarrange2_3inarow3.py
#
# Adapted from puckarrange17.py
#
# How to run: python puckarrange_iter1.py runs the full curriculum cascade. Have a
#             look at that code to get examples of single-runs of this code.
#
#
import sys as sys
import gym
import numpy as np
import time as time
import tensorflow as tf
import tf_util_rob as U
import models as models
from replay_buffer9 import ReplayBuffer, PrioritizedReplayBuffer
from schedules import LinearSchedule
import matplotlib.pyplot as plt
import copy as cp
import scipy as sp

# This environment allows for either disks or blocks at arbitrary levels of
# discretization in the plane. Agent gets reward when the proper arrangement
# is achieved.
import envs.puckarrange_env16 as envstandalone


# **** Make tensorflow functions ****

# Evaluate the q function for a given input.
def build_getq(make_actionDeic_ph, q_func, num_discrete_states, num_cascade, scope="deepq", qscope="q_func", reuse=None):

    with tf.variable_scope(scope, reuse=reuse):
        actions_ph = U.ensure_tf_input(make_actionDeic_ph("actions"))
        q_values = q_func(actions_ph.get(), 1, scope=qscope)
        getq = U.function(inputs=[actions_ph], outputs=q_values)
        return getq

# Train q-function
def build_targetTrain(make_actionDeic_ph,
                        make_target_ph,
                        make_weight_ph,
                        q_func,
                        num_discrete_states,
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

# Get candidate patch descriptors given an input image. Candidates are found by
# sliding a window over x/y plane in the image with the given stride (same 
# stride applied both in x and y). All patches have a single orientation with
# respect to the image.
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


# Same as build_getMoveActionDescriptors (get patch descriptors) except that
# this version extracts patches at 16 orientations.
def build_getMoveActionDescriptorsRot(make_obs_ph,actionShape,actionShapeSmall,stride,numOrientations):
    
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
    patchesTiledRot1 = tf.contrib.image.rotate(patchesTiled,np.pi/16)
    patchesTiledRot2 = tf.contrib.image.rotate(patchesTiled,2*np.pi/16)
    patchesTiledRot3 = tf.contrib.image.rotate(patchesTiled,3*np.pi/16)
    patchesTiledRot4 = tf.contrib.image.rotate(patchesTiled,4*np.pi/16)
    patchesTiledRot5 = tf.contrib.image.rotate(patchesTiled,5*np.pi/16)
    patchesTiledRot6 = tf.contrib.image.rotate(patchesTiled,6*np.pi/16)
    patchesTiledRot7 = tf.contrib.image.rotate(patchesTiled,7*np.pi/16)
    patchesTiledRot8 = tf.contrib.image.rotate(patchesTiled,8*np.pi/16)
    patchesTiledRot9 = tf.contrib.image.rotate(patchesTiled,9*np.pi/16)
    patchesTiledRot10 = tf.contrib.image.rotate(patchesTiled,10*np.pi/16)
    patchesTiledRot11 = tf.contrib.image.rotate(patchesTiled,11*np.pi/16)
    patchesTiledRot12 = tf.contrib.image.rotate(patchesTiled,12*np.pi/16)
    patchesTiledRot13 = tf.contrib.image.rotate(patchesTiled,13*np.pi/16)
    patchesTiledRot14 = tf.contrib.image.rotate(patchesTiled,14*np.pi/16)
    patchesTiledRot15 = tf.contrib.image.rotate(patchesTiled,15*np.pi/16)

    if numOrientations == 2:
        patchesTiledAll = tf.concat([patchesTiledRot0,patchesTiledRot8],axis=0)
    elif numOrientations == 4:
        patchesTiledAll = tf.concat([patchesTiledRot0,patchesTiledRot4,patchesTiledRot8,patchesTiledRot12],axis=0)
    elif numOrientations == 8:
        patchesTiledAll = tf.concat([patchesTiledRot0,patchesTiledRot2,patchesTiledRot4,patchesTiledRot6,patchesTiledRot8,patchesTiledRot10,patchesTiledRot12,patchesTiledRot14],axis=0)
    elif numOrientations == 16:
        patchesTiledAll = tf.concat([patchesTiledRot0,patchesTiledRot1,patchesTiledRot2,patchesTiledRot3,patchesTiledRot4,patchesTiledRot5,patchesTiledRot6,patchesTiledRot7,patchesTiledRot8,patchesTiledRot9,patchesTiledRot10,patchesTiledRot11,patchesTiledRot12,patchesTiledRot13,patchesTiledRot14,patchesTiledRot15],axis=0)
    else:
        print('ERROR: invalid number of orientations')

    
    patchesTiledSmall = tf.image.resize_images(patchesTiledAll, [actionShapeSmall[0], actionShapeSmall[1]])
    patchesTiledSmall = tf.reshape(patchesTiledSmall,[-1,actionShapeSmall[0],actionShapeSmall[1]])

    getMoveActionDescriptors = U.function(inputs=[observations_ph], outputs=patchesTiledSmall)
    return getMoveActionDescriptors



def main(initEnvStride, envStride, fileIn, fileOut, inputmaxtimesteps, vispolicy, objType, numOrientations, useRotHierarchy, useHandCodeHierarchy):

    np.set_printoptions(formatter={'float_kind':lambda x: "%.2f" % x})

    # Create environment and set two stride parameters (stride-x and stride-y) 
    # for this problem instance. Most of the time, the two stride parameters will be equal. 
    env = envstandalone.PuckArrange()
    env.initStride = initEnvStride # stride for initial puck placement
    env.stride = envStride # stride for action specification
    env.blockType = objType
    env.num_orientations = numOrientations
    env.reset()
    
    # Standard q-learning parameters
    reuseModels = None
    max_timesteps=inputmaxtimesteps
    exploration_fraction=0.75
    exploration_final_eps=0.1
    gamma=.90
    num_cpu = 16

    # Used by buffering and DQN
    learning_starts=60
    buffer_size=10000
    batch_size=10
    target_network_update_freq=1
    train_freq=1
    print_freq=1
    
    # SGD learning rate 
    lr=0.0003 

    # Set parameters related to shape of the patch and the number of patches
    descriptorShape = (env.blockSize*3,env.blockSize*3,2) # size of patch descriptor relative to number of "blocks" on board (each block is a 28x28 region)
    descriptorShapeSmall = (25,25,2) # size to which each patch gets resized to. Code runs faster w/ smaller sizes, but could miss detail needed to solve the problem.
    num_discrete_states = 2 # number of discrete states: either holding or not
    num_patches = len(env.moveCenters)**2 # env.moveCenters is num of patches along one side of image
    num_actions = num_discrete_states*num_patches*env.num_orientations # total actions = num discrete states X num non-rotated descriptor patches X num of orientations per patch location

    # e-greedy exploration schedule. I find that starting at e=50% helps curriculum learning "remember" what was learned in the prior run.
    exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * max_timesteps),
                                 initial_p=0.5,
                                 final_p=exploration_final_eps)

    # Set parameters for prioritized replay. You  can turn this off just by 
    # setting the line below to False
    prioritized_replay=True
#    prioritized_replay=False
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
        convs=[(16,3,1), (32,3,1)],
        hiddens=[64],
        dueling=True
    )

    # Build tensorflow functions
    def make_obs_ph(name):
        return U.BatchInput(env.observation_space.spaces[0].shape, name=name)
    def make_actionDeic_ph(name):
        return U.BatchInput(descriptorShapeSmall, name=name)
    def make_target_ph(name):
        return U.BatchInput([1], name=name)
    def make_weight_ph(name):
        return U.BatchInput([1], name=name)

    getMoveActionDescriptorsNoRot = build_getMoveActionDescriptors(make_obs_ph=make_obs_ph,actionShape=descriptorShape,actionShapeSmall=descriptorShapeSmall,stride=env.stride)
    getMoveActionDescriptorsRot = build_getMoveActionDescriptorsRot(make_obs_ph=make_obs_ph,actionShape=descriptorShape,actionShapeSmall=descriptorShapeSmall,stride=env.stride,numOrientations=numOrientations)
    
    getqNotHoldingRot = build_getq(
            make_actionDeic_ph=make_actionDeic_ph,
            q_func=q_func,
            num_discrete_states=num_discrete_states,
            num_cascade=5,
            scope="deepq",
            qscope="q_func_notholding_rot",
            reuse=reuseModels
            )
    getqHoldingRot = build_getq(
            make_actionDeic_ph=make_actionDeic_ph,
            q_func=q_func,
            num_discrete_states=num_discrete_states,
            num_cascade=5,
            scope="deepq",
            qscope="q_func_holding_rot",
            reuse=reuseModels
            )

    targetTrainNotHoldingRot = build_targetTrain(
        make_actionDeic_ph=make_actionDeic_ph,
        make_target_ph=make_target_ph,
        make_weight_ph=make_weight_ph,
        q_func=q_func,
        num_discrete_states=num_discrete_states,
        num_cascade=5,
        optimizer=tf.train.AdamOptimizer(learning_rate=lr/2.), # rotation learns slower than norot
#        optimizer=tf.train.GradientDescentOptimizer(learning_rate=lr/2.), # rotation learns slower than norot
        scope="deepq", 
        qscope="q_func_notholding_rot",
#        grad_norm_clipping=1.,
        reuse=reuseModels
    )

    targetTrainHoldingRot = build_targetTrain(
        make_actionDeic_ph=make_actionDeic_ph,
        make_target_ph=make_target_ph,
        make_weight_ph=make_weight_ph,
        q_func=q_func,
        num_discrete_states=num_discrete_states,
        num_cascade=5,
        optimizer=tf.train.AdamOptimizer(learning_rate=lr/2.), # rotation learns slower than norot
#        optimizer=tf.train.GradientDescentOptimizer(learning_rate=lr/2.), # rotation learns slower than norot
        scope="deepq", 
        qscope="q_func_holding_rot",
#        grad_norm_clipping=1.,
        reuse=reuseModels
    )
    
    getqNotHoldingNoRot = build_getq(
            make_actionDeic_ph=make_actionDeic_ph,
            q_func=q_func,
            num_discrete_states=num_discrete_states,
            num_cascade=5,
            scope="deepq",
            qscope="q_func_notholding_norot",
            reuse=reuseModels
            )
    getqHoldingNoRot = build_getq(
            make_actionDeic_ph=make_actionDeic_ph,
            q_func=q_func,
            num_discrete_states=num_discrete_states,
            num_cascade=5,
            scope="deepq",
            qscope="q_func_holding_norot",
            reuse=reuseModels
            )

    targetTrainNotHoldingNoRot = build_targetTrain(
        make_actionDeic_ph=make_actionDeic_ph,
        make_target_ph=make_target_ph,
        make_weight_ph=make_weight_ph,
        q_func=q_func,
        num_discrete_states=num_discrete_states,
        num_cascade=5,
        optimizer=tf.train.AdamOptimizer(learning_rate=lr),
#        optimizer=tf.train.GradientDescentOptimizer(learning_rate=lr),
        scope="deepq", 
        qscope="q_func_notholding_norot",
#        grad_norm_clipping=1.,
        reuse=reuseModels
    )

    targetTrainHoldingNoRot = build_targetTrain(
        make_actionDeic_ph=make_actionDeic_ph,
        make_target_ph=make_target_ph,
        make_weight_ph=make_weight_ph,
        q_func=q_func,
        num_discrete_states=num_discrete_states,
        num_cascade=5,
        optimizer=tf.train.AdamOptimizer(learning_rate=lr),
#        optimizer=tf.train.GradientDescentOptimizer(learning_rate=lr),
        scope="deepq", 
        qscope="q_func_holding_norot",
#        grad_norm_clipping=1.,
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
        
        # Get NoRot descriptors. Each x-y position gets one descriptor patch in
        # a single orientation. Encode pick/place using a stack of two image channels.
        # Pick actions are denoted by the patch in channel 0 and zeros in channel 1.
        # Place actions have zeros in channel 0 and the patch in channel 1.
        # Each elt of actionDescriptorsNoRot is a pick/place action to a specific
        # position with orientation left unspecified.
        moveDescriptorsNoRot = getMoveActionDescriptorsNoRot([obs[0]])
        moveDescriptorsNoRot = moveDescriptorsNoRot*2-1
        actionsPickDescriptorsNoRot = np.stack([moveDescriptorsNoRot, np.zeros(np.shape(moveDescriptorsNoRot))],axis=3)
        actionsPlaceDescriptorsNoRot = np.stack([np.zeros(np.shape(moveDescriptorsNoRot)),moveDescriptorsNoRot],axis=3)
        actionDescriptorsNoRot = np.r_[actionsPickDescriptorsNoRot,actionsPlaceDescriptorsNoRot]

        # If useHandCodeHierarchy == 1, we exclude patches that are completely zero
        if useHandCodeHierarchy == 1:
            nonZeroMoves = np.sum(np.sum(moveDescriptorsNoRot > 0,-1),-1) > 0
            movesCandidates = np.nonzero(nonZeroMoves)[0]
            actionsCandidates = []
            for jj in range(0,num_discrete_states):
                for ii in range(0,env.num_orientations):
                    actionsCandidates = np.r_[actionsCandidates,movesCandidates + ii*env.num_moves + jj*env.num_orientations*env.num_moves]
            actionsCandidatesHandCodeHierarchy = np.int32(actionsCandidates)
            movesCandidatesHandCodeHierarchy = np.int32(movesCandidates)
            
        else:
            actionsCandidatesHandCodeHierarchy = range(num_discrete_states*env.num_moves*env.num_orientations)
            movesCandidatesHandCodeHierarchy = range(env.num_moves)
            
        # If useRotHierarchy == 1, we evaluate the Q function using a two-level hierarchy.
        # The first level (getq<Not>HoldingNoRot) is position but no rotation. 
        # The second level (getq<Not>HoldingRot) is both position and orientation.
        # Specifically, we evaluate getq<Not>HoldingRot only for the top 20% of positions
        # found using getq<Not>HoldingNoRot.
        if useRotHierarchy == 1:
        
            # Get NoRot values
            if obs[1] == 0:
                qCurrPick = getqNotHoldingNoRot(actionsPickDescriptorsNoRot[movesCandidatesHandCodeHierarchy])
                qCurrPlace = getqNotHoldingNoRot(actionsPlaceDescriptorsNoRot[movesCandidatesHandCodeHierarchy])
            elif obs[1] == 1:
                qCurrPick = getqHoldingNoRot(actionsPickDescriptorsNoRot[movesCandidatesHandCodeHierarchy])
                qCurrPlace = getqHoldingNoRot(actionsPlaceDescriptorsNoRot[movesCandidatesHandCodeHierarchy])
            else:
                print("error: state out of bounds")
            qCurrNoRot = np.squeeze(np.r_[qCurrPick,qCurrPlace])
            qCurrNoRotIdx = np.r_[movesCandidatesHandCodeHierarchy,env.num_moves + movesCandidatesHandCodeHierarchy]
    
            # Get Rot actions corresponding to top k% NoRot actions
            k=0.2 # top k% of NoRot actions
#            k=0.1 # DEBUG: TRYING TO VISUALIZE AND RAN OUT OF MEM ON LAPTOP...
            valsNoRot = qCurrNoRot
            topKactionsNoRot = np.argsort(valsNoRot)[-np.int32(np.shape(valsNoRot)[0]*k):]
            topKpositionsNoRot = qCurrNoRotIdx[topKactionsNoRot] % env.num_moves
            topKpickplaceNoRot = qCurrNoRotIdx[topKactionsNoRot] / env.num_moves
            actionsCandidates = []
            for ii in range(2):
                eltsPos = topKpositionsNoRot[topKpickplaceNoRot==ii]
                for jj in range(env.num_orientations):
                    actionsCandidates = np.r_[actionsCandidates,eltsPos + jj*env.num_moves + ii*(env.num_moves*env.num_orientations)]
            actionsCandidatesRotHierarchy = np.int32(actionsCandidates)

        # No rot hierarchy
        else:
            actionsCandidatesRotHierarchy = range(num_discrete_states*env.num_moves*env.num_orientations)
        
        # Intersect two types of hierarchy and get final list of actions to consider
        actionsCandidates = np.intersect1d(actionsCandidatesRotHierarchy,actionsCandidatesHandCodeHierarchy)
        
        # Get all patch descriptors (position + rotation)
        moveDescriptorsRot = getMoveActionDescriptorsRot([obs[0]])
        moveDescriptorsRot = moveDescriptorsRot*2-1
        actionsPickDescriptorsRot = np.stack([moveDescriptorsRot, np.zeros(np.shape(moveDescriptorsRot))],axis=3)
        actionsPlaceDescriptorsRot = np.stack([np.zeros(np.shape(moveDescriptorsRot)),moveDescriptorsRot],axis=3)
        actionDescriptorsRot = np.r_[actionsPickDescriptorsRot,actionsPlaceDescriptorsRot]

        # Get qCurr for selected actions, i.e. actions contained in actionCandidates
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

#        # Select e-greedy action to execute
#        qCurrNoise = qCurr + np.random.random(np.shape(qCurr))*0.01 # add small amount of noise to break ties randomly
#        action = np.argmax(qCurrNoise)
#        if (np.random.rand() < exploration.value(t)) and not vispolicy:
#            action = np.random.randint(num_actions)

        # e-greedy + softmax action selection
        qCurrExp = np.exp(qCurr/0.1)
        probs = qCurrExp / np.sum(qCurrExp)
        action = np.random.choice(range(np.size(probs)),p=probs)
        if (np.random.rand() < exploration.value(t)) and not vispolicy:
            action = np.random.randint(num_actions)

        # factor action into position, orientation, pick-or-place
        position = action % env.num_moves
        pickplace = action / (env.num_moves * env.num_orientations)
        orientation = (action - pickplace * env.num_moves * env.num_orientations) / env.num_moves
        actionNoRot = position+pickplace*env.num_moves

        if vispolicy:
            print("action: " + str(action))
            print("position: " + str(position))
            print("pickplace: " + str(pickplace))
            print("orientation: " + str(orientation))
            plt.subplot(1,2,1)
            plt.imshow(env.state[0][:,:,0])
            sp.misc.imsave('temp1.png',env.state[0][:,:,0])

        # Execute action
        new_obs, rew, done, _ = env.step(action)

        # Add to buffer
        replay_buffer.add(cp.copy(obs[1]), np.copy(actionDescriptorsNoRot[actionNoRot,:]), np.copy(actionDescriptorsRot[action,:]), cp.copy(rew), cp.copy(new_obs[1]), cp.copy(float(done)))

        # If vispolicy==True, then visualize policy
        if vispolicy:
            print("rew: " + str(rew))
            print("done: " + str(done))
            plt.subplot(1,2,2)
            plt.imshow(env.state[0][:,:,0])
            plt.show()
            sp.misc.imsave('temp2.png',env.state[0][:,:,0])

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

            targetTrainNotHoldingNoRot(actionPatchesNoRot, np.reshape(qCurrTarget[:,0],[batch_size,1]), np.reshape(weights,[batch_size,1]))
            targetTrainHoldingNoRot(actionPatchesNoRot, np.reshape(qCurrTarget[:,1],[batch_size,1]), np.reshape(weights,[batch_size,1]))

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
            print("steps: " + str(t) + ", episodes: " + str(num_episodes) + ", mean 100 episode reward: " + str(mean_100ep_reward) + ", % time spent exploring: " + str(int(100 * exploration.value(t))) + ", time elapsed: " + str(timerFinal - timerStart))
            timerStart = timerFinal
        
        obs = np.copy(new_obs)

    # save learning curve
    filename = 'PA18_deictic_rewards.dat'
    np.savetxt(filename,episode_rewards)

    # save what we learned
    if fileOut != "None":
        saver = tf.train.Saver()
        saver.save(sess, fileOut)
        fileOutV = fileOut + 'V'
        print("fileOutV: " + fileOutV)
        np.save(fileOutV,V)

    # Display value function from this run
    obs = env.reset()
    
    moveDescriptorsNoRot = getMoveActionDescriptorsNoRot([obs[0]])
    moveDescriptorsNoRot = moveDescriptorsNoRot*2-1
    actionsPickDescriptors = np.stack([moveDescriptorsNoRot, np.zeros(np.shape(moveDescriptorsNoRot))],axis=3)
    actionsPlaceDescriptors = np.stack([np.zeros(np.shape(moveDescriptorsNoRot)), moveDescriptorsNoRot],axis=3)
    qPickNotHoldingNoRot = getqNotHoldingNoRot(actionsPickDescriptors)
    qPickHoldingNoRot = getqHoldingNoRot(actionsPickDescriptors)
    qPickNoRot = np.concatenate([qPickNotHoldingNoRot,qPickHoldingNoRot],axis=1)
    qPlaceNotHoldingNoRot = getqNotHoldingNoRot(actionsPlaceDescriptors)
    qPlaceHoldingNoRot = getqHoldingNoRot(actionsPlaceDescriptors)
    qPlaceNoRot = np.concatenate([qPlaceNotHoldingNoRot,qPlaceHoldingNoRot],axis=1)
    
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
    print(str(np.reshape(qPickNoRot[:gridSize**2,0],[gridSize,gridSize])))
    for ii in range(env.num_orientations):
        print("Value function for pick action for rot" + str(ii) + " in hold-0 state:")
        print(str(np.reshape(qPick[ii*(gridSize**2):(ii+1)*(gridSize**2),0],[gridSize,gridSize])))
        
        
    print("Value function for place action in hold-1 state:")
    print(str(np.reshape(qPlaceNoRot[:gridSize**2,1],[gridSize,gridSize])))
    for ii in range(env.num_orientations):
        print("Value function for place action for rot" + str(ii) + " in hold-1 state:")
        print(str(np.reshape(qPlace[ii*(gridSize**2):(ii+1)*(gridSize**2),0],[gridSize,gridSize])))


if len(sys.argv) == 11:
    initEnvStride = np.int32(sys.argv[1])
    envStride = np.int32(sys.argv[2])
    fileIn = sys.argv[3]
    fileOut = sys.argv[4]
    inputmaxtimesteps = np.int32(sys.argv[5])
    vispolicy = np.int32(sys.argv[6])
    objType = sys.argv[7]
    numOrientations = np.int32(sys.argv[8])
    useRotHierarchy = np.int32(sys.argv[9])
    useHandCodeHierarchy = np.int32(sys.argv[10])
    
else:
    initEnvStride = 28
    envStride = 28
    fileIn = 'None'
#    fileIn = './disk_28_2'
#    fileIn = './rect_4_16'    
    fileOut = 'None'
#    fileOut = './whatilearned28'
    inputmaxtimesteps = 5000
#    inputmaxtimesteps = 100
    vispolicy = False
#    vispolicy = True
    objType = 'Disks'
#    objType = 'Blocks'
    numOrientations = 2
#    numOrientations = 16
    useRotHierarchy = 1
    useHandCodeHierarchy = 1


main(initEnvStride, envStride, fileIn, fileOut, inputmaxtimesteps, vispolicy, objType, numOrientations, useRotHierarchy, useHandCodeHierarchy)
    
#if __name__ == '__main__':
#    main()
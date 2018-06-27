#
# Here, I solve a pick and place task involving disks of various diameters using
# the deictic action method. Essentially, we have a large action branching factor.
# The state representation is just a single bit indicating whether the object
# is held in the hand or not. The optimal policy is only two time steps long.
#
# This version uses a state value function *and* an action value function to speed up 
# learning for the large action branching factor.
#
# Adapted from puckarrange1.py
#
# Results: After I fixed a memory leak, I think this code is working well. I can
#          learn the full sequence of seven learning tasks (see puckarrange_iter1.py)
#          in a total of 13 minutes on a machine w/ a single nvidia 1080.
#
#
import sys as sys
import gym
import numpy as np
import time as time
import tensorflow as tf
import tf_util_rob as U
import models as models
#from replay_buffer2 import ReplayBuffer, PrioritizedReplayBuffer
from replay_buffer8 import ReplayBuffer, PrioritizedReplayBuffer
from schedules import LinearSchedule
import matplotlib.pyplot as plt
import copy as cp
import scipy as sp

# Two disks placed in a 224x224 image. Disks placed randomly initially. 
# Reward given when the pucks are placed adjacent. Agent must learn to pick
# up one of the disks and place it next to the other.
import envs.puckarrange_env2 as envstandalone


# Define a couple of cosmetic functions
# input: im -> image to draw lines in
#        val -> value to set lines to
def draw_lines_plt(im,val):
    im[:,28,0] = val
    im[:,56,0] = val
    im[:,84,0] = val
    im[28,:,0] = val
    im[56,:,0] = val
    im[84,:,0] = val


# **** Make tensorflow functions ****

# Evaluate the q function for a given input.
def build_getq(make_actionDeic_ph, q_func, num_states, num_cascade, scope="deepq", qscope="q_func", reuse=None):

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
#            obs,
            ksizes=[1, actionShape[0], actionShape[1], 1],
#            strides=[1, deicticPad[0]/2, deicticPad[1]/2, 1],
            strides=[1, stride, stride, 1], 
            rates=[1, 1, 1, 1], 
            padding='VALID')
    patchesShape = tf.shape(patches)
    patchesTiled = tf.reshape(patches,[patchesShape[0]*patchesShape[1]*patchesShape[2],actionShape[0],actionShape[1],1])
    patchesTiledSmall = tf.image.resize_images(patchesTiled, [actionShapeSmall[0], actionShapeSmall[1]])
    patchesTiledSmall = tf.reshape(patchesTiledSmall,[-1,actionShapeSmall[0],actionShapeSmall[1]])

    getMoveActionDescriptors = U.function(inputs=[observations_ph], outputs=patchesTiledSmall)
    return getMoveActionDescriptors



def main(initEnvStride, envStride, fileIn, fileOut, inputmaxtimesteps):

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
#    exploration_fraction=0.75
    exploration_fraction=1
    exploration_final_eps=0.1
    gamma=.90
    num_cpu = 16

    # Used by buffering and DQN
    learning_starts=60
#    buffer_size=1000
#    batch_size=32
    buffer_size=10000
    batch_size=10
    target_network_update_freq=1
    train_freq=1
    print_freq=1
    lr=0.0003

    # Set parameters related to shape of the patch and the number of patches
    descriptorShape = (env.blockSize*3,env.blockSize*3,2)
#    descriptorShapeSmall = (10,10,2)
#    descriptorShapeSmall = (15,15,2)
    descriptorShapeSmall = (20,20,2)
    num_states = 2 # either holding or not
    num_patches = len(env.moveCenters)**2
    num_actions = 2*num_patches

    # Create the schedule for exploration starting from 1.
    exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * max_timesteps),
                                 initial_p=1.0,
                                 final_p=exploration_final_eps)
#    exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * max_timesteps),
#                                 initial_p=exploration_final_eps,
#                                 final_p=exploration_final_eps)

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
        convs=[(16,3,1)],
        hiddens=[32],
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

    getMoveActionDescriptors = build_getMoveActionDescriptors(make_obs_ph=make_obs_ph,actionShape=descriptorShape,actionShapeSmall=descriptorShapeSmall,stride=env.stride)
    
    getqNotHolding = build_getq(
            make_actionDeic_ph=make_actionDeic_ph,
            q_func=q_func,
            num_states=num_states,
            num_cascade=5,
            scope="deepq",
            qscope="q_func_notholding",
            reuse=reuseModels
            )
    getqHolding = build_getq(
            make_actionDeic_ph=make_actionDeic_ph,
            q_func=q_func,
            num_states=num_states,
            num_cascade=5,
            scope="deepq",
            qscope="q_func_holding",
            reuse=reuseModels
            )

    targetTrainNotHolding = build_targetTrain(
        make_actionDeic_ph=make_actionDeic_ph,
        make_target_ph=make_target_ph,
        make_weight_ph=make_weight_ph,
        q_func=q_func,
        num_states=num_states,
        num_cascade=5,
        optimizer=tf.train.AdamOptimizer(learning_rate=lr),
        scope="deepq", 
        qscope="q_func_notholding",
        grad_norm_clipping=1.,
        reuse=reuseModels
    )

    targetTrainHolding = build_targetTrain(
        make_actionDeic_ph=make_actionDeic_ph,
        make_target_ph=make_target_ph,
        make_weight_ph=make_weight_ph,
        q_func=q_func,
        num_states=num_states,
        num_cascade=5,
        optimizer=tf.train.AdamOptimizer(learning_rate=lr),
        scope="deepq", 
        qscope="q_func_holding",
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

#    sp.misc.imsave('temp.png',obs[0][:,:,0])

    # Iterate over time steps
    for t in range(max_timesteps):
        
        # Get action set: <num_patches> pick actions followed by <num_patches> place actions
        moveDescriptors = getMoveActionDescriptors([obs[0]])
        moveDescriptors = moveDescriptors*2-1
        actionsPickDescriptors = np.stack([moveDescriptors, np.zeros(np.shape(moveDescriptors))],axis=3)
        actionsPlaceDescriptors = np.stack([np.zeros(np.shape(moveDescriptors)),moveDescriptors],axis=3)
        actionDescriptors = np.r_[actionsPickDescriptors,actionsPlaceDescriptors]

        # Get qCurr. I split up pick and place in order to accomodate larger batches
        qCurrNotHoldingPick = getqNotHolding(actionsPickDescriptors)
        qCurrHoldingPick = getqHolding(actionsPickDescriptors)
        qCurrNotHoldingPlace = getqNotHolding(actionsPlaceDescriptors)
        qCurrHoldingPlace = getqHolding(actionsPlaceDescriptors)
        qCurr = np.concatenate([np.r_[qCurrNotHoldingPick,qCurrNotHoldingPlace],np.r_[qCurrHoldingPick,qCurrHoldingPlace]],axis=1)

        # Update tabular state-value function using V(s) = max_a Q(s,a)
        thisStateValues = np.max(qCurr[:,obs[1]])
        V[obs[1]] = (1-lrState) * V[obs[1]] + lrState * thisStateValues

        # Select e-greedy action to execute
        qCurrNoise = qCurr + np.random.random(np.shape(qCurr))*0.01 # add small amount of noise to break ties randomly
        action = np.argmax(qCurrNoise[:,obs[1]])
        if np.random.rand() < exploration.value(t):
            action = np.random.randint(num_actions)

#        draw_lines_plt(obs[0],1)
#        env.render()
#        sp.misc.imsave('temp.png',obs[0][:,:,0])
#        draw_lines_plt(obs[0],0)
        
        # Execute action
        new_obs, rew, done, _ = env.step(action)        
        replay_buffer.add(cp.copy(obs[1]), np.copy(actionDescriptors[action,:]), cp.copy(rew), cp.copy(new_obs[1]), cp.copy(float(done)))

        if t > learning_starts and t % train_freq == 0:

            # Get batch
            if prioritized_replay:
                beta=beta_schedule.value(t)
                states_t, actionPatches, rewards, states_tp1, dones, weights, batch_idxes = replay_buffer.sample(batch_size, beta)
            else:
                states_t, actionPatches, rewards, states_tp1, dones = replay_buffer.sample(batch_size)
                weights, batch_idxes = np.ones_like(rewards), None

            # Calculate target
            targets = rewards + (1-dones) * gamma * V[states_tp1]
            
            # Get current q-values and calculate td error and q-value targets
            qCurrTargetNotHolding = getqNotHolding(actionPatches)
            qCurrTargetHolding = getqHolding(actionPatches)
            qCurrTarget = np.concatenate([qCurrTargetNotHolding,qCurrTargetHolding],axis=1)
            td_error = qCurrTarget[range(batch_size),states_t] - targets
            qCurrTarget[range(batch_size),states_t] = targets

            # Train
            targetTrainNotHolding(actionPatches, np.reshape(qCurrTarget[:,0],[batch_size,1]), np.reshape(weights,[batch_size,1]))
            targetTrainHolding(actionPatches, np.reshape(qCurrTarget[:,1],[batch_size,1]), np.reshape(weights,[batch_size,1]))

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

    # save learning curve
    filename = 'PA2_deictic_rewards_' +str(num_patches) + "_" + str(max_timesteps) + '.dat'
    np.savetxt(filename,episode_rewards)

    # save what we learned
    if fileOut != "None":
        saver = tf.train.Saver()
        saver.save(sess, fileOut)
        fileOutV = fileOut + 'V'
        print("fileOutV: " + fileOutV)
        np.save(fileOutV,V)

#    # display value function
#    obs = env.reset()
#    moveDescriptors = getMoveActionDescriptors([obs[0]])
#    moveDescriptors = moveDescriptors*2-1
#    gridSize = np.int32(np.sqrt(np.shape(moveDescriptors)[0]))
#
#    actionsPickDescriptors = np.stack([moveDescriptors, np.zeros(np.shape(moveDescriptors))],axis=3)
#    actionsPlaceDescriptors = np.stack([np.zeros(np.shape(moveDescriptors)), moveDescriptors],axis=3)
#    
#    print(str(obs[0][:,:,0]))
#    
#    qPickNotHolding = getqNotHolding(actionsPickDescriptors)
#    qPickHolding = getqHolding(actionsPickDescriptors)
#    qPick = np.concatenate([qPickNotHolding,qPickHolding],axis=1)
#    print("Value function for pick action in hold-nothing state:")
#    print(str(np.reshape(qPick[:,0],[gridSize,gridSize])))
#    print("Value function for pick action in hold-1 state:")
#    print(str(np.reshape(qPick[:,1],[gridSize,gridSize])))
#
#    qPlaceNotHolding = getqNotHolding(actionsPlaceDescriptors)
#    qPlaceHolding = getqHolding(actionsPlaceDescriptors)
#    qPlace = np.concatenate([qPlaceNotHolding,qPlaceHolding],axis=1)
#    print("Value function for place action in hold-nothing state:")
#    print(str(np.reshape(qPlace[:,0],[gridSize,gridSize])))
#    print("Value function for place action in hold-1 state:")
#    print(str(np.reshape(qPlace[:,1],[gridSize,gridSize])))
#    
#    plt.subplot(1,3,1)
#    plt.imshow(np.tile(env.state[0],[1,1,3]))
#    plt.subplot(1,3,2)
#    plt.imshow(np.reshape(qPick[:,0],[gridSize,gridSize]))
#    plt.subplot(1,3,3)
#    plt.imshow(np.reshape(qPlace[:,1],[gridSize,gridSize]))
#    plt.show()

if len(sys.argv) == 6:
    initEnvStride = np.int32(sys.argv[1])
    envStride = np.int32(sys.argv[2])
    fileIn = sys.argv[3]
    fileOut = sys.argv[4]
    inputmaxtimesteps = np.int32(sys.argv[5])
else:
#    envStride = 28
#    envStride = 7
    initEnvStride = 28
    envStride = 28
    fileIn = 'None'
    fileOut = 'None'
#    fileOut = './whatilearned28'
    inputmaxtimesteps = 2000
#    inputmaxtimesteps = 100

main(initEnvStride, envStride, fileIn, fileOut, inputmaxtimesteps)
    
#if __name__ == '__main__':
#    main()
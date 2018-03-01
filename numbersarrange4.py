#
# This function is called iteratively by numbersarrange_iter1.py.
#
# In this version, I used a "developmental sequence" where the agent learns to 
# perform the task first with a grid size of 28, then goes to 14, then goes to 7, 
# and finally to 4. 
#
# Adapted from numbersarrange2.py
#
# Results: The developmental sequence was very effective at learning for a 4-step
#          resolution. W/o it, learning could take hours. But, with it, learning
#          takes only 10 min or so (most of that is spent at the 4-step level).
#
#
import sys as sys
import gym
import numpy as np
import time as time
import tensorflow as tf
import tf_util_rob as U
import models as models
#from replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from replay_buffer2 import ReplayBuffer, PrioritizedReplayBuffer
from schedules import LinearSchedule
import matplotlib.pyplot as plt

#import envs.numbersarrange_env1 as envstandalone
import envs.numbersarrange_env2 as envstandalone

# **** Make tensorflow functions ****

def build_getq(make_actionDeic_ph, q_func, num_states, num_cascade, scope="deepq", qscope="q_func", reuse=None):

    with tf.variable_scope(scope, reuse=reuse):

        actions_ph = U.ensure_tf_input(make_actionDeic_ph("actions"))
        q_values = q_func(actions_ph.get(), 1, scope=qscope)
        getq = U.function(inputs=[actions_ph], outputs=q_values)
        return getq


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


#def build_getMoveActionDescriptors(make_obs_ph,actionShape,actionShapeSmall):
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
#    getMoveActionDescriptors = U.function(inputs=[observations_ph], outputs=patchesTiled)
#    getMoveActionDescriptors = U.function(inputs=[observations_ph], outputs=[patches,obsZeroPadded])
    return getMoveActionDescriptors




#def main():
def numbersarrange4(envStride, fileIn, fileOut, inputmaxtimesteps):

    reuseModels = None
    
    np.set_printoptions(formatter={'float_kind':lambda x: "%.2f" % x})

#    # Dictionary-based value function
#    q_func_tabular = {}
#
#    # cols of vectorKey must be boolean less than 64 bits long
#    def getTabularKeys(vectorKey):
#        obsBits = np.packbits(vectorKey,1)
#        obsKeys = 0
#        for i in range(np.shape(obsBits)[1]):
#            # IMPORTANT: the number of bits in the type cast below (UINT64) must be at least as big
#            # as the bits required to encode obsBits. If it is too small, we get hash collisions...
#            obsKeys = obsKeys + (256**i) * np.uint64(obsBits[:,i])
#        return obsKeys
#    
#    def getTabular(vectorKey):
#        keys = getTabularKeys(vectorKey)
#        return np.array([q_func_tabular[x] if x in q_func_tabular else 10*np.ones(num_states) for x in keys])
#    
##    def trainTabular(vectorKey,qCurrTargets,weights):
#    def trainTabular(vectorKey,qCurrTargets,weights):
#        keys = getTabularKeys(vectorKey)
#        alpha=0.2
#        for i in range(len(keys)):
#            if keys[i] in q_func_tabular:
##                q_func_tabular[keys[i]] = (1-alpha)*q_func_tabular[keys[i]] + alpha*qCurrTargets[i]
#                q_func_tabular[keys[i]] = q_func_tabular[keys[i]] + alpha*weights[i]*(qCurrTargets[i] - q_func_tabular[keys[i]]) # (1-alpha)*q_func[keys[i]] + alpha*qCurrTargets[i]
#            else:
#                q_func_tabular[keys[i]] = qCurrTargets[i]


    env = envstandalone.NumbersArrange()
    env.stride = envStride # stride input to this problem
    env.reset() # need to do the reset her in order to populate parameters
    
#    tf.reset_default_graph()
    
    # Standard q-learning parameters
#    max_timesteps=2000
    max_timesteps=inputmaxtimesteps
    exploration_fraction=0.3
    exploration_final_eps=0.1
    gamma=.90
    num_cpu = 16

    # Used by buffering and DQN
    learning_starts=60
    buffer_size=1000
    batch_size=10
    target_network_update_freq=1
    train_freq=1
    print_freq=1
    lr=0.0003

    # first two elts of deicticShape must be odd
    descriptorShape = (env.blockSize*3,env.blockSize*3,2)
#    descriptorShapeSmall = (10,10,2)
#    descriptorShapeSmall = (15,15,2)
    descriptorShapeSmall = (20,20,2)
    num_states = 2 # either holding or not
    num_patches = len(env.moveCenters)**2
    num_actions = 2*num_patches
    num_actions_discrete = 2
#    valueFunctionType = "TABULAR"
    valueFunctionType = "DQN"
#    actionSelectionStrategy = "UNIFORM_RANDOM" # actions are selected randomly from collection of all actions
    actionSelectionStrategy = "RANDOM_UNIQUE" # each unique action descriptor has equal chance of being selected

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
        convs=[(16,3,1)],
        hiddens=[32],
#        convs=[(32,3,1)],
#        hiddens=[48],
#        convs=[(48,3,1)],
#        hiddens=[48],
        dueling=True
    )

    def make_obs_ph(name):
        return U.BatchInput(env.observation_space.spaces[0].shape, name=name)

    def make_actionDeic_ph(name):
        return U.BatchInput(descriptorShapeSmall, name=name)

    def make_target_ph(name):
        return U.BatchInput([1], name=name)

    def make_weight_ph(name):
        return U.BatchInput([1], name=name)

    getMoveActionDescriptors = build_getMoveActionDescriptors(make_obs_ph=make_obs_ph,actionShape=descriptorShape,actionShapeSmall=descriptorShapeSmall,stride=env.stride)
    
    if valueFunctionType == 'DQN':
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
        
    sess = U.make_session(num_cpu)
    sess.__enter__()

    obs = env.reset()

    episode_rewards = [0.0]
    td_errors = [0.0]
    timerStart = time.time()
    U.initialize()
    
    # load prior model
    if fileIn != "None":
#        tf.reset_default_graph()
        saver = tf.train.Saver()
        saver.restore(sess, fileIn)

    for t in range(max_timesteps):
        
        # Get action set: <num_patches> pick actions followed by <num_patches> place actions
        moveDescriptors = getMoveActionDescriptors([obs[0]])
        moveDescriptors = moveDescriptors*2-1
        actionsPickDescriptors = np.stack([moveDescriptors, np.zeros(np.shape(moveDescriptors))],axis=3)
        actionsPlaceDescriptors = np.stack([np.zeros(np.shape(moveDescriptors)),moveDescriptors],axis=3)
        actionDescriptors = np.r_[actionsPickDescriptors,actionsPlaceDescriptors]

        qCurrNotHolding = getqNotHolding(actionDescriptors)
        qCurrHolding = getqHolding(actionDescriptors)
        qCurr = np.concatenate([qCurrNotHolding,qCurrHolding],axis=1)

        # select action at random
        qCurrNoise = qCurr + np.random.random(np.shape(qCurr))*0.01 # add small amount of noise to break ties randomly
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

        # take action
        new_obs, rew, done, _ = env.step(action)
        
        replay_buffer.add(obs[1], actionDescriptors[action,:], rew, np.copy(new_obs), float(done))

        if t > learning_starts and t % train_freq == 0:

            # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
            if prioritized_replay:
                beta=beta_schedule.value(t)
                states_t, actionPatches, rewards, images_tp1, states_tp1, dones, weights, batch_idxes = replay_buffer.sample(batch_size, beta)
            else:
                states_t, actionPatches, rewards, images_tp1, states_tp1, dones = replay_buffer.sample(batch_size)
                weights, batch_idxes = np.ones_like(rewards), None

            moveDescriptorsNext = getMoveActionDescriptors(images_tp1)
            moveDescriptorsNext = moveDescriptorsNext*2-1

            actionsPickDescriptorsNext = np.stack([moveDescriptorsNext, np.zeros(np.shape(moveDescriptorsNext))],axis=3)
            actionsPlaceDescriptorsNext = np.stack([np.zeros(np.shape(moveDescriptorsNext)), moveDescriptorsNext],axis=3)
            actionDescriptorsNext = np.stack([actionsPickDescriptorsNext, actionsPlaceDescriptorsNext], axis=1) # I sometimes get this axis parameter wrong... pay attention!
            actionDescriptorsNext = np.reshape(actionDescriptorsNext,[batch_size*num_patches*num_actions_discrete,descriptorShapeSmall[0],descriptorShapeSmall[1],descriptorShapeSmall[2]])

            qNextNotHolding = getqNotHolding(actionDescriptorsNext)
            qNextHolding = getqHolding(actionDescriptorsNext)
            qNextFlat = np.concatenate([qNextNotHolding,qNextHolding],axis=1)

            qNext = np.reshape(qNextFlat,[batch_size,num_patches,num_actions_discrete,num_states])
            qNextmax = np.max(np.max(qNext[range(batch_size),:,:,states_tp1],2),1)
            targets = rewards + (1-dones) * gamma * qNextmax
            
            qCurrTargetNotHolding = getqNotHolding(actionPatches)
            qCurrTargetHolding = getqHolding(actionPatches)
            qCurrTarget = np.concatenate([qCurrTargetNotHolding,qCurrTargetHolding],axis=1)

            td_error = qCurrTarget[range(batch_size),states_t] - targets
            qCurrTarget[range(batch_size),states_t] = targets

            targetTrainNotHolding(actionPatches, np.reshape(qCurrTarget[:,0],[batch_size,1]), np.reshape(weights,[batch_size,1]))
            targetTrainHolding(actionPatches, np.reshape(qCurrTarget[:,1],[batch_size,1]), np.reshape(weights,[batch_size,1]))

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
#        mean_100ep_tderror = round(np.mean(td_errors[-51:-1]), 1)
        num_episodes = len(episode_rewards)
        if done and print_freq is not None and len(episode_rewards) % print_freq == 0:
            timerFinal = time.time()
#            print("steps: " + str(t) + ", episodes: " + str(num_episodes) + ", mean 100 episode reward: " + str(mean_100ep_reward) + ", % time spent exploring: " + str(int(100 * exploration.value(t))) + ", time elapsed: " + str(timerFinal - timerStart) + ", tderror: " + str(mean_100ep_tderror))
            print("steps: " + str(t) + ", episodes: " + str(num_episodes) + ", mean 100 episode reward: " + str(mean_100ep_reward) + ", % time spent exploring: " + str(int(100 * exploration.value(t))) + ", time elapsed: " + str(timerFinal - timerStart))
            timerStart = timerFinal
        
        obs = np.copy(new_obs)

    # save what we learned
#    fname = "./whatwelearned2"
#    fname = "./whatwelearned3"
#    fname = "./whatwelearned4"
    if fileOut != "None":
        saver = tf.train.Saver()
        saver.save(sess, fileOut)

    # display value function
    obs = env.reset()
    moveDescriptors = getMoveActionDescriptors([obs[0]])
    moveDescriptors = moveDescriptors*2-1
    gridSize = np.int32(np.sqrt(np.shape(moveDescriptors)[0]))

    actionsPickDescriptors = np.stack([moveDescriptors, np.zeros(np.shape(moveDescriptors))],axis=3)
    actionsPlaceDescriptors = np.stack([np.zeros(np.shape(moveDescriptors)), moveDescriptors],axis=3)
    
    print(str(obs[0][:,:,0]))
    
    qPickNotHolding = getqNotHolding(actionsPickDescriptors)
    qPickHolding = getqHolding(actionsPickDescriptors)
    qPick = np.concatenate([qPickNotHolding,qPickHolding],axis=1)
    print("Value function for pick action in hold-nothing state:")
    print(str(np.reshape(qPick[:,0],[gridSize,gridSize])))
    print("Value function for pick action in hold-1 state:")
    print(str(np.reshape(qPick[:,1],[gridSize,gridSize])))

    qPlaceNotHolding = getqNotHolding(actionsPlaceDescriptors)
    qPlaceHolding = getqHolding(actionsPlaceDescriptors)
    qPlace = np.concatenate([qPlaceNotHolding,qPlaceHolding],axis=1)
    print("Value function for place action in hold-nothing state:")
    print(str(np.reshape(qPlace[:,0],[gridSize,gridSize])))
    print("Value function for place action in hold-1 state:")
    print(str(np.reshape(qPlace[:,1],[gridSize,gridSize])))
    
    plt.subplot(1,3,1)
    plt.imshow(np.tile(env.state[0],[1,1,3]))
    plt.subplot(1,3,2)
    plt.imshow(np.reshape(qPick[:,0],[gridSize,gridSize]))
    plt.subplot(1,3,3)
    plt.imshow(np.reshape(qPlace[:,1],[gridSize,gridSize]))
    plt.show()

    
envStride = np.int32(sys.argv[1])
fileIn = sys.argv[2]
fileOut = sys.argv[3]
inputmaxtimesteps = np.int32(sys.argv[4])

numbersarrange4(envStride, fileIn, fileOut, inputmaxtimesteps)
    
#if __name__ == '__main__':
#    main()


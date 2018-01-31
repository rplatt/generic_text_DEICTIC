#
# In this version, I create two dqn networks: one for pick and the other for place.
# This version of things works very well for batch_size=1, buffer_size=1.
#
# Adapted from blockarrange8_old.py
#
# I think I have some kind of problem w/ my replay buffer...
#
import gym
import numpy as np
import time as time
import tensorflow as tf
import tf_util_rob as U
import models as models
#import models2 as models
from replay_buffer2 import ReplayBuffer, PrioritizedReplayBuffer
from schedules import LinearSchedule

import envs.blockarrange2_rewardonsuccess_standalone as envstandalone

# **** Make tensorflow functions ****

def build_getq(make_actionDeic_ph, q_func, num_states, num_cascade, scope="deepq", qscope="q_func", reuse=None):

    with tf.variable_scope(scope, reuse=reuse):

        actions_ph = U.ensure_tf_input(make_actionDeic_ph("actions"))
#        q_values = q_func(actions_ph.get(), num_states, scope=qscope)
        q_values = q_func(actions_ph.get(), 1, scope=qscope)
#        q_valuesTiled = tf.reshape(q_values,[-1,num_states])
#        getq = U.function(inputs=[actions_ph], outputs=q_valuesTiled)
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
#        q_t_raw = q_func(obs_t_input.get(), num_states*num_cascade, scope=qscope, reuse=True)
#        targetTiled = tf.reshape(target_input.get(), shape=(-1,num_cascade*num_states))
#        q_t_raw = q_func(obs_t_input.get(), num_states, scope=qscope, reuse=True)
#        targetTiled = tf.reshape(target_input.get(), shape=(-1,num_states))
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
    
#        optimize_expr = optimizer.minimize(errors, var_list=q_func_vars)
        
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
#            observations_ph.get(),
            obsZeroPadded,
            ksizes=[1, deicticShape[0], deicticShape[1], 1], 
            strides=[1, 1, 1, 1], 
            rates=[1, 1, 1, 1], 
            padding='VALID')
    patchesShape = tf.shape(patches)
    patchesTiled = tf.reshape(patches,[patchesShape[0]*patchesShape[1]*patchesShape[2],deicticShape[0],deicticShape[1]])
#    patchesTiledStacked = tf.stack([tf.equal(patchesTiled,1), tf.equal(patchesTiled,2)],axis=-1)
#    X,Y = tf.meshgrid(tf.range(shape[1]),tf.range(shape[2]))
#    moveActions = tf.stack([tf.reshape(Y,[shape[1]*shape[2],]), tf.reshape(X,[shape[1]*shape[2],])],axis=0)
#    getMoveActionDescriptors = U.function(inputs=[observations_ph], outputs=patchesTiledStacked)
    getMoveActionDescriptors = U.function(inputs=[observations_ph], outputs=patchesTiled)
    return getMoveActionDescriptors



def main():
    
    np.set_printoptions(formatter={'float_kind':lambda x: "%.2f" % x})
    
    # Define environment
    env = envstandalone.BlockArrange()

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
#        return np.array([q_func[x] if x in q_func else 0*np.ones(num_states) for x in keys])
        return np.array([q_func_tabular[x] if x in q_func_tabular else 10*np.ones(num_states) for x in keys])
    
    def trainTabular(vectorKey,qCurrTargets,weights):
        keys = getTabularKeys(vectorKey)
        alpha=0.2
        for i in range(len(keys)):
            if keys[i] in q_func_tabular:
#                q_func[keys[i]] = (1-alpha)*q_func[keys[i]] + alpha*qCurrTargets[i]
                q_func_tabular[keys[i]] = q_func_tabular[keys[i]] + alpha*weights[i,:]*(qCurrTargets[i] - q_func_tabular[keys[i]]) # (1-alpha)*q_func[keys[i]] + alpha*qCurrTargets[i]
            else:
                q_func_tabular[keys[i]] = qCurrTargets[i]


    # Standard DQN parameters
#    max_timesteps=20000
    max_timesteps=3000
#    max_timesteps=2000
#    learning_starts=1000
    learning_starts=100
#    learning_starts=10
#    buffer_size=50000
#    buffer_size=10000
#    buffer_size=1000
#    buffer_size=320
#    buffer_size=32
#    buffer_size=2
    buffer_size=1
#    exploration_fraction=0.2
    exploration_fraction=0.3
#    exploration_final_eps=0.02
    exploration_final_eps=0.1
    print_freq=1
#    gamma=.98
    gamma=.9
    target_network_update_freq=1
#    batch_size=32
#    batch_size=2
    batch_size=1
    train_freq=1
#    train_freq=2
    num_cpu = 16
#    lr=0.001
    lr=0.0003
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
    
    # Deictic state/action parameters
    deicticShape = (3,3,2) # IMPORTANT: first two elts of deicticShape must be odd
    deicticActionShape = (3,3,2)
    num_cascade = 5
#    num_states = env.num_blocks + 1 # one more state than blocks to account for not holding anything
    num_states = 2 # either holding or not
    num_patches = env.maxSide**2
    num_actions = 2*num_patches
    num_actions_discrete = 2
    valueFunctionType = "TABULAR"
#    valueFunctionType = "DQN"
#    actionSelectionStrategy = "UNIFORM_RANDOM" # actions are selected randomly from collection of all actions
    actionSelectionStrategy = "RANDOM_UNIQUE" # each unique action descriptor has equal chance of being selected
    
    # ******* Build tensorflow functions ********

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

    def make_actionDeic_ph(name):
        return U.BatchInput(deicticActionShape, name=name)

    def make_target_ph(name):
#        return U.BatchInput([num_actions], name=name)
#        return U.BatchInput([num_cascade,num_states], name=name)
#        return U.BatchInput([num_states], name=name)
        return U.BatchInput([1], name=name)
#        return U.BatchInput(1, name=name)

    def make_weight_ph(name):
#        return U.BatchInput([num_states], name=name)
        return U.BatchInput([1], name=name)

    getMoveActionDescriptors = build_getMoveActionDescriptors(make_obs_ph=make_obs_ph,deicticShape=deicticShape)

    if valueFunctionType == 'DQN':
        getqNotHolding = build_getq(
                make_actionDeic_ph=make_actionDeic_ph,
                q_func=q_func,
                num_states=num_states,
                num_cascade=num_cascade,
                scope="deepq",
                qscope="q_func_notholding"
                )
        getqHolding = build_getq(
                make_actionDeic_ph=make_actionDeic_ph,
                q_func=q_func,
                num_states=num_states,
                num_cascade=num_cascade,
                scope="deepq",
                qscope="q_func_holding"
                )
    
        targetTrainNotHolding = build_targetTrain(
            make_actionDeic_ph=make_actionDeic_ph,
            make_target_ph=make_target_ph,
            make_weight_ph=make_weight_ph,
            q_func=q_func,
            num_states=num_states,
            num_cascade=num_cascade,
            optimizer=tf.train.AdamOptimizer(learning_rate=lr),
    #        optimizer=tf.train.GradientDescentOptimizer(learning_rate=lr),
            scope="deepq", 
            qscope="q_func_notholding",
            grad_norm_clipping=1.
        )

        targetTrainHolding = build_targetTrain(
            make_actionDeic_ph=make_actionDeic_ph,
            make_target_ph=make_target_ph,
            make_weight_ph=make_weight_ph,
            q_func=q_func,
            num_states=num_states,
            num_cascade=num_cascade,
            optimizer=tf.train.AdamOptimizer(learning_rate=lr),
    #        optimizer=tf.train.GradientDescentOptimizer(learning_rate=lr),
            scope="deepq", 
            qscope="q_func_holding",
            grad_norm_clipping=1.
        )

    # Start tensorflow session
    sess = U.make_session(num_cpu)
    sess.__enter__()

    episode_rewards = [0.0]
    timerStart = time.time()
    U.initialize()
    obs = env.reset()
    for t in range(max_timesteps):

        # Get state: in range(0,env.num_blocks)
        stateDeictic = np.int32(obs[1]>0) # holding

        # Get action set: <num_patches> pick actions followed by <num_patches> place actions
        moveDescriptorsRaw = getMoveActionDescriptors([obs[0]])
        moveDescriptors = np.int32(moveDescriptorsRaw>0)
        moveDescriptors = moveDescriptors*2-1

        actionsPickDescriptors = np.stack([moveDescriptors, np.zeros(np.shape(moveDescriptors))],axis=3)
        actionsPlaceDescriptors = np.stack([np.zeros(np.shape(moveDescriptors)),moveDescriptors],axis=3)
        actionDescriptors = np.r_[actionsPickDescriptors,actionsPlaceDescriptors]

        if valueFunctionType == "TABULAR":
            actionDescriptorsFlat = np.reshape(actionDescriptors,[-1,deicticActionShape[0]*deicticActionShape[1]*deicticActionShape[2]]) == 1
            qCurr = getTabular(actionDescriptorsFlat)
        else:
            qCurrNotHolding = getqNotHolding(actionDescriptors)
            qCurrHolding = getqHolding(actionDescriptors)
            qCurr = np.concatenate([qCurrNotHolding,qCurrHolding],axis=1)
        qCurrNoise = qCurr + np.random.random(np.shape(qCurr))*0.01 # add small amount of noise to break ties randomly

        # select action at random
        if actionSelectionStrategy == "UNIFORM_RANDOM":
            action = np.argmax(qCurrNoise[:,stateDeictic])
            if np.random.rand() < exploration.value(t):
                action = np.random.randint(num_actions)
        elif actionSelectionStrategy == "RANDOM_UNIQUE":
            _,idx,inv = np.unique(actionDescriptors,axis=0,return_index=True,return_inverse=True)
            actionIdx = np.argmax(qCurrNoise[idx,stateDeictic])
            if np.random.rand() < exploration.value(t):
                actionIdx = np.random.randint(len(idx))
            actionsSelected = np.nonzero(inv==actionIdx)[0]
            action = actionsSelected[np.random.randint(len(actionsSelected))]
        else:
            print("Error...")

        # take action
        new_obs, rew, done, _ = env.step(action)
        
#        if new_obs[1] > 0:
#            new_obs
            
        replay_buffer.add(stateDeictic, actionDescriptors[action,:], rew, new_obs, float(done))

        if t > learning_starts and t % train_freq == 0:

            # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
            if prioritized_replay:
                beta=beta_schedule.value(t)
                states_t, actions, rewards, images_tp1, states_tp1, dones, weights, batch_idxes = replay_buffer.sample(batch_size, beta)
            else:
                states_t, actions, rewards, images_tp1, states_tp1, dones = replay_buffer.sample(batch_size)
                weights, batch_idxes = np.ones_like(rewards), None

            states_tp1 = np.int32(states_tp1>0)

            moveDescriptorsNext1 = getMoveActionDescriptors(images_tp1)
            moveDescriptorsNext1 = np.int32(moveDescriptorsNext1>0)
            moveDescriptorsNext1 = moveDescriptorsNext1*2-1

            actionsPickDescriptorsNext1 = np.stack([moveDescriptorsNext1, np.zeros(np.shape(moveDescriptorsNext1))],axis=3)
            actionsPlaceDescriptorsNext1 = np.stack([np.zeros(np.shape(moveDescriptorsNext1)), moveDescriptorsNext1],axis=3)
            actionDescriptorsNext1 = np.stack([actionsPickDescriptorsNext1, actionsPlaceDescriptorsNext1], axis=0)
            actionDescriptorsNext1 = np.reshape(actionDescriptorsNext1,[batch_size*num_patches*num_actions_discrete,deicticActionShape[0],deicticActionShape[1],deicticActionShape[2]])
            
            if valueFunctionType == "TABULAR":
                actionDescriptorsNextFlat1 = np.reshape(actionDescriptorsNext1,[batch_size*num_patches*num_actions_discrete,-1]) == 1
                qNextFlat1 = getTabular(actionDescriptorsNextFlat1)
            else:
                qNextNotHolding = getqNotHolding(actionDescriptorsNext1)
                qNextHolding = getqHolding(actionDescriptorsNext1)
                qNextFlat1 = np.concatenate([qNextNotHolding,qNextHolding],axis=1)
            
            qNext1 = np.reshape(qNextFlat1,[batch_size,num_patches,num_actions_discrete,num_states])
            qNextmax1 = np.max(np.max(qNext1[range(batch_size),:,:,states_tp1],2),1)
            targets1 = rewards + (1-dones) * gamma * qNextmax1

            if valueFunctionType == "TABULAR":
                actionsFlat = np.reshape(actions,[batch_size,-1]) == 1
                qCurrTarget1 = getTabular(actionsFlat)
            else:
                qCurrTargetNotHolding = getqNotHolding(actions)
                qCurrTargetHolding = getqHolding(actions)
                qCurrTarget1 = np.concatenate([qCurrTargetNotHolding,qCurrTargetHolding],axis=1)
#                qCurrTarget1 = getq(actions)

            td_errors = qCurrTarget1[range(batch_size),states_t] - targets1
            qCurrTarget1[range(batch_size),states_t] = targets1

            if valueFunctionType == "TABULAR":
                trainTabular(actionsFlat, qCurrTarget1, np.transpose(np.tile(weights,[num_states,1]))) # (TABULAR)
            else:
                targetTrainNotHolding(actions, np.reshape(qCurrTarget1[:,0],[batch_size,1]), np.reshape(weights,[batch_size,1]))
                targetTrainHolding(actions, np.reshape(qCurrTarget1[:,1],[batch_size,1]), np.reshape(weights,[batch_size,1]))

            if prioritized_replay:
                new_priorities = np.abs(td_errors) + prioritized_replay_eps
                replay_buffer.update_priorities(batch_idxes, new_priorities)


        # bookkeeping for storing episode rewards
        episode_rewards[-1] += rew
        if done:
            new_obs = env.reset()
            episode_rewards.append(0.0)
        mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
        num_episodes = len(episode_rewards)
        if done and print_freq is not None and len(episode_rewards) % print_freq == 0:
            timerFinal = time.time()
            print("steps: " + str(t) + ", episodes: " + str(num_episodes) + ", mean 100 episode reward: " + str(mean_100ep_reward) + ", % time spent exploring: " + str(int(100 * exploration.value(t))) + ", beta: " + str(beta) + ", time elapsed: " + str(timerFinal - timerStart))
            timerStart = timerFinal
        
        obs = np.copy(new_obs)
        
    # display value function
    obs = env.reset()
    moveDescriptorsRaw = getMoveActionDescriptors([obs[0]])
    moveDescriptors = np.int32(moveDescriptorsRaw>0)
    moveDescriptors = moveDescriptors*2-1

    actionsPickDescriptors = np.stack([moveDescriptors, np.zeros(np.shape(moveDescriptors))],axis=3)
    actionsPlaceDescriptors = np.stack([np.zeros(np.shape(moveDescriptors)), moveDescriptors],axis=3)
    
    print(str(obs[0][:,:,0]))
    
#    qPick = getq(actionsPickDescriptors)
    qPickNotHolding = getqNotHolding(actionsPickDescriptors)
    qPickHolding = getqHolding(actionsPickDescriptors)
    qPick = np.concatenate([qPickNotHolding,qPickHolding],axis=1)
    
#    qPick = getTabular(np.reshape(actionsPickDescriptors,[num_patches,-1])==1)
    print("Value function for pick action in hold-nothing state:")
    print(str(np.reshape(qPick[:,0],[8,8])))
    print("Value function for pick action in hold-1 state:")
    print(str(np.reshape(qPick[:,1],[8,8])))

#    qPlace = getq(actionsPlaceDescriptors)
    qPlaceNotHolding = getqNotHolding(actionsPlaceDescriptors)
    qPlaceHolding = getqHolding(actionsPlaceDescriptors)
    qPlace = np.concatenate([qPlaceNotHolding,qPlaceHolding],axis=1)    
#    qPlace = getTabular(np.reshape(actionsPlaceDescriptors,[num_patches,-1])==1)
    print("Value function for place action in hold-nothing state:")
    print(str(np.reshape(qPlace[:,0],[8,8])))
    print("Value function for place action in hold-1 state:")
    print(str(np.reshape(qPlace[:,1],[8,8])))
    


if __name__ == '__main__':
    main()


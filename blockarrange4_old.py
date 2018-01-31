#
# Implementation of the tabular version of the cascade.
#
# Adapted from blockarrange3.py
#
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

import envs.blockarrange1_standalone as envstandalone

# **** Make tensorflow functions ****

def build_getq(make_actionDeic_ph, q_func, num_states, num_cascade, scope="deepq", qscope="q_func", reuse=None):

    with tf.variable_scope(scope, reuse=reuse):

        actions_ph = U.ensure_tf_input(make_actionDeic_ph("actions"))
        q_values = q_func(actions_ph.get(), num_states*num_cascade, scope=qscope)
        q_valuesTiled = tf.reshape(q_values,[-1,num_cascade,num_states])
        getq = U.function(inputs=[actions_ph], outputs=q_valuesTiled)
        return getq


#def build_targetTrain(make_actionDeic_ph, 
#                        make_target_ph,
#                        q_func,
#                        num_states,
#                        num_cascade,
#                        optimizer,
#                        scope="deepq", 
#                        qscope="q_func",
#                        grad_norm_clipping=None,
#                        reuse=None):
#
#    with tf.variable_scope(scope, reuse=reuse):
#        
#        # set up placeholders
#        obs_t_input = U.ensure_tf_input(make_actionDeic_ph("action_t_deic"))
#        target_input = U.ensure_tf_input(make_target_ph("target"))
#    
#        # get variables
#        q_func_vars = U.scope_vars(U.absolute_scope_name(qscope))
#    
#        # q values for all actions
#        q_t_raw = q_func(obs_t_input.get(), num_actions*num_cascade, scope=qscope, reuse=True)
#    
#        targetTiled = tf.reshape(target_input.get(), shape=(-1,num_cascade*num_actions))
#        
#        # calculate error
#        td_error = q_t_raw - tf.stop_gradient(targetTiled)
#        errors = U.huber_loss(td_error)
#
#        # compute optimization op (potentially with gradient clipping)
#        if grad_norm_clipping is not None:
#            optimize_expr = U.minimize_and_clip(optimizer,
#                                                errors,
#                                                var_list=q_func_vars,
#                                                clip_val=grad_norm_clipping)
#        else:
#            optimize_expr = optimizer.minimize(errors, var_list=q_func_vars)
#    
##        optimize_expr = optimizer.minimize(errors, var_list=q_func_vars)
#        
#        targetTrain = U.function(
#            inputs=[
#                obs_t_input,
#                target_input
#            ],
#            outputs=[td_error, obs_t_input.get(), target_input.get()],
#            updates=[optimize_expr]
#        )
#    
#        return targetTrain
    
    
def build_getMoveActionDescriptors(make_obs_ph,deicticShape):
    
    if (deicticShape[0] % 2 == 0) or (deicticShape[1] % 2 == 0):
        print("build_getActionDescriptors ERROR: first two elts of deicticShape must be odd")
        
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
#        return np.array([q_func[x] if x in q_func else 0*np.ones(num_states) for x in keys])
        return np.array([q_func[x] if x in q_func else 0*np.ones([num_cascade,num_states]) for x in keys])
    
    def trainTabular(vectorKey,qCurrTargets):
        keys = getTabularKeys(vectorKey)
        alpha=0.3
        for i in range(len(keys)):
            if keys[i] in q_func:
                q_func[keys[i]] = (1-alpha)*q_func[keys[i]] + alpha*qCurrTargets[i]
            else:
                q_func[keys[i]] = qCurrTargets[i]


    # Standard DQN parameters
    max_timesteps=40000
    learning_starts=1000
#    learning_starts=10
#    buffer_size=50000
#    buffer_size=10000
    buffer_size=1000
#    buffer_size=100
#    buffer_size=2
    exploration_fraction=0.2
    exploration_final_eps=0.02
    print_freq=1
    gamma=.98
    target_network_update_freq=1
    batch_size=32
#    batch_size=8
    train_freq=1
    num_cpu = 16
    lr=0.001
    exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * max_timesteps),
                                 initial_p=1.0,
                                 final_p=exploration_final_eps)
    replay_buffer = ReplayBuffer(buffer_size)

    # Deictic state/action parameters
    deicticShape = (3,3,2) # IMPORTANT: first two elts of deicticShape must be odd
    deicticActionShape = (3,3,4) # IMPORTANT: first two elts of deicticShape must be odd
    num_cascade = 5
    num_states = env.num_blocks + 1 # one more state than blocks to account for not holding anything
    num_patches = env.maxSide**2
    num_actions = 2*num_patches

    # ******* Build tensorflow functions ********
    
#    q_func = models.cnn_to_mlp(
#        convs=[(32,3,1)],
#        hiddens=[32],
#        dueling=True
#    )

    def make_obs_ph(name):
        return U.BatchInput(env.observation_space.spaces[0].shape, name=name)
    
    def make_actionDeic_ph(name):
        return U.BatchInput(deicticActionShape, name=name)

    def make_target_ph(name):
#        return U.BatchInput([num_actions], name=name)
        return U.BatchInput([num_cascade,num_states], name=name)

    getMoveActionDescriptors = build_getMoveActionDescriptors(make_obs_ph=make_obs_ph,deicticShape=deicticActionShape)

#    getq = build_getq(
#            make_actionDeic_ph=make_actionDeic_ph,
#            q_func=q_func,
#            num_states=num_states,
#            num_cascade=num_cascade,
#            scope="deepq",
#            qscope="q_func"
#            )

#    targetTrain = build_targetTrain(
#        make_actionDeic_ph=make_actionDeic_ph,
#        make_target_ph=make_target_ph,
#        q_func=q_func,
#        num_states=env.observation_space.spaces[1].n,
#        num_cascade=num_cascade,
#        optimizer=tf.train.AdamOptimizer(learning_rate=lr),
##        optimizer=tf.train.GradientDescentOptimizer(learning_rate=lr),
#        scope="deepq", 
#        qscope="q_func",
#        grad_norm_clipping=1.
#    )

    # Start tensorflow session
    sess = U.make_session(num_cpu)
    sess.__enter__()

    episode_rewards = [0.0]
    timerStart = time.time()
#    U.initialize()
    obs = env.reset()
    for t in range(max_timesteps):
        
        # Get state: in range(0,env.num_blocks)
        stateDeictic = obs[1] # obj in hand

        # Get action set: <num_patches> pick actions followed by <num_patches> place actions

        # TABULAR version -- get q-values
        moveDescriptors = np.reshape(getMoveActionDescriptors([obs[0]]),[-1,deicticShape[0]*deicticShape[1]*deicticShape[2]])
        actionDescriptors = np.r_[np.c_[np.zeros([num_patches,1])==1,moveDescriptors],np.c_[np.ones([num_patches,1])==1,moveDescriptors]]
        qCurr = getTabular(actionDescriptors)

#        # DQN version -- get q-values
#        moveDescriptors = getMoveActionDescriptors([obs[0]])
#        actionsPickDescriptors = np.concatenate([moveDescriptors,np.zeros(np.shape(moveDescriptors))],axis=3)
#        actionsPlaceDescriptors = np.concatenate([np.zeros(np.shape(moveDescriptors)),moveDescriptors],axis=3)
#        actionDescriptors = np.r_[actionsPickDescriptors,actionsPlaceDescriptors]
#        qCurr = getq(actionDescriptors)

        
        # select action
        qCurrNoise = qCurr + np.random.random(np.shape(qCurr))*0.01 # add small amount of noise to break ties randomly
#        action = np.argmax(qCurrNoise[:,stateDeictic])
        action = np.argmax(qCurrNoise[:,-1,stateDeictic]) # USE CASCADE
        if np.random.rand() < exploration.value(t):
            action = np.random.randint(num_actions)

        # take action
        new_obs, rew, done, _ = env.step(action)
        replay_buffer.add(stateDeictic, actionDescriptors[action,:], rew, new_obs, float(done))

        # sample from replay buffer and train
        if t > learning_starts and t % train_freq == 0:

            states_t, actions, rewards, images_tp1, states_tp1, dones = replay_buffer.sample(batch_size)
            
            moveDescriptorsNextTiled = np.reshape(getMoveActionDescriptors(images_tp1),[batch_size,num_patches,deicticShape[0]*deicticShape[1]*deicticShape[2]])
            actionDescriptorsNextTiled = np.stack(
                    [np.c_[np.zeros([batch_size,num_patches,1])==1,moveDescriptorsNextTiled],
                    np.c_[np.ones([batch_size,num_patches,1])==1,moveDescriptorsNextTiled]]
                    ,axis=1)
            actionDescriptorsNext = np.reshape(actionDescriptorsNextTiled,[batch_size*2*num_patches,-1])
            
            qNext = getTabular(actionDescriptorsNext)
            
            states_tp1Full = np.repeat(states_tp1,2*num_patches)
            
            
            qNextTiled = np.reshape(qNext[range(2*batch_size*num_patches),-1,states_tp1Full],[batch_size,2,num_patches,-1])
            qNextmax = np.max(np.max(np.max(qNextTiled,3),2),1)
            
            targets = rewards + (1-dones) * gamma * qNextmax

#            qCurrTarget = getTabular(actions)
#            qCurrTarget[range(batch_size),states_tp1] = np.minimum(qCurrTarget[range(batch_size),states_tp1], targets)
#            trainTabular(actions,qCurrTarget)

            qCurr = getTabular(actions)
            qCurrTarget = np.copy(qCurr)
            qCurrTarget[range(batch_size),0,states_tp1] = targets
            for i in range(num_cascade-1):
                mask = targets < qCurr[range(batch_size),i,states_tp1]
                qCurrTarget[range(batch_size),i+1,states_tp1] = \
                    mask*targets + \
                    (1-mask)*qCurrTarget[range(batch_size),i+1,states_tp1]
            trainTabular(actions,qCurrTarget)


            # ********************************************
            
#            # Sample from replay buffer
#            states_t, actions, rewards, images_tp1, states_tp1, dones = replay_buffer.sample(batch_size)
#
#            # Get action set: <num_patches> pick actions followed by <num_patches> place actions
#            moveDescriptorsNext = np.reshape(getMoveActionDescriptors(images_tp1),[batch_size,num_patches,deicticShape[0]*deicticShape[1]*deicticShape[2]])
#            actionDescriptorsNext = np.stack([np.c_[np.zeros([batch_size,num_patches,1])==1,moveDescriptorsNext], 
#                            np.c_[np.ones([batch_size,num_patches,1])==1,moveDescriptorsNext]],
#                            axis=1)
#            actionDescriptorsNext = np.reshape(actionDescriptorsNext,[batch_size*2*num_patches,-1])
#            
#            # Get targets
#            qNext = getTabular(actionDescriptorsNext)
#            np.repeat(states_tp1,2*num_patches)
#            qNextAtState = qNext[range(batch_size*2*num_patches),np.repeat(states_tp1,2*num_patches)]
#            qNextTiled = np.reshape(qNextAtState,[batch_size,2*num_patches])
#            qNextmax = np.max(qNextTiled,1)
#            targets = rewards + (1-dones) * gamma * qNextmax
#            
#            qCurrTarget = getTabular(actions)
#            qCurrTarget[range(batch_size),states_t] = targets
#            trainTabular(actions,qCurrTarget)
            
            # ********************************************

#        # Get state: in range(0,env.num_blocks)
#        stateDeicticNext = new_obs[1] # holding
#
#        # Get action set: <num_patches> pick actions followed by <num_patches> place actions
#        moveDescriptorsNext = np.reshape(getMoveActionDescriptors([new_obs[0]]),[-1,deicticShape[0]*deicticShape[1]*deicticShape[2]])
#        actionDescriptorsNext = np.r_[np.c_[np.zeros([num_patches,1])==1,moveDescriptorsNext],np.c_[np.ones([num_patches,1])==1,moveDescriptorsNext]]
#
#        # Calculate TD target
#        qNext = getTabular(actionDescriptorsNext)
#        qNextmax = np.max(qNext[:,stateDeicticNext])
#        target = rew + (1-done) * gamma * qNextmax
#
#        # Update dictionary value function
#        qCurrTarget = qCurr[action,:]
#        qCurrTarget[stateDeictic] = np.minimum(qCurrTarget[stateDeictic], target)
#        trainTabular([actionDescriptors[action,:]],[qCurrTarget])



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


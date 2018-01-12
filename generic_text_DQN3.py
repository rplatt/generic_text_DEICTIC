#
# Adapted from generic_text4.py. Straight DQN version.
#
# This version gets similar performance to generic_text_DQN2.py
#
import gym
import numpy as np
import time as time
import tensorflow as tf
import tf_util_rob as U
import models2 as models
from replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from schedules import LinearSchedule
#import matplotlib.pyplot as plt

import envs.multi_ghost_evade1_standalone as envstandalone
#import envs.ghost_evade1_standalone as envstandalone
#import envs.ballcatch2_standalone as envstandalone

# **** Make tensorflow functions ****


def build_get_2channelobs(make_obs_ph):
    observations_ph = U.ensure_tf_input(make_obs_ph("observation"))
    patchesTiledStacked = tf.stack([tf.equal(observations_ph.get()[:,:,:,0],1), tf.equal(observations_ph.get()[:,:,:,0],2)],axis=-1)
    getDeic = U.function(inputs=[observations_ph], outputs=patchesTiledStacked)
    return getDeic


def build_getq_DQN(make_obs_ph, q_func, num_actions, scope="deepq", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        observations_ph = U.ensure_tf_input(make_obs_ph("observation"))
        q_values = q_func(observations_ph.get(), num_actions, scope="q_func")
        getq = U.function(inputs=[observations_ph], outputs=q_values)
        return getq

def build_targetTrain_DQN(make_obs_ph, 
                        make_target_ph,
                        q_func,
                        num_actions,
                        optimizer,
                        scope="deepq", 
                        reuse=None):

    with tf.variable_scope(scope, reuse=reuse):
        
        # set up placeholders
        obs_t_input = U.ensure_tf_input(make_obs_ph("obs_t"))
        target_input = U.ensure_tf_input(make_target_ph("target"))
    
        # get variables
        q_func_vars = U.scope_vars(U.absolute_scope_name("q_func"))
    
        # q values for all actions
        q_t_raw = q_func(obs_t_input.get(), num_actions, scope="q_func", reuse=True)
            
        # calculate error
        td_error = q_t_raw - tf.stop_gradient(target_input.get())
        errors = U.huber_loss(td_error)
    
        optimize_expr = optimizer.minimize(errors, var_list=q_func_vars)
        
        targetTrain = U.function(
            inputs=[
                obs_t_input,
                target_input
            ],
            outputs=[td_error],
            updates=[optimize_expr]
        )
    
        return targetTrain
    
def main():

    env = envstandalone.MultiGhostEvade()
#    env = envstandalone.GhostEvade()
#    env = envstandalone.BallCatch()
    
    max_timesteps=40000
    learning_starts=1000
    buffer_size=50000
#    exploration_fraction=0.2
    exploration_fraction=0.4
    exploration_final_eps=0.02
    print_freq=10
    gamma=.98
#    target_network_update_freq=500
#    target_network_update_freq=100
#    target_network_update_freq=10
    target_network_update_freq=1
    learning_alpha = 0.2
    
    batch_size=32
    train_freq=1

    obsShape = (8,8,1)
#    obsShape = (8,8,2)
#    deicticShape = (3,3,2)
#    deicticShape = (3,3,4)
#    deicticShape = (4,4,2)
#    deicticShape = (4,4,4)
    deicticShape = (8,8,2)
#    num_deictic_patches = 36
#    num_deictic_patches = 25
    num_deictic_patches = 1

#    num_actions = 4
#    num_actions = 3
    num_actions = env.action_space.n

    episode_rewards = [0.0]
    num_cpu=16
    num_cascade = 5
    
    # Create the schedule for exploration starting from 1.
    exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * max_timesteps),
                                 initial_p=1.0,
                                 final_p=exploration_final_eps)


    # CNN version
    # conv model parameters: (num_outputs, kernel_size, stride)
    model = models.cnn_to_mlp(
#    model = models.cnn_to_mlp_2pathways(
#        convs=[(16,3,1)],
        convs=[(32,3,1)],
#        convs=[(32,4,1)],
#        convs=[(16,4,1)],
        hiddens=[16],
        dueling=True
    )
    
    # MLP version
#    model = models.mlp([8, 16])
#    model = models.mlp([16, 16])
#    model = models.mlp([16, 32])
#    model = models.mlp([16, 16])
#    model = models.mlp([32, 32])

    q_func=model
    lr=0.001
    
    def make_obs_ph(name):
        return U.BatchInput(obsShape, name=name)
    
#    def make_obsDeic_ph(name):
#        return U.BatchInput(deicticShape, name=name)
        
    def make_target_ph(name):
        return U.BatchInput([num_actions], name=name)
#        return U.BatchInput([num_cascade,num_actions], name=name)

    sess = U.make_session(num_cpu)
    sess.__enter__()

    getq = build_getq_DQN(
            make_obs_ph=make_obs_ph,
            q_func=q_func,
            num_actions=num_actions
            )
        
    targetTrain = build_targetTrain_DQN(
        make_obs_ph=make_obs_ph,
        make_target_ph=make_target_ph,
        q_func=q_func,
        num_actions=env.action_space.n,
        optimizer=tf.train.AdamOptimizer(learning_rate=lr)
    )

    get_2channelobs = build_get_2channelobs(make_obs_ph=make_obs_ph)
    
#    getq = build_getq(
#            make_obsDeic_ph=make_obsDeic_ph,
#            q_func=q_func,
#            num_actions=num_actions,
#            num_cascade=num_cascade,
#            scope="deepq",
#            qscope="q_func"
#            )
#    
#    getqTarget = build_getq(
#            make_obsDeic_ph=make_obsDeic_ph,
#            q_func=q_func,
#            num_actions=num_actions,
#            num_cascade=num_cascade,
#            scope="deepq",
#            qscope="q_func_target"
#            )
#
#    update_target = build_update_target(scope="deepq", 
#                                        qscope="q_func",
#                                        qscopeTarget="q_func_target")
#                      
#    targetTrain = build_targetTrain(
#        make_obsDeic_ph=make_obsDeic_ph,
#        make_target_ph=make_target_ph,
#        q_func=q_func,
#        num_actions=env.action_space.n,
#        num_cascade=num_cascade,
#        optimizer=tf.train.AdamOptimizer(learning_rate=lr),
#        scope="deepq", 
#        qscope="q_func"
#    )
#    
#    getDeic = build_getDeic_Foc(make_obs_ph=make_obs_ph,deicticShape=deicticShape)
##    getDeic = build_getDeic_FocCoarse(make_obs_ph=make_obs_ph,deicticShape=deicticShape)
    
    # Initialize the parameters and copy them to the target network.
    U.initialize()
    
    replay_buffer = ReplayBuffer(buffer_size)
    obs = env.reset()

    timerStart = time.time()
    for t in range(max_timesteps):
        
#        obs2channel = get_2channelobs([obs])
        
        # CNN version
        qCurr = getq(np.array([obs]))
#        qCurr = getq(np.array(obs2channel))
        
#        # MLP version
#        qCurr = getq(np.reshape(obsDeictic,[-1,deicticShape[0]*deicticShape[1]*deicticShape[2]]))

        # select action
        qCurrNoise = qCurr + np.random.random(np.shape(qCurr))*0.01 # add small amount of noise to break ties randomly
        action = np.argmax(qCurrNoise,1)
        if np.random.rand() < exploration.value(t):
            action = np.random.randint(env.action_space.n)

        # take action
        new_obs, rew, done, _ = env.step(action)
        replay_buffer.add(obs, action, rew, new_obs, float(done))

        # sample from replay buffer and train
        if t > learning_starts and t % train_freq == 0:

            # Sample from replay buffer
            obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(batch_size)
            actions = np.int32(np.reshape(actions,[batch_size,]))
            
#            # Put observations in deictic form
#            obses_t_deic = getDeic(obses_t)
#            obses_tp1_deic = getDeic(obses_tp1)
#            obses_t_deic = getDeic(obses_t)[:,:,:,0:2]
#            obses_tp1_deic = getDeic(obses_tp1)[:,:,:,0:2]
#            
#            # Reshape everything to (1152,) form
#            donesTiled = np.repeat(dones,num_deictic_patches)
#            rewardsTiled = np.repeat(rewards,num_deictic_patches)
#            actionsTiled = np.repeat(actions,num_deictic_patches)
            
            # Get curr, next values: CNN version
#            qNextTarget = getqTarget(obses_tp1_deic)
#            qNext = getq(obses_tp1_deic)
#            qCurr = getq(obses_t_deic)
            qNext = getq(obses_tp1)
            qCurr = getq(obses_t)

#            # Get curr, next values: MLP version
#            qNext = getq(np.reshape(obses_tp1_deic,[-1,deicticShape[0]*deicticShape[1]*deicticShape[2]]))
#            qCurr = getq(np.reshape(obses_t_deic,[-1,deicticShape[0]*deicticShape[1]*deicticShape[2]]))

            # This version pairs a glimpse with the same glimpse on the next time step
            qNextmax = np.max(qNext,1) # standard
#            actionsNext = np.argmax(qNextTarget[:,-1,:],1) # double-q
#            qNextmax = qNext[range(num_deictic_patches*batch_size),-1,actionsNext]
            
#            # This version takes the max over all glimpses
#            qNextTiled = np.reshape(qNext[:,-1,:],[batch_size,num_deictic_patches,num_actions])
#            qNextmax = np.repeat(np.max(np.max(qNextTiled,2),1),num_deictic_patches)

            # Compute Bellman estimate
#            targets = rewardsTiled + (1-donesTiled) * gamma * qNextmax
            targets = rewards + (1-dones) * gamma * qNextmax

#            # Take min over targets in same group
#            obses_t_deic_reshape = np.reshape(obses_t_deic,[-1,deicticShape[0]*deicticShape[1]*deicticShape[2]])
#            unique_deic, uniqueIdx, uniqueCounts= np.unique(obses_t_deic_reshape,return_inverse=True,return_counts=True,axis=0)
#            for i in range(np.shape(uniqueCounts)[0]):
#                targets[uniqueIdx==i] = np.min(targets[uniqueIdx==i])
            
            
#            qCurrTargets = np.copy(qCurr)
#            qCurrTargets[:,np.int32(actions)] = targets
            qCurrTargets = np.zeros(np.shape(qCurr))
            for i in range(num_actions):
                myActions = actions == i
                qCurrTargets[:,i] = myActions * targets + (1 - myActions) * qCurr[:,i]
            
#            # Copy into cascade with pruning.
#            qCurrTargets[range(batch_size*num_deictic_patches),0,actionsTiled] = targets
#            for i in range(num_cascade-1):
#                mask = targets < qCurrTargets[range(batch_size*num_deictic_patches),i,actionsTiled]
#                qCurrTargets[range(batch_size*num_deictic_patches),i+1,actionsTiled] = \
#                    mask*targets + \
#                    (1-mask)*qCurrTargets[range(batch_size*num_deictic_patches),i+1,actionsTiled]
            
            # CNN version
            td_error_out = targetTrain(
                    obses_t,
                    qCurrTargets
                    )
#                    obses_t_deic,
            
#            # MLP version
#            td_error_out, obses_deic_out, targets_out = targetTrain(
#                    np.reshape(obses_t_deic,[-1,deicticShape[0]*deicticShape[1]*deicticShape[2]]),
#                    qCurrTargets
#                    )
                
#        # Update target network periodically.
#        if t > learning_starts and t % target_network_update_freq == 0:
#            update_target()

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


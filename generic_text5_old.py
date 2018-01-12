#
# Adapted from generic_text4.py. 
#
# In this version, I'm planning to introduce an 4-orientation deictic version
#
# Results:
#         -- the best I've seen deictic do on this domain is about 20 (w/ four ghosts)
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

#import envs.multi_ghost_evade1_standalone as envstandalone
import envs.ghost_evade1_standalone as envstandalone
#import envs.ballcatch2_standalone as envstandalone

# **** Make tensorflow functions ****

#def build_getq(make_obsDeic_ph, q_func, num_actions, scope="deepq", reuse=None):
def build_getq(make_obsDeic_ph, q_func, num_actions, num_cascade, scope="deepq", qscope="q_func", reuse=None):

    with tf.variable_scope(scope, reuse=reuse):

        observations_ph = U.ensure_tf_input(make_obsDeic_ph("observation"))
        q_values = q_func(observations_ph.get(), num_actions*num_cascade, scope=qscope)
        q_valuesTiled = tf.reshape(q_values,[-1,num_cascade,num_actions])
        getq = U.function(inputs=[observations_ph], outputs=q_valuesTiled)
        return getq

#def build_getqRotated(make_obsDeic_ph, q_func, num_actions, num_cascade, scope="deepq", reuse=None):
#
#    observations_ph = U.ensure_tf_input(make_obsDeic_ph("obsDeic"))
#    obsDeic = observations_ph.get()
#    rotated1 = tf.map_fn(lambda img: tf.image.rot90(img,k=1), obsDeic)
#    rotated2 = tf.map_fn(lambda img: tf.image.rot90(img,k=1), rotated1)
#    rotated3 = tf.map_fn(lambda img: tf.image.rot90(img,k=1), rotated2)
##    obsDeicRotated = tf.stack([obsin,rotated1,rotated2,rotated3])
#
#    with tf.variable_scope(scope, reuse=reuse):
#        q_values0 = tf.reshape(q_func(obsDeic, num_actions*num_cascade, scope="q_func"),[-1,num_cascade,num_actions])
##        q_values1 = tf.reshape(q_func(obsDeic, num_actions*num_cascade, scope="q_func"),[-1,num_cascade,num_actions])
##        q_values2 = tf.reshape(q_func(obsDeic, num_actions*num_cascade, scope="q_func"),[-1,num_cascade,num_actions])
##        q_values3 = tf.reshape(q_func(obsDeic, num_actions*num_cascade, scope="q_func"),[-1,num_cascade,num_actions])
#        q_values1 = tf.reshape(q_func(rotated1, num_actions*num_cascade, scope="q_func"),[-1,num_cascade,num_actions])
#        q_values2 = tf.reshape(q_func(rotated2, num_actions*num_cascade, scope="q_func"),[-1,num_cascade,num_actions])
#        q_values3 = tf.reshape(q_func(rotated3, num_actions*num_cascade, scope="q_func"),[-1,num_cascade,num_actions])
#
##    q_values = q_values0 + tf.roll(qCurrTemp2[1],1) + np.roll(qCurrTemp2[2],2) + np.roll(qCurrTemp2[3],3)
#
#    # these operations implements the equivalent of the roll above
#    q_values1_tiled = tf.tile(q_values1,[1,1,2])
#    q_values2_tiled = tf.tile(q_values2,[1,1,2])
#    q_values3_tiled = tf.tile(q_values3,[1,1,2])
#    q_values = q_values0 + q_values1_tiled[:,:,3:7] + q_values2_tiled[:,:,2:6] + q_values3_tiled[:,:,1:5]
#    
#    getqRotated = U.function(inputs=[observations_ph], outputs=q_values)
#        
#    return getqRotated

def build_getDeic_Foc(make_obs_ph,deicticShape):
    observations_ph = U.ensure_tf_input(make_obs_ph("observation"))
    patches = tf.extract_image_patches(
            observations_ph.get(),
            ksizes=[1, deicticShape[0], deicticShape[1], 1], 
            strides=[1, 1, 1, 1], 
            rates=[1, 1, 1, 1], 
            padding='VALID')
    patchesShape = tf.shape(patches)
    patchesTiled = tf.reshape(patches,[patchesShape[0]*patchesShape[1]*patchesShape[2],deicticShape[0],deicticShape[1]])
    patchesTiledStacked = tf.stack([tf.equal(patchesTiled,1), tf.equal(patchesTiled,2)],axis=-1)
    getDeic = U.function(inputs=[observations_ph], outputs=patchesTiledStacked)
    return getDeic


# Both focused and coarse representation. Assumes one-channel image
def build_getDeic_FocCoarse(make_obs_ph,deicticShape):
    
    observations_ph = U.ensure_tf_input(make_obs_ph("observation"))

    # create padded image
    obs = observations_ph.get()
    shape = tf.shape(obs)
    obsZeroPadded = tf.image.resize_image_with_crop_or_pad(obs,shape[1]+2*deicticShape[0],shape[2]+2*deicticShape[0])

    # extract large patches from padded image
    patchesLarge = tf.extract_image_patches(
            obsZeroPadded,
            ksizes=[1, 3*deicticShape[0], 3*deicticShape[1], 1], 
            strides=[1, 1, 1, 1], 
            rates=[1, 1, 1, 1], 
            padding='VALID')

    # reshape into focused and large images
    patchesShape = tf.shape(patchesLarge)
    patchesTiledLarge = tf.reshape(patchesLarge,[patchesShape[0]*patchesShape[1]*patchesShape[2],3*deicticShape[0],3*deicticShape[1],1])
    patchesTiledFocused = patchesTiledLarge[:,deicticShape[0]:2*deicticShape[0],deicticShape[1]:2*deicticShape[1],0]

    # get two coarse images: one for agent and one for the ghost (might make this more efficient by doing the resize only once...)
    coarseAgent = tf.image.resize_area(tf.cast(tf.equal(patchesTiledLarge,1.),tf.int32), deicticShape[0:2])[:,:,:,0] > 0
    coarseGhost = tf.image.resize_area(tf.cast(tf.equal(patchesTiledLarge,2.),tf.int32), deicticShape[0:2])[:,:,:,0] > 0

    patchesTiledStacked = tf.stack([tf.equal(patchesTiledFocused,1), tf.equal(patchesTiledFocused,2), coarseAgent, coarseGhost],axis=-1)
    
    getDeic = U.function(inputs=[observations_ph], outputs=patchesTiledStacked)
    return getDeic

def build_targetTrain(make_obsDeic_ph, 
                        make_target_ph,
                        q_func,
                        num_actions,
                        num_cascade,
                        optimizer,
                        scope="deepq", 
                        qscope="q_func",
                        reuse=None):

    with tf.variable_scope(scope, reuse=reuse):
        
        # set up placeholders
        obs_t_input = U.ensure_tf_input(make_obsDeic_ph("obs_t_deic"))
        target_input = U.ensure_tf_input(make_target_ph("target"))
    
        # get variables
        q_func_vars = U.scope_vars(U.absolute_scope_name(qscope))
    
        # q values for all actions
        q_t_raw = q_func(obs_t_input.get(), num_actions*num_cascade, scope=qscope, reuse=True)
    
        targetTiled = tf.reshape(target_input.get(), shape=(-1,num_cascade*num_actions))
        
        # calculate error
        td_error = q_t_raw - tf.stop_gradient(targetTiled)
        errors = U.huber_loss(td_error)
    
        optimize_expr = optimizer.minimize(errors, var_list=q_func_vars)
        
        targetTrain = U.function(
            inputs=[
                obs_t_input,
                target_input
            ],
            outputs=[td_error, obs_t_input.get(), target_input.get()],
            updates=[optimize_expr]
        )
    
        return targetTrain

def build_update_target(scope="deepq",
                      qscope="q_func",
                      qscopeTarget="q_func_target"
                     ):

    with tf.variable_scope(scope, reuse=True):
        
        q_func_vars = U.scope_vars(U.absolute_scope_name(qscope))
        target_q_func_vars = U.scope_vars(U.absolute_scope_name(qscopeTarget))


        # update_target_fn will be called periodically to copy Q network to target Q network
        update_target_expr = []
        for var, var_target in zip(sorted(q_func_vars, key=lambda v: v.name),
                                   sorted(target_q_func_vars, key=lambda v: v.name)):
            update_target_expr.append(var_target.assign(var))
        update_target_expr = tf.group(*update_target_expr)
        update_target = U.function([], [], updates=[update_target_expr])
        
        return update_target
    

def main():

#    env = envstandalone.MultiGhostEvade()
    env = envstandalone.GhostEvade()
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
    deicticShape = (3,3,2)
#    deicticShape = (3,3,4)
#    deicticShape = (4,4,2)
#    deicticShape = (4,4,4)
#    deicticShape = (8,8,2)
    num_deictic_patches = 36
#    num_deictic_patches = 25
#    num_deictic_patches = 1

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
    
    def make_obsDeic_ph(name):

        # CNN version
        return U.BatchInput(deicticShape, name=name)
        
#        # MLP version
#        return U.BatchInput([deicticShape[0]*deicticShape[1]*deicticShape[2]], name=name)

    def make_target_ph(name):
#        return U.BatchInput([num_actions], name=name)
        return U.BatchInput([num_cascade,num_actions], name=name)

    sess = U.make_session(num_cpu)
    sess.__enter__()

    getq = build_getq(
            make_obsDeic_ph=make_obsDeic_ph,
            q_func=q_func,
            num_actions=num_actions,
            num_cascade=num_cascade,
            scope="deepq",
            qscope="q_func"
            )
    
    getqTarget = build_getq(
            make_obsDeic_ph=make_obsDeic_ph,
            q_func=q_func,
            num_actions=num_actions,
            num_cascade=num_cascade,
            scope="deepq",
            qscope="q_func_target"
            )

    update_target = build_update_target(scope="deepq", 
                                        qscope="q_func",
                                        qscopeTarget="q_func_target")
                      
    targetTrain = build_targetTrain(
        make_obsDeic_ph=make_obsDeic_ph,
        make_target_ph=make_target_ph,
        q_func=q_func,
        num_actions=env.action_space.n,
        num_cascade=num_cascade,
        optimizer=tf.train.AdamOptimizer(learning_rate=lr),
        scope="deepq", 
        qscope="q_func"
    )
    
    getDeic = build_getDeic_Foc(make_obs_ph=make_obs_ph,deicticShape=deicticShape)
#    getDeic = build_getDeic_FocCoarse(make_obs_ph=make_obs_ph,deicticShape=deicticShape)
    
#    getqRotated = build_getqRotated(make_obsDeic_ph=make_obsDeic_ph, 
#                                    q_func=q_func, 
#                                    num_actions=num_actions, 
#                                    num_cascade=num_cascade, 
#                                    reuse=True)
    
    # Initialize the parameters and copy them to the target network.
    U.initialize()
    update_target()
    
    replay_buffer = ReplayBuffer(buffer_size)
    obs = env.reset()

    timerStart = time.time()
    for t in range(max_timesteps):

        obsDeictic = getDeic([obs])
        
        qCurr = getq(np.array(obsDeictic))

#        # average Q values from all four orientations
#        qCurrRot0 = getq(np.array(obsDeictic))
#        qCurrRot1 = getq(np.rot90(obsDeictic,k=1,axes=(1,2)))
#        qCurrRot2 = getq(np.rot90(obsDeictic,k=2,axes=(1,2)))
#        qCurrRot3 = getq(np.rot90(obsDeictic,k=3,axes=(1,2)))
#        qCurr = 0.25 * qCurrRot0 + np.roll(qCurrRot1,1,axis=2) + np.roll(qCurrRot2,2,axis=2) + np.roll(qCurrRot3,3,axis=2)

        # select action
        qCurrNoise = qCurr + np.random.random(np.shape(qCurr))*0.01 # add small amount of noise to break ties randomly
        action = np.argmax(np.max(qCurrNoise[:,-1,:],0)) # USE CASCADE
#        action = np.argmax(np.max(qCurrNoise[:,0,:],0)) # DO NOT USE CASCADE
        if np.random.rand() < exploration.value(t):
            action = np.random.randint(env.action_space.n)

        # take action
        new_obs, rew, done, _ = env.step(action)
        replay_buffer.add(obs, action, rew, new_obs, float(done))

        # sample from replay buffer and train
        if t > learning_starts and t % train_freq == 0:

            # Sample from replay buffer
            obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(batch_size)

            # Put observations in deictic form
            obses_t_deic = getDeic(obses_t)
            obses_tp1_deic = getDeic(obses_tp1)
            
            # Reshape everything to (1152,) form
            donesTiled = np.repeat(dones,num_deictic_patches)
            rewardsTiled = np.repeat(rewards,num_deictic_patches)
            actionsTiled = np.repeat(actions,num_deictic_patches)
            
            # WITHOUT ROTATIONS
#            qNextTarget = getqTarget(obses_tp1_deic)
            qNext = getq(obses_tp1_deic)
            qCurr = getq(obses_t_deic)

#            # WITH ROTATIONS
#            qNextRot0 = getq(np.array(obses_tp1_deic))
#            qNextRot1 = getq(np.rot90(obses_tp1_deic,k=1,axes=(1,2)))
#            qNextRot2 = getq(np.rot90(obses_tp1_deic,k=2,axes=(1,2)))
#            qNextRot3 = getq(np.rot90(obses_tp1_deic,k=3,axes=(1,2)))
#            qNext = 0.25 * qNextRot0 + np.roll(qNextRot1,1,axis=2) + np.roll(qNextRot2,2,axis=2) + np.roll(qNextRot3,3,axis=2)
#            
#            obses_t_deicRot1 = np.rot90(obses_t_deic,k=1,axes=(1,2))
#            obses_t_deicRot2 = np.rot90(obses_t_deic,k=2,axes=(1,2))
#            obses_t_deicRot3 = np.rot90(obses_t_deic,k=3,axes=(1,2))
##            obses_t_deicFull = np.r_[obses_t_deic, obses_t_deicRot1, obses_t_deicRot2, obses_t_deicRot3]
#            qCurrRot0 = getq(np.array(obses_t_deic))
#            qCurrRot1 = getq(np.array(obses_t_deicRot1))
#            qCurrRot2 = getq(np.array(obses_t_deicRot2))
#            qCurrRot3 = getq(np.array(obses_t_deicRot3))
#            qCurr = 0.25 * qCurrRot0 + np.roll(qCurrRot1,1,axis=2) + np.roll(qCurrRot2,2,axis=2) + np.roll(qCurrRot3,3,axis=2)
##            qCurrFull = np.r_[qCurrRot0, qCurrRot1, qCurrRot2, qCurrRot3]

            # This version pairs a glimpse with the same glimpse on the next time step
            qNextmax = np.max(qNext[:,-1,:],1) # standard
#            actionsNext = np.argmax(qNextTarget[:,-1,:],1) # double-q
#            qNextmax = qNext[range(num_deictic_patches*batch_size),-1,actionsNext]
            
#            # This version takes the max over all glimpses
#            qNextTiled = np.reshape(qNext[:,-1,:],[batch_size,num_deictic_patches,num_actions])
#            qNextmax = np.repeat(np.max(np.max(qNextTiled,2),1),num_deictic_patches)

            # Compute Bellman estimate
            targets = rewardsTiled + (1-donesTiled) * gamma * qNextmax

#            # Take min over targets in same group
#            obses_t_deic_reshape = np.reshape(obses_t_deic,[-1,deicticShape[0]*deicticShape[1]*deicticShape[2]])
#            unique_deic, uniqueIdx, uniqueCounts= np.unique(obses_t_deic_reshape,return_inverse=True,return_counts=True,axis=0)
#            for i in range(np.shape(uniqueCounts)[0]):
#                targets[uniqueIdx==i] = np.min(targets[uniqueIdx==i])
            
            
            # Copy into cascade with pruning -- WITHOUT ROTATIONS
            qCurrTargets = np.copy(qCurr)
            qCurrTargets[range(batch_size*num_deictic_patches),0,actionsTiled] = targets
            for i in range(num_cascade-1):
                mask = targets < qCurrTargets[range(batch_size*num_deictic_patches),i,actionsTiled]
                qCurrTargets[range(batch_size*num_deictic_patches),i+1,actionsTiled] = \
                    mask*targets + \
                    (1-mask)*qCurrTargets[range(batch_size*num_deictic_patches),i+1,actionsTiled]
#            qCurrTargetsFull = np.tile(qCurrTargets,[4,1,1])
            
#            # Copy into cascade with pruning -- WITH ROTATIONS
#            actionsTiledFull = np.concatenate([actionsTiled, actionsTiled-1, actionsTiled-2, actionsTiled-3])
#            actionsTiledFull = actionsTiledFull + 4 * (actionsTiledFull<0)
#            targetsFull = np.repeat(targets,4)
#            qCurrTargets = np.copy(qCurrFull)
#            qCurrTargets[range(4*batch_size*num_deictic_patches),0,actionsTiledFull] = targetsFull
#            for i in range(num_cascade-1):
#                maskFull = np.repeat(targets < qCurr[range(batch_size*num_deictic_patches),i,actionsTiled],4)
#                qCurrTargets[range(4*batch_size*num_deictic_patches),i+1,actionsTiledFull] = \
#                    maskFull*targetsFull + \
#                    (1-maskFull)*qCurrTargets[range(4*batch_size*num_deictic_patches),i+1,actionsTiledFull]
            
            td_error_out, obses_deic_out, targets_out = targetTrain(obses_t_deic, qCurrTargets)

#            td_error_out, obses_deic_out, targets_out = targetTrain(obses_t_deicFull, qCurrTargets)

            
#            # MLP version
#            td_error_out, obses_deic_out, targets_out = targetTrain(
#                    np.reshape(obses_t_deic,[-1,deicticShape[0]*deicticShape[1]*deicticShape[2]]),
#                    qCurrTargets
#                    )
                
        # Update target network periodically.
        if t > learning_starts and t % target_network_update_freq == 0:
            update_target()

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


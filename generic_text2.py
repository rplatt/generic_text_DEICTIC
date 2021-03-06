#
# Adapted from generic_text1.py. 
#
# In this version, I introduce build_getDeic_FocCoarse. 
#
# Results:
#
#        Ghostevade, 4x4x2: starts to converge around 10k steps and typically reaches >70
#                    4x4x4: starts to converge around 6k steps, peaks at ~40, and then drops to ~20
#
#                    3x3x2: starts to converge around 10k steps and typically peaks >70
#                    3x3x4: starts to converge around 10k steps and typically peaks ~60 or ~70
#
#        Ballcatch,  3x3x2: peaks at ~0.3
#                    3x3x4: seen it peak at ~0, ~0.7, ~0.8 on diff runs
#
#                    4x4x2: peaks at ~0.4 or ~0.5
#                    4x4x4: peaks at ~0.0
#
# Conclusions: the coarse view isn't helping as much as I hoped. Couple of thoughts:
#              1) perhaps I should change network structure
#              2) perhaps the coarse images in the 4x4x4 version is confusing
#
import gym
import numpy as np
import time as time
import tensorflow as tf
import tf_util_rob as U
import models as models
from replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from schedules import LinearSchedule
#import matplotlib.pyplot as plt

#import envs.ghost_evade1_standalone as envstandalone
import envs.ballcatch2_standalone as envstandalone

# **** Make tensorflow functions ****

#def build_getq(make_obsDeic_ph, q_func, num_actions, scope="deepq", reuse=None):
def build_getq(make_obsDeic_ph, q_func, num_actions, num_cascade, scope="deepq", qscope="q_func", reuse=None):

    with tf.variable_scope(scope, reuse=reuse):

        observations_ph = U.ensure_tf_input(make_obsDeic_ph("observation"))
        q_values = q_func(observations_ph.get(), num_actions*num_cascade, scope=qscope)
        q_valuesTiled = tf.reshape(q_values,[-1,num_cascade,num_actions])
        getq = U.function(inputs=[observations_ph], outputs=q_valuesTiled)
        return getq


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

#    env = envstandalone.GhostEvade()
    env = envstandalone.BallCatch()
    
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
#    deicticShape = (3,3,2)
#    deicticShape = (3,3,4)
    deicticShape = (4,4,2)
#    deicticShape = (4,4,4)
#    deicticShape = (8,8,2)
#    num_deictic_patches = 36
    num_deictic_patches = 25
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
#        convs=[(16,4,1)],
        convs=[(16,3,1)],
#        convs=[(16,2,1)],
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
    
    # Initialize the parameters and copy them to the target network.
    U.initialize()
    update_target()
    
    replay_buffer = ReplayBuffer(buffer_size)
    obs = env.reset()

    timerStart = time.time()
    for t in range(max_timesteps):

        obsDeictic = getDeic([obs])
#        obsDeictic = getDeic([obs])[:,:,:,0:2]
        
        # CNN version
        qCurr = getq(np.array(obsDeictic))
        
#        # MLP version
#        qCurr = getq(np.reshape(obsDeictic,[-1,deicticShape[0]*deicticShape[1]*deicticShape[2]]))

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
#            obses_t_deic = getDeic(obses_t)[:,:,:,0:2]
#            obses_tp1_deic = getDeic(obses_tp1)[:,:,:,0:2]
            
            # Reshape everything to (1152,) form
            donesTiled = np.repeat(dones,num_deictic_patches)
            rewardsTiled = np.repeat(rewards,num_deictic_patches)
            actionsTiled = np.repeat(actions,num_deictic_patches)
            
            # Get curr, next values: CNN version
            qNextTarget = getqTarget(obses_tp1_deic)
            qNext = getq(obses_tp1_deic)
            qCurr = getq(obses_t_deic)

#            # Get curr, next values: MLP version
#            qNext = getq(np.reshape(obses_tp1_deic,[-1,deicticShape[0]*deicticShape[1]*deicticShape[2]]))
#            qCurr = getq(np.reshape(obses_t_deic,[-1,deicticShape[0]*deicticShape[1]*deicticShape[2]]))

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
            
            
            qCurrTargets = np.copy(qCurr)
            
            # Copy into cascade with pruning.
            qCurrTargets[range(batch_size*num_deictic_patches),0,actionsTiled] = targets
            for i in range(num_cascade-1):
                mask = targets < qCurrTargets[range(batch_size*num_deictic_patches),i,actionsTiled]
                qCurrTargets[range(batch_size*num_deictic_patches),i+1,actionsTiled] = \
                    mask*targets + \
                    (1-mask)*qCurrTargets[range(batch_size*num_deictic_patches),i+1,actionsTiled]
            
            # CNN version
            td_error_out, obses_deic_out, targets_out = targetTrain(
                    obses_t_deic,
                    qCurrTargets
                    )
            
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


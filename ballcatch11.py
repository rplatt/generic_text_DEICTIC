#
# This version of ballcatch is adapted from ghost_evade7.py
# I'm currently using just a single high-res patch rather than the multi-res
# patch I used earlier. As a result, I only achieve a reward of approx -6 (for a 3x3 patch) rather
# than -1. This is b/c I can only solve approx 1/3 of the problems w/ this small-aperture
# patch. With a 4x4 patch, I get a reward of approx -4.5 (solve approx 1/2 the problems...)
# 
import gym
import numpy as np
import time as time
import tensorflow as tf
import tf_util_rob as U
import models as models
#import build_graph_rob3 as build_graph
from replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from schedules import LinearSchedule
import matplotlib.pyplot as plt

import envs.ballcatch1_standalone as envstandalone

# **** Make tensorflow functions ****

#def build_getq(make_obsDeic_ph, q_func, num_actions, scope="deepq", reuse=None):
def build_getq(make_obsDeic_ph, q_func, num_actions, num_cascade, scope="deepq", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        observations_ph = U.ensure_tf_input(make_obsDeic_ph("observation"))
#        q_values = q_func(observations_ph.get(), num_actions, scope="q_func")
        q_values = q_func(observations_ph.get(), num_actions*num_cascade, scope="q_func")
        q_valuesTiled = tf.reshape(q_values,[-1,num_cascade,num_actions])
#        getq = U.function(inputs=[observations_ph], outputs=q_values)
        getq = U.function(inputs=[observations_ph], outputs=q_valuesTiled)
        return getq

def build_getDeic(make_obs_ph,deicticShape):
    observations_ph = U.ensure_tf_input(make_obs_ph("observation"))
    patches = tf.extract_image_patches(
            observations_ph.get(),
            ksizes=[1, deicticShape[0], deicticShape[1], 1], 
            strides=[1, 1, 1, 1], 
            rates=[1, 1, 1, 1], 
            padding='VALID')
    patchesShape = tf.shape(patches)
    patchesTiled = tf.reshape(patches,[patchesShape[0]*patchesShape[1]*patchesShape[2],deicticShape[0],deicticShape[1]])

    coarse = tf.image.resize_area(
        observations_ph.get(),
        (4,4)) * 4
            
    coarseTiled = tf.transpose(tf.tile([coarse],[patchesShape[1]*patchesShape[2],1,1,1,1]),[1,0,2,3,4])
    
    coarseTiledReshape = tf.reshape(coarseTiled,[patchesShape[0]*patchesShape[1]*patchesShape[2],deicticShape[0],deicticShape[1]])

#    patchesTiledStacked = tf.stack([tf.cast(tf.equal(patchesTiled,1),tf.float32), tf.cast(tf.equal(patchesTiled,2),tf.float32)],axis=-1)
    patchesTiledStacked = tf.stack([tf.cast(tf.equal(patchesTiled,1),tf.float32), tf.cast(tf.equal(patchesTiled,2),tf.float32), coarseTiledReshape],axis=-1)

#    getDeic = U.function(inputs=[observations_ph], outputs=[patchesTiledStacked, coarseTiledReshape, patchesTiled, patchesTiledStacked2])
#    getDeic = U.function(inputs=[observations_ph], outputs=[patchesTiledStacked, patchesTiledStacked2])
    getDeic = U.function(inputs=[observations_ph], outputs=patchesTiledStacked)

    return getDeic

def build_targetTrain(make_obsDeic_ph, 
                        make_target_ph,
                        q_func,
                        num_actions,
                        num_cascade,
                        optimizer,
                        scope="deepq", 
                        reuse=None):

    with tf.variable_scope(scope, reuse=reuse):
        
        # set up placeholders
        obs_t_input = U.ensure_tf_input(make_obsDeic_ph("obs_t_deic"))
        target_input = U.ensure_tf_input(make_target_ph("target"))

        # get variables
        q_func_vars = U.scope_vars(U.absolute_scope_name("q_func"))

        # q values for all actions
#        q_t_raw = q_func(obs_t_input.get(), num_actions, scope="q_func", reuse=True)
        q_t_raw = q_func(obs_t_input.get(), num_actions*num_cascade, scope="q_func", reuse=True)

        targetTiled = tf.reshape(target_input.get(), shape=(-1,num_cascade*num_actions))
        
        # calculate error
#        td_error = q_t - tf.stop_gradient(target_input.get())
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


def main():

    env = envstandalone.BallCatch()
    
    max_timesteps=40000
    learning_starts=1000
    buffer_size=50000
#    buffer_size=1
    exploration_fraction=0.2
    exploration_final_eps=0.02
    print_freq=10
    gamma=.98
    target_network_update_freq=500
    learning_alpha = 0.2
    
    batch_size=32
    train_freq=1

    obsShape = (8,8,1)
#    deicticShape = (3,3,1)
#    deicticShape = (3,3,2)
#    deicticShape = (4,4,1)
#    deicticShape = (4,4,2)
    deicticShape = (4,4,3)
#    deicticShape = (3,3,4)
    num_deictic_patches=25

#    num_actions = 4
    num_actions = 3
    episode_rewards = [0.0]
    num_cpu=16
    num_cascade = 5
    
    # Create the schedule for exploration starting from 1.
    exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * max_timesteps),
                                 initial_p=1.0,
                                 final_p=exploration_final_eps)

    # Extract deictic patches for an input obs. Each deictic patch has a low level
    # and a foveated view.
    # input: n x n x 1
    # output: dn x dn x 4
    def getDeicticObs(obs):
        windowLen = deicticShape[0]
        obsShape = np.shape(obs)
        obsPadded = np.zeros((obsShape[0]+2*windowLen,obsShape[1]+2*windowLen))
        obsPadded[windowLen:windowLen+obsShape[0],windowLen:windowLen+obsShape[1]] = obs[:,:,0]
        deicticObsThis = np.zeros((windowLen,windowLen,4)) # channel1: zoomin window; channel2: agent in zoomout window; channel3: ball in zoomout window
        deicticObs = []
        for i in range(obsShape[0] - windowLen + 1):
            for j in range(obsShape[1] - windowLen + 1):
                deicticObsThis[:,:,0] = obs[i:i+windowLen,j:j+windowLen,0] == 1 # agent zoomin
                deicticObsThis[:,:,1] = obs[i:i+windowLen,j:j+windowLen,0] == 2 # ball zoomin
                patch = obsPadded[i:i+3*windowLen,j:j+3*windowLen]
                for k in range(1,3):
# THE VERSION BELOW USES A FIXED VIEW
#                    deicticObsThis[:,:,k+1] = [[(k in obs[0:3,0:3,0]), (k in obs[0:3,3:5]), (k in obs[0:3,5:8,0])], 
#                                 [(k in obs[3:5,0:3,0]), (k in obs[3:5,3:5,0]), (k in obs[3:5,5:8,0])],
#                                 [(k in obs[5:8,0:3,0]), (k in obs[5:8,3:5,0]), (k in obs[5:8,5:8,0])]]
# THE VERSION BELOW USES A WIDE VIEW W/ 2 UNITS IN EACH CELL
#                    deicticObsThis[:,:,k+1] = [[(k in patch[1:3,1:3]), (k in patch[1:3,3:5]), (k in patch[1:3,5:7])], 
#                                 [(k in patch[3:5,1:3]), (k in patch[3:5,3:5]), (k in patch[3:5,5:7])], 
#                                 [(k in patch[5:7,1:3]), (k in patch[5:7,3:5]), (k in patch[5:7,5:7])]]
# THE VERSION BELOW USES A WIDE VIEW W/ 3 UNITS IN EACH CELL
                    deicticObsThis[:,:,k+1] = [[(k in patch[0:3,0:3]), (k in patch[0:3,3:6]), (k in patch[0:3,6:9])], 
                                 [(k in patch[3:6,0:3]), (k in patch[3:6,3:6]), (k in patch[3:6,6:9])], 
                                 [(k in patch[6:9,0:3]), (k in patch[6:9,3:6]), (k in patch[6:9,6:9])]]
                deicticObs.append(deicticObsThis.copy()) # CAREFUL WITH APPENDING REFERENCES VS APPENDING COPIES!!! THIS WAS A BUG BEFORE I CORRECTED IT...

        return np.array(deicticObs)
    

    # Same as getDeicticObs, but it operates on a batch rather than a single obs
    # input: obs -> batches x glances x 3 x 3 x 4
    def getDeicticObsBatch(obs):
        obsShape = np.shape(obs)
        deicticObsBatch = []
        for batch in range(obsShape[0]):
            deicticObsBatch.append(getDeicticObs(obs[batch]))
        shape = np.shape(deicticObsBatch)
        return(np.reshape(np.array(deicticObsBatch),[shape[0]*shape[1],shape[2],shape[3],shape[4]]))
        

    # CNN version
    # conv model parameters: (num_outputs, kernel_size, stride)
#    model = models.cnn_to_mlp(
#        convs=[(16,4,1)],
#        hiddens=[16],
#        dueling=True
#    )
    
    # MLP version
    model = models.mlp([16, 32])

    q_func=model
    lr=0.001
    
    def make_obs_ph(name):
        return U.BatchInput(obsShape, name=name)
    
    def make_obsDeic_ph(name):

        # CNN version
#        return U.BatchInput(deicticShape, name=name)
        
        # MLP version
        return U.BatchInput([deicticShape[0]*deicticShape[1]*deicticShape[2]], name=name)

    def make_target_ph(name):
#        return U.BatchInput([num_actions], name=name)
        return U.BatchInput([num_cascade,num_actions], name=name)

    sess = U.make_session(num_cpu)
    sess.__enter__()

    getq = build_getq(
            make_obsDeic_ph=make_obsDeic_ph,
            q_func=q_func,
            num_actions=num_actions,
            num_cascade=num_cascade)
    
    targetTrain = build_targetTrain(
        make_obsDeic_ph=make_obsDeic_ph,
        make_target_ph=make_target_ph,
        q_func=q_func,
        num_actions=env.action_space.n,
        num_cascade=num_cascade,
        optimizer=tf.train.AdamOptimizer(learning_rate=lr)
    )
    
    getDeic = build_getDeic(make_obs_ph=make_obs_ph,deicticShape=deicticShape)
    
    # Initialize the parameters and copy them to the target network.
    U.initialize()

    replay_buffer = ReplayBuffer(buffer_size)
    obs = env.reset()

    timerStart = time.time()
    for t in range(max_timesteps):

#        obsDeictic = getDeicticObs(obs)
        obsDeictic = getDeic([obs])
#        obsDeictic, patchesTiledStacked2 = getDeic([obs])
        
#        # CNN version
#        qCurr = getq(np.array(obsDeictic))
        
        # MLP version
        qCurr = getq(np.reshape(obsDeictic,[-1,deicticShape[0]*deicticShape[1]*deicticShape[2]]))

        # select action
        qCurrNoise = qCurr + np.random.random(np.shape(qCurr))*0.01 # add small amount of noise to break ties randomly
        action = np.argmax(np.max(qCurrNoise[:,-1,:],0))
        selPatch = np.argmax(np.max(qCurrNoise[:,-1,:],1))
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
#            obses_t_deic = getDeicticObsBatch(obses_t)
#            obses_tp1_deic = getDeicticObsBatch(obses_tp1)
            
            # Reshape everything to (1152,) form
            donesTiled = np.repeat(dones,num_deictic_patches)
            rewardsTiled = np.repeat(rewards,num_deictic_patches)
            actionsTiled = np.repeat(actions,num_deictic_patches)
            
#            # Get curr, next values: CNN version
#            qNext = getq(obses_tp1_deic)
#            qCurr = getq(obses_t_deic)

            # Get curr, next values: MLP version
            qNext = getq(np.reshape(obses_tp1_deic,[-1,deicticShape[0]*deicticShape[1]*deicticShape[2]]))
            qCurr = getq(np.reshape(obses_t_deic,[-1,deicticShape[0]*deicticShape[1]*deicticShape[2]]))

            # This version pairs a glimpse with the same glimpse on the next time step
            qNextmax = np.max(qNext[:,-1,:],1)
            
#            # This version takes the max over all glimpses
#            qNextTiled = np.reshape(qNext[:,-1,:],[batch_size,num_deictic_patches,num_actions])
#            qNextmax = np.repeat(np.max(np.max(qNextTiled,2),1),num_deictic_patches)

            # Compute Bellman estimate
            targets = rewardsTiled + (1-donesTiled) * gamma * qNextmax

#            targetsTiled = np.tile(np.reshape(targets,[-1,1]),[1,num_cascade])
            
            qCurrTargets = np.copy(qCurr)
            
#            # Copy into cascade without pruning
#            for i in range(num_cascade):
#                qCurrTargets[range(batch_size*num_deictic_patches),i,actionsTiled] = targets
            
            # Copy into cascade with pruning.
            qCurrTargets[range(batch_size*num_deictic_patches),0,actionsTiled] = targets
            for i in range(num_cascade-1):
                mask = targets < qCurrTargets[range(batch_size*num_deictic_patches),i,actionsTiled]
                qCurrTargets[range(batch_size*num_deictic_patches),i+1,actionsTiled] = \
                    mask*targets + \
                    (1-mask)*qCurrTargets[range(batch_size*num_deictic_patches),i+1,actionsTiled]
            
#            # CNN version
#            td_error_out, obses_deic_out, targets_out = targetTrain(
#                    obses_t_deic,
#                    qCurrTargets
#                    )
            
            # MLP version
            td_error_out, obses_deic_out, targets_out = targetTrain(
                    np.reshape(obses_t_deic,[-1,deicticShape[0]*deicticShape[1]*deicticShape[2]]),
                    qCurrTargets
                    )
                
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


#
# This should be the same as ghost_evade5.py except that I have written 
# the deictic conversion function in tensorflow instead of numpy. It's at
# least twice as fast...
# 
# All the tensorflow code is in this file -- none is in build_graph_rob3...
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

# I had problems w/ the gym environment, so here I made my own standalone class
#import envs.frozen_lake
#import envs.ballcatch1_standalone as envstandalone
import envs.testrob3_standalone as envstandalone

# **** Make tensorflow functions ****

def build_getq(make_obsDeic_ph, q_func, num_actions, scope="deepq", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        observations_ph = U.ensure_tf_input(make_obsDeic_ph("observation"))
        q_values = q_func(observations_ph.get(), num_actions, scope="q_func")
        getq = U.function(inputs=[observations_ph], outputs=q_values)
        return getq

def build_getDeic(make_obs_ph,deicticShape):
    observations_ph = U.ensure_tf_input(make_obs_ph("observation"))
    patches = tf.extract_image_patches(
            observations_ph.get(),
            ksizes=[1, 3, 3, 1], 
            strides=[1, 1, 1, 1], 
            rates=[1, 1, 1, 1], 
            padding='VALID')
    patchesShape = tf.shape(patches)
    patchesTiled = tf.reshape(patches,[patchesShape[0]*patchesShape[1]*patchesShape[2],deicticShape[0],deicticShape[1]])
    patchesTiledStacked = tf.stack([tf.equal(patchesTiled,10), tf.equal(patchesTiled,20)],axis=-1)
    getDeic = U.function(inputs=[observations_ph], outputs=patchesTiledStacked)
    return getDeic

def build_targetTrain(make_obsDeic_ph, 
                        make_target_ph,
                        q_func,
                        num_actions,
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
        q_t_raw = q_func(obs_t_input.get(), num_actions, scope="q_func", reuse=True)
        q_t = tf.reshape(q_t_raw, shape=(-1,num_actions))
        
        # calculate error
        td_error = q_t - tf.stop_gradient(target_input.get())
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

    env = envstandalone.TestRob3Env()
    
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
    deicticShape = (3,3,2)
    num_deictic_patches=36

    num_actions = 4
    episode_rewards = [0.0]
    num_cpu=16

    # Create the schedule for exploration starting from 1.
    exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * max_timesteps),
                                 initial_p=1.0,
                                 final_p=exploration_final_eps)

    # same as getDeictic except this one just calculates for the observation
    # input: n x n x channels
    # output: dn x dn x channels
    def getDeicticObs(obs):
        windowLen = deicticShape[0]
        deicticObs = []
        for i in range(np.shape(obs)[0] - windowLen + 1):
            for j in range(np.shape(obs)[1] - windowLen + 1):
                
#                # one-channel output
#                deicticObsThis = obs[i:i+windowLen,j:j+windowLen,:]
                
                # two channel output
                deicticObsThis = np.zeros(deicticShape)
                deicticObsThis[:,:,0] = obs[i:i+windowLen,j:j+windowLen,0]==10
                deicticObsThis[:,:,1] = obs[i:i+windowLen,j:j+windowLen,0]==20
                
                deicticObs.append(deicticObsThis)
                
                
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
    model = models.cnn_to_mlp(
        convs=[(16,3,1)],
        hiddens=[16],
        dueling=True
    )
    
#    # MLP version
#    model = models.mlp([16, 32])

    q_func=model
    lr=0.001
    
    def make_obs_ph(name):
        return U.BatchInput(obsShape, name=name)
    
    def make_obsDeic_ph(name):

        # CNN version
        return U.BatchInput(deicticShape, name=name)
        
#        # MLP version
#        return U.BatchInput([9], name=name)

    def make_target_ph(name):
        return U.BatchInput([num_actions], name=name)

    sess = U.make_session(num_cpu)
    sess.__enter__()

    getq = build_getq(
            make_obsDeic_ph=make_obsDeic_ph,
            q_func=q_func,
            num_actions=num_actions)
    
    targetTrain = build_targetTrain(
        make_obsDeic_ph=make_obsDeic_ph,
        make_target_ph=make_target_ph,
        q_func=q_func,
        num_actions=env.action_space.n,
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
        
        # CNN version
        qCurr = getq(np.array(obsDeictic))
        
#        # MLP version
#        qCurr = getq(np.reshape(obsDeictic,[-1,9]))

        # select action
        qCurrNoise = qCurr + np.random.random(np.shape(qCurr))*0.01 # add small amount of noise to break ties randomly
        action = np.argmax(np.max(qCurrNoise,0))
        selPatch = np.argmax(np.max(qCurrNoise,1))
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
#            obses_t_deic = getDeicticObsBatch(obses_t)
#            obses_tp1_deic = getDeicticObsBatch(obses_tp1)            
            obses_t_deic = getDeic(obses_t)
            obses_tp1_deic = getDeic(obses_tp1)
            
            # Reshape everything to (1152,) form
            donesTiled = np.repeat(dones,num_deictic_patches)
            rewardsTiled = np.repeat(rewards,num_deictic_patches)
            actionsTiled = np.repeat(actions,num_deictic_patches)
            
            # Get curr, next values: CNN version
            qNext = getq(obses_tp1_deic)
            qCurr = getq(obses_t_deic)

#            # Get curr, next values: MLP version
#            qNext = getq(np.reshape(obses_tp1_deic,[-1,9]))
#            qCurr = getq(np.reshape(obses_t_deic,[-1,9]))

            # Get "raw" targets (no masking for cascade levels)
            qNextmax = np.max(qNext,1)
            targets = rewardsTiled + (1-donesTiled) * gamma * qNextmax

            # Update values: neural network version
            qCurrTargets = np.copy(qCurr)
            qCurrTargets[range(batch_size*num_deictic_patches),actionsTiled] = targets
            
            # CNN version
            td_error_out, obses_deic_out, targets_out = targetTrain(
                    obses_t_deic,
                    qCurrTargets
                    )

#            # MLP version
#            td_error_out, obses_deic_out, targets_out = targetTrain(
#                    np.reshape(obses_t_deic,[-1,9]),
#                    qCurrTargets
#                    )
                
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


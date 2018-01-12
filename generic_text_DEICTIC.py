#
# This demonstrates a variety of "sequential deictic" algorithms that I have explored recently
# for a small suite of text-based domains. This version of the code does most of
# the key calculations in numpy instead of tensorflow -- this makes it a bit easier to
# debug, but is probably a bit slower than it could be...
#
# Adapted from generic_text6.py. Runs in python 2.7
#
#
import gym
import numpy as np
import time as time
import tensorflow as tf
import tf_util_rob as U
import models as models
from replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from schedules import LinearSchedule

# **** Make tensorflow functions ****

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
                        grad_norm_clipping=None,
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
                target_input
            ],
            outputs=[td_error, obs_t_input.get(), target_input.get()],
            updates=[optimize_expr]
        )
    
        return targetTrain

    
#import envs.multi_ghost_evade1_standalone as envstandalone
import envs.ghost_evade1_standalone as envstandalone
#import envs.ballcatch2_standalone as envstandalone


def main():

    
    # ******* Deictic parameters ********
    
    # deicticShape is the shape of the patch that is used. For example, a 3,3,2 patch
    # is a 2-channel 3x3 patch. num_deictic_patches must be set to the number of deicticShape 
    # patches in an entire image.
    # For example, there are 36 3x3 patches that are contained in an 8x8 observation space
    # (assuming no zero padding). You must set this number to correspond to deicticShape.    
    
#    deicticShape = (3,3,2)
#    deicticShape = (3,3,4)
    deicticShape = (4,4,2)
#    deicticShape = (4,4,4)

#    num_deictic_patches = 36
    num_deictic_patches = 25
    
    # Desired network type. So far, I've done better w/ CNN
    WHICH_Q = "CNN"
#    WHICH_Q = "MLP"
    
    # Method used to evaluate value of next state. So far, I've found that PAIRED_NEXT works
    # much better than MAX_NEXT. MAX_NEXT only works if you also set MIN_OVER_BATCH to True.
    # OW, it doesn't converge.
    # PAIRED_NEXT -> use value of corresponding patch on the next step
    # MAX_NEXT -> use max value over all next-step patches
    NEXT_PATCH = "PAIRED_NEXT"
#    NEXT_PATCH = "MAX_NEXT"

    # If MIN_OVER_BATCH is true, then we find the min value over all targets that have
    # the same corresponding patch. In principle, this should always help. The larger
    # the batch size, the more it should help. However, in practice, I find that
    # it seems to cap the maximum achievable performance. On the other hand, it can
    # help convergence when using NEXT_PATCH = "MAX_NEXT".
#    MIN_OVER_BATCH = True
    MIN_OVER_BATCH = False

    # If MIN_OR_AVG_Q is "MIN", then we use the minimum Q value as calculated via the cascade.
    # OW (if "AVG"), we use the standard expected value Q value. "MIN" should work. "AVG" is
    # equivalent to the standard DQN backup applied to the patches.
    # best here.
    MIN_OR_AVG_Q = "MIN"
#    MIN_OR_AVG_Q = "AVG"
    
    # If true, ROTATION_AUGMENTATION augments the agent's experience with
    # rotated versions of the patches. I typically turn this off.
#    ROTATION_AUGMENTATION = True
    ROTATION_AUGMENTATION = False
    

    # ******* Load the environment ********
    
    env = envstandalone.StandaloneEnv()
    obsShape = env.observation_space.shape
    num_actions = env.action_space.n

    # ******* Standard DQN parameters ********
    
    max_timesteps=40000
    learning_starts=1000
    buffer_size=50000
    exploration_fraction=0.4
    exploration_final_eps=0.02
    print_freq=10
    gamma=.98
    target_network_update_freq=1
    lr=0.001
    batch_size=32
    train_freq=1
    num_cascade = 5 # number of Q-functions in the cascade used to estimate a minimum value for each s,a pair
    num_cpu=16
    replay_buffer = ReplayBuffer(buffer_size)
    exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * max_timesteps),
                                 initial_p=1.0,
                                 final_p=exploration_final_eps)

    if MIN_OR_AVG_Q == "MIN":
        minoravg = -1
    elif MIN_OR_AVG_Q == "AVG":
        minoravg = 0
    else:
        print("error")


    # ******* Create neural network model ********

    if WHICH_Q == "CNN":
        # conv model parameters: (num_outputs, kernel_size, stride)
        model = models.cnn_to_mlp(
            convs=[(32,3,1)],
            hiddens=[32],
            dueling=True
        )
        networkShapeOfObservation = [-1,deicticShape[0],deicticShape[1],deicticShape[2]]
    elif WHICH_Q == "MLP":
        # MLP version
#        model = models.mlp([8, 16])
        model = models.mlp([16, 32])
#        model = models.mlp([32])
#        model = models.mlp([])
        networkShapeOfObservation = [-1,deicticShape[0]*deicticShape[1]*deicticShape[2]]
    else:
        print("WHICH_Q error: must select valid q-function")    
    q_func=model


    # ******* Build tensorflow functions ********
    
    def make_obs_ph(name):
        return U.BatchInput(obsShape, name=name)
    
    def make_obsDeic_ph(name):

        if WHICH_Q == "CNN":
            return U.BatchInput(deicticShape, name=name)
        elif WHICH_Q == "MLP":
            return U.BatchInput([deicticShape[0]*deicticShape[1]*deicticShape[2]], name=name)
        else:
            print("WHICH_Q error: must select valid q-function")

    def make_target_ph(name):
#        return U.BatchInput([num_actions], name=name)
        return U.BatchInput([num_cascade,num_actions], name=name)

    getq = build_getq(
            make_obsDeic_ph=make_obsDeic_ph,
            q_func=q_func,
            num_actions=num_actions,
            num_cascade=num_cascade,
            scope="deepq",
            qscope="q_func"
            )

    targetTrain = build_targetTrain(
        make_obsDeic_ph=make_obsDeic_ph,
        make_target_ph=make_target_ph,
        q_func=q_func,
        num_actions=env.action_space.n,
        num_cascade=num_cascade,
        optimizer=tf.train.AdamOptimizer(learning_rate=lr),
#        optimizer=tf.train.GradientDescentOptimizer(learning_rate=lr),
        scope="deepq", 
        qscope="q_func",
        grad_norm_clipping=1.
    )
    
    getDeic = build_getDeic_Foc(make_obs_ph=make_obs_ph,deicticShape=deicticShape)
#    getDeic = build_getDeic_FocCoarse(make_obs_ph=make_obs_ph,deicticShape=deicticShape)
    
    

    sess = U.make_session(num_cpu)
    sess.__enter__()

    obs = env.reset()
    U.initialize()
    episode_rewards = [0.0]
    timerStart = time.time()
    for t in range(max_timesteps):

        # get q-values for current deictic patches
        obsDeictic = getDeic([obs])
        qCurr = getq(np.reshape(obsDeictic,networkShapeOfObservation))

        # select action
        qCurrNoise = qCurr + np.random.random(np.shape(qCurr))*0.01 # add small amount of noise to break ties randomly
        action = np.argmax(np.max(qCurrNoise[:,minoravg,:],0)) # USE CASCADE
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
            
            # Reshape such that patches and batches are interleaved in the same column
            donesTiled = np.repeat(dones,num_deictic_patches)
            rewardsTiled = np.repeat(rewards,num_deictic_patches)
            actionsTiled = np.repeat(actions,num_deictic_patches)
            
#            # Get curr, next values: NO ROTATION-AUGMENTATION 
            qNext = getq(np.reshape(obses_tp1_deic,networkShapeOfObservation))
            qCurr = getq(np.reshape(obses_t_deic,networkShapeOfObservation))

#            # ROTATION-AUGMENTATION: AUGMENT EXPERIENCES WITH FOUR ROTATIONS
            if ROTATION_AUGMENTATION:
                obses_t_deicRot1 = np.rot90(obses_t_deic,k=3,axes=(1,2))
                obses_t_deicRot2 = np.rot90(obses_t_deic,k=2,axes=(1,2))
                obses_t_deicRot3 = np.rot90(obses_t_deic,k=1,axes=(1,2))
                obses_t_deic = np.r_[obses_t_deic, obses_t_deicRot1, obses_t_deicRot2, obses_t_deicRot3]
                obses_tp1_deicRot1 = np.rot90(obses_tp1_deic,k=3,axes=(1,2))
                obses_tp1_deicRot2 = np.rot90(obses_tp1_deic,k=2,axes=(1,2))
                obses_tp1_deicRot3 = np.rot90(obses_tp1_deic,k=1,axes=(1,2))
                obses_tp1_deic = np.r_[obses_tp1_deic, obses_tp1_deicRot1, obses_tp1_deicRot2, obses_tp1_deicRot3]
                qCurr = getq(np.array(obses_t_deic))
                qNext = getq(np.array(obses_tp1_deic))
                actionsTiled = np.r_[actionsTiled, actionsTiled+1, actionsTiled+2, actionsTiled+3]
                actionsTiled = actionsTiled - 4 * (actionsTiled>3)
                rewardsTiled = np.r_[rewardsTiled,rewardsTiled,rewardsTiled,rewardsTiled]
                donesTiled = np.r_[donesTiled,donesTiled,donesTiled,donesTiled]            
            
            # Get value of next state
            if NEXT_PATCH=="PAIRED_NEXT":
                qNextmax = np.max(qNext[:,minoravg,:],1) # standard
            elif NEXT_PATCH=="MAX_NEXT":
                qNextTiled = np.reshape(qNext[:,minoravg,:],[-1,num_deictic_patches,num_actions])
                qNextmax = np.repeat(np.max(np.max(qNextTiled,2),1), num_deictic_patches)
            else:
                print("error")
                
            # Compute Bellman estimate
            targets = rewardsTiled + (1-donesTiled) * gamma * qNextmax

            # Take min over targets in same group
            if MIN_OVER_BATCH:
                obses_t_deic_reshape = np.reshape(obses_t_deic,[-1,deicticShape[0]*deicticShape[1]*deicticShape[2]])
                unique_deic, uniqueIdx, uniqueCounts= np.unique(obses_t_deic_reshape,return_inverse=True,return_counts=True,axis=0)
                for i in range(np.shape(uniqueCounts)[0]):
                    targets[uniqueIdx==i] = np.min(targets[uniqueIdx==i])
            
            # Copy into cascade with pruning.
            qCurrTargets = np.copy(qCurr)
            expLen = np.shape(qCurr)[0]
            qCurrTargets[range(expLen),0,actionsTiled] = targets
            for i in range(num_cascade-1):
                mask = targets < qCurrTargets[range(expLen),i,actionsTiled]
                qCurrTargets[range(expLen),i+1,actionsTiled] = \
                    mask*targets + \
                    (1-mask)*qCurrTargets[range(expLen),i+1,actionsTiled]
            
            td_error_out, obses_deic_out, targets_out = targetTrain(
                    np.reshape(obses_t_deic,networkShapeOfObservation),
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


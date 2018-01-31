#
# DQN Q-learning w/ a replay buffer. Has an option to use tabular value function instead.
#
# You can play with the parameters listed just after def main() (commonly used options):
#
# The following parameter settings are equivalent to tabular learning w/ no replay buffer:
#   buffer_size=1, batch_size=1, valueFunctionType = "TABULAR", prioritized_replay=False
#
# You can do DQN w/o a replay buffer using:
#   buffer_size=1, batch_size=1, valueFunctionType = "DQN", prioritized_replay=False
#
# Or, for DQN w/ replay buffers, use:
#   buffer_size=100, batch_size=10, valueFunctionType = "DQN", prioritized_replay=False
#   
import gym
import numpy as np
import time as time
import tensorflow as tf
import tf_util_rob as U
import models as models
from replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from schedules import LinearSchedule

import envs.frozen_lake

# **** Make tensorflow functions ****

def build_getq(make_obs_ph, q_func, num_actions, scope="deepq", qscope="q_func", reuse=None):

    with tf.variable_scope(scope, reuse=reuse):

        obs_ph = U.ensure_tf_input(make_obs_ph("obs"))
        q_values = q_func(obs_ph.get(), num_actions, scope=qscope)
        getq = U.function(inputs=[obs_ph], outputs=q_values)
        return getq


def build_targetTrain(make_obs_ph,
                        make_target_ph,
                        make_weight_ph,
                        q_func,
                        num_actions,
                        optimizer,
                        scope="deepq", 
                        qscope="q_func",
                        grad_norm_clipping=None,
                        reuse=None):

    with tf.variable_scope(scope, reuse=reuse):
        
        # set up placeholders
        obs_t_input = U.ensure_tf_input(make_obs_ph("obs"))
        target_input = U.ensure_tf_input(make_target_ph("target"))
        importance_weights_ph = U.ensure_tf_input(make_weight_ph("weights"))
    
        # get variables
        q_func_vars = U.scope_vars(U.absolute_scope_name(qscope))
    
        # q values for all actions
        q_t_raw = q_func(obs_t_input.get(), num_actions, scope=qscope, reuse=True)
        targetTiled = tf.reshape(target_input.get(), shape=(-1,num_actions))
        
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


def main():

    # ********* Commonly used options. *************
    buffer_size=1
    batch_size=1
#    valueFunctionType = "TABULAR"
    valueFunctionType = "DQN"
#    prioritized_replay=True
    prioritized_replay=False
    # ********* *********************** *************
    
    np.set_printoptions(formatter={'float_kind':lambda x: "%.2f" % x})
    
    env = gym.make("FrozenLake-v0")
#    env = gym.make("FrozenLake8x8-v0")
    obs_space = np.int32([np.sqrt(env.observation_space.n),np.sqrt(env.observation_space.n)])

    # Dictionary-based value function
    q_func_tabular = {}
    defaultQValue = np.ones(env.action_space.n)
    
    # Given an integer, return the corresponding boolean array 
    def getBoolBits(state):
        return np.unpackbits(np.uint8(state),axis=1)==1
    
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
        return np.array([q_func_tabular[x] if x in q_func_tabular else defaultQValue for x in keys])
    
#    def trainTabular(vectorKey,qCurrTargets,weights):
    def trainTabular(vectorKey,qCurrTargets):
        keys = getTabularKeys(vectorKey)
        alpha=0.1
        for i in range(len(keys)):
            if keys[i] in q_func_tabular:
                q_func_tabular[keys[i]] = (1-alpha)*q_func_tabular[keys[i]] + alpha*qCurrTargets[i]
#                q_func_tabular[keys[i]] = q_func_tabular[keys[i]] + alpha*weights[i,:]*(qCurrTargets[i] - q_func_tabular[keys[i]]) # (1-alpha)*q_func[keys[i]] + alpha*qCurrTargets[i]
            else:
                q_func_tabular[keys[i]] = qCurrTargets[i]



    max_timesteps=100000
    exploration_fraction=0.3
    exploration_final_eps=0.02
    print_freq=1
    gamma=.98
    num_cpu = 16

    # Used by buffering and DQN
    learning_starts=10
    target_network_update_freq=1
    train_freq=1
    print_freq=1
    lr=0.0003

    
    episode_rewards = [0.0]
    
    # Create the schedule for exploration starting from 1.
    exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * max_timesteps),
                                 initial_p=1.0,
                                 final_p=exploration_final_eps)

    # Set up replay buffer
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


    q_func = models.cnn_to_mlp(
        convs=[(16,4,1)],
        hiddens=[32],
        dueling=True
    )

    def make_obs_ph(name):
        return U.BatchInput(obs_space, name=name)

    def make_target_ph(name):
        return U.BatchInput([env.action_space.n], name=name)

    def make_weight_ph(name):
        return U.BatchInput([env.action_space.n], name=name)
    
    if valueFunctionType == 'DQN':
        getq = build_getq(
                make_obs_ph=make_obs_ph,
                q_func=q_func,
                num_actions=env.action_space.n,
                scope="deepq"
                )
    
        targetTrain = build_targetTrain(
            make_obs_ph=make_obs_ph,
            make_target_ph=make_target_ph,
            make_weight_ph=make_weight_ph,
            q_func=q_func,
            num_actions=env.action_space.n,
            optimizer=tf.train.AdamOptimizer(learning_rate=lr),
            scope="deepq", 
            qscope="q_func",
            grad_norm_clipping=1.
        )


        
        
    sess = U.make_session(num_cpu)
    sess.__enter__()

    state = env.reset()

    episode_rewards = [0.0]
    timerStart = time.time()
    U.initialize()
    for t in range(max_timesteps):
        
        if valueFunctionType == "TABULAR":
            qCurr = getTabular(getBoolBits([[state]]))
        else:
            qCurr = getq(np.reshape(np.eye(16)[state,:],[1,obs_space[0],obs_space[1]]))

        qCurrNoise = qCurr + np.random.random(np.shape(qCurr))*0.01 # add small amount of noise to break ties randomly

        # select action at random
        action = np.argmax(qCurrNoise)
        if np.random.rand() < exploration.value(t):
            action = np.random.randint(env.action_space.n)
        
        # take action
        nextState, rew, done, _ = env.step(action)

        replay_buffer.add(state, action, rew, nextState, float(done))

        if t > learning_starts and t % train_freq == 0:

            # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
            if prioritized_replay:
                beta=beta_schedule.value(t)
                states_t, actions, rewards, states_tp1, dones, weights, batch_idxes = replay_buffer.sample(batch_size, beta)
            else:
                states_t, actions, rewards, states_tp1, dones = replay_buffer.sample(batch_size)
                weights, batch_idxes = np.ones_like(rewards), None

            if valueFunctionType == "TABULAR":
                qNext = getTabular(getBoolBits(np.reshape(states_tp1,[batch_size,1])))
            else:
                qNext = getq(np.reshape(np.eye(16)[states_tp1,:],[batch_size,obs_space[0],obs_space[1]]))

            qNextmax = np.max(qNext,axis=1)
            targets = rewards + (1-dones) * gamma * qNextmax
            
            if valueFunctionType == "TABULAR":
                qCurrTarget = getTabular(getBoolBits(np.reshape(states_t,[batch_size,1])))
            else:
                qCurrTarget = getq(np.reshape(np.eye(16)[states_t,:],[batch_size,obs_space[0],obs_space[1]]))

            td_error = qCurrTarget[range(batch_size),actions] - targets
            qCurrTarget[range(batch_size),actions] = targets
            
            if valueFunctionType == "TABULAR":
                trainTabular(getBoolBits(np.reshape(states_t,[batch_size,1])), qCurrTarget)
            else:
                targetTrain(np.reshape(np.eye(16)[states_t,:],[batch_size,obs_space[0],obs_space[1]]), qCurrTarget, np.tile(np.reshape(weights,[batch_size,1]),env.action_space.n))

            if prioritized_replay:
                new_priorities = np.abs(td_error) + prioritized_replay_eps
                replay_buffer.update_priorities(batch_idxes, new_priorities)
            


#        qNext = getTabular(getBoolBits(nextState))
#
#        # Calculate TD target
#        qNextmax = np.max(qNext)
#        target = rew + (1-done) * gamma * qNextmax
#
#
#        # Update value function
#        qCurrTarget = qCurr
#        qCurrTarget[0][action] = target
#        trainTabular(getBoolBits(state),qCurrTarget)


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
        
        state = np.copy(nextState)




if __name__ == '__main__':
    main()


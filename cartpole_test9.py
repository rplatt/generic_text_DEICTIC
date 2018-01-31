#
# This version solves TestRob3Env ghost gridworld problem using a cnn representation.
# The agent gets r=+1 on each time step it survives. Starts in a random position.
#
# This is my first deictic implementation using DL. It calculates the min value by
# taking a min over the batch. We're not explicitly estimating the min values -- we're
# just setting the targets to the min value for that deictic path (instead of the 
# avg value) in the batch.
#
# So far, I get the best performance out of this thing by setting the batch size 
# to 512. 
#
import gym
import numpy as np
import tensorflow as tf
import tf_util_rob as U
import models as models
import build_graph as build_graph
from replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from schedules import LinearSchedule
import tempfile
import matplotlib.pyplot as plt
import envs.frozen_lake



def main():

#    env = gym.make("CartPoleRob-v0")
#    env = gym.make("CartPole-v0")
#    env = gym.make("CartPole-v1")
#    env = gym.make("Acrobot-v1")
#    env = gym.make("MountainCarRob-v0")
#    env = gym.make("FrozenLake-v0")
#    env = gym.make("FrozenLake8x8-v0")
#    env = gym.make("FrozenLake8x8rob-v0")
#    env = gym.make("FrozenLake16x16rob-v0")
    env = gym.make("TestRob3-v0")
    
    
    
    # same as getDeictic except this one just calculates for the observation
    # input: n x n x channels
    # output: dn x dn x channels
    def getDeicticObs(obses_t, windowLen):
        deicticObses_t = []
        for i in range(np.shape(obses_t)[0] - windowLen + 1):
            for j in range(np.shape(obses_t)[1] - windowLen + 1):
                deicticObses_t.append(obses_t[i:i+windowLen,j:j+windowLen,:])
        return np.array(deicticObses_t)

    # get set of deictic alternatives
    # input: batch x n x n x channels
    # output: (batch x deictic) x dn x dn x channels
    def getDeictic(obses_t, actions, obses_tp1, weights, windowLen):
        deicticObses_t = []
        deicticActions = []
        deicticObses_tp1 = []
        deicticWeights = []
        for i in range(np.shape(obses_t)[0]):
            for j in range(np.shape(obses_t)[1] - windowLen + 1):
                for k in range(np.shape(obses_t)[2] - windowLen + 1):
                    deicticObses_t.append(obses_t[i,j:j+windowLen,k:k+windowLen,:])
                    deicticActions.append(actions[i])
                    deicticObses_tp1.append(obses_tp1[i,j:j+windowLen,k:k+windowLen,:])
                    deicticWeights.append(weights[i])
            
        return np.array(deicticObses_t), np.array(deicticActions), np.array(deicticObses_tp1), np.array(deicticWeights)

    # conv model parameters: (num_outputs, kernel_size, stride)
    model = models.cnn_to_mlp(
#        convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)], # used in pong
#        hiddens=[256],  # used in pong
#        convs=[(8,4,1)], # used for non-deictic TestRob3-v0
#        convs=[(8,3,1)], # used for deictic TestRob3-v0
        convs=[(16,3,1)], # used for deictic TestRob3-v0
#        convs=[(4,3,1)], # used for deictic TestRob3-v0
#        convs=[(16,3,1)], # used for deictic TestRob3-v0
#        convs=[(8,2,1)], # used for deictic TestRob3-v0
        hiddens=[16],
        dueling=True
    )

#    model = models.mlp([6])

    # parameters
    q_func=model
    lr=1e-3
#    lr=1e-4
#    max_timesteps=100000
#    max_timesteps=50000
    max_timesteps=20000
    buffer_size=50000
#    exploration_fraction=0.1
    exploration_fraction=0.2
    exploration_final_eps=0.02
#    exploration_final_eps=0.005
#    exploration_final_eps=0.1
    print_freq=10
    checkpoint_freq=10000
    learning_starts=1000
    gamma=.98
    target_network_update_freq=500
    prioritized_replay=False
#    prioritized_replay=True
    prioritized_replay_alpha=0.6
    prioritized_replay_beta0=0.4
    prioritized_replay_beta_iters=None
    prioritized_replay_eps=1e-6
    num_cpu=16
    
#    batch_size=32
#    train_freq=1
#    batch_size=64
#    train_freq=2
#    batch_size=128
#    train_freq=4
#    batch_size=256
#    train_freq=4
    batch_size=512
    train_freq=8

    # deicticShape must be square. 
    # These two parameters need to be consistent w/ each other.
#    deicticShape = (2,2,1)
#    num_deictic_patches=36
    deicticShape = (3,3,1)
    num_deictic_patches=36
#    deicticShape = (4,4,1)
#    num_deictic_patches=25
#    deicticShape = (5,5,1)
#    num_deictic_patches=16
#    deicticShape = (6,6,1)
#    num_deictic_patches=9
#    deicticShape = (7,7,1)
#    num_deictic_patches=4
#    deicticShape = (8,8,1)
#    num_deictic_patches=1

    def make_obs_ph(name):
#        return U.BatchInput(env.observation_space.shape, name=name)
        return U.BatchInput(deicticShape, name=name)

    matchShape = (batch_size*25,)
    def make_match_ph(name):
        return U.BatchInput(matchShape, name=name)

    
    sess = U.make_session(num_cpu)
    sess.__enter__()

#    act, train, update_target, debug = build_graph.build_train(
#    getq, train, trainWOUpdate, update_target, debug = build_graph.build_train_deictic(
#    getq, train, trainWOUpdate, debug = build_graph.build_train_deictic(
#    getq, train, trainWOUpdate, update_target, debug = build_graph.build_train_deictic(
    getq, train, trainWOUpdate, update_target, debug = build_graph.build_train_deictic_min(
        make_obs_ph=make_obs_ph,
        make_match_ph=make_match_ph,
        q_func=q_func,
        num_actions=env.action_space.n,
        batch_size=batch_size,
        num_deictic_patches=num_deictic_patches,
        optimizer=tf.train.AdamOptimizer(learning_rate=lr),
        gamma=gamma,
        grad_norm_clipping=10,
        double_q=False
    )

    act_params = {
        'make_obs_ph': make_obs_ph,
        'q_func': q_func,
        'num_actions': env.action_space.n,
    }

    # Create the replay buffer
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

    # Create the schedule for exploration starting from 1.
    exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * max_timesteps),
                                 initial_p=1.0,
                                 final_p=exploration_final_eps)

    # Initialize the parameters and copy them to the target network.
    U.initialize()
    update_target()


    episode_rewards = [0.0]
    saved_mean_reward = None
    obs = env.reset()
    
#    with tempfile.TemporaryDirectory() as td:
    model_saved = False
#        model_file = os.path.join(td, "model")
    for t in range(max_timesteps):
        
        # get action to take
#        action = act(np.array(obs)[None], update_eps=exploration.value(t))[0]
#        qvalues = getq(np.array(obs)[None])
#        action = np.argmax(qvalues)
#        if np.random.rand() < exploration.value(t):
#            action = np.random.randint(env.action_space.n)
        
        deicticObs = getDeicticObs(obs,deicticShape[0])
        qvalues = getq(np.array(deicticObs))
        action = np.argmax(np.max(qvalues,0))
        selPatch = np.argmax(np.max(qvalues,1))
        if np.random.rand() < exploration.value(t):
            action = np.random.randint(env.action_space.n)
        
#        # temporarily take uniformly random actions all the time
#        action = np.random.randint(env.action_space.n)
#        env.render()
        
        new_obs, rew, done, _ = env.step(action)
        
        # display state, action, nextstate
        if t > 20000:
            toDisplay = np.reshape(new_obs,(8,8))
            toDisplay[np.int32(np.floor_divide(selPatch, np.sqrt(num_deictic_patches))),np.int32(np.remainder(selPatch, np.sqrt(num_deictic_patches)))] = 50
            print("Current/next state. 50 denotes the upper left corner of the deictic patch.")
            print(str(toDisplay))
        
#        env.render()
        
        # Store transition in the replay buffer.
        replay_buffer.add(obs, action, rew, new_obs, float(done))
        obs = new_obs
        
        episode_rewards[-1] += rew
        if done:
            obs = env.reset()
            episode_rewards.append(0.0)
            if t > 20000:
                print("q-values:")
                print(str(qvalues))
                print("*** Episode over! ***\n\n")

        if t > learning_starts and t % train_freq == 0:
            
            # Get batch
            if prioritized_replay:
                experience = replay_buffer.sample(batch_size, beta=beta_schedule.value(t))
                (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
            else:
                obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(batch_size)
                weights, batch_idxes = np.ones_like(rewards), None
            
            # Convert batch to deictic format
            obses_t_deic, actions_deic, obses_tp1_deic, weights_deic = getDeictic(obses_t, actions, obses_tp1, weights, deicticShape[0])
            
            obses_t_deic_fingerprints = [np.reshape(obses_t_deic[i],[deicticShape[0] * deicticShape[1]]) for i in range(np.shape(obses_t_deic)[0])]
            _, _, fingerprintMatch = np.unique(obses_t_deic_fingerprints,axis=0,return_index=True,return_inverse=True)
#            matchTemplates = [fingerprintMatch == i for i in range(np.max(fingerprintMatch)+1)]
            
#            td_errors = train(obses_t, actions, rewards, obses_tp1, dones, weights)
#            td_errors = train(obses_t_deic, actions_deic, rewards, obses_tp1_deic, dones, weights_deic)
#            debug1, debug2, debug3 = trainWOUpdate(obses_t_deic, actions_deic, rewards, obses_tp1_deic, dones, weights_deic)
#            debug1, debug2, debug3, debug4 = trainWOUpdate(obses_t_deic, actions_deic, rewards, obses_tp1_deic, fingerprintMatch, dones, weights_deic)
#            td_errors = train(obses_t_deic, actions_deic, rewards, obses_tp1_deic, fingerprintMatch, dones, weights_deic)
#            td_errors2, min_values_of_groups2, match_onehot2 = train(obses_t_deic, actions_deic, rewards, obses_tp1_deic, fingerprintMatch, dones, weights_deic)

            td_errors, min_values_of_groups, match_onehot = train(obses_t_deic, actions_deic, rewards, obses_tp1_deic, fingerprintMatch, dones, weights_deic)

            if prioritized_replay:
                new_priorities = np.abs(td_errors) + prioritized_replay_eps
                replay_buffer.update_priorities(batch_idxes, new_priorities)

        if t > learning_starts and t % target_network_update_freq == 0:
            
            # Update target network periodically.
            update_target()

        mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
        num_episodes = len(episode_rewards)
        
        if done and print_freq is not None and len(episode_rewards) % print_freq == 0:
            print("steps: " + str(t) + ", episodes: " + str(num_episodes) + ", mean 100 episode reward: " + str(mean_100ep_reward) + ", % time spent exploring: " + str(int(100 * exploration.value(t))))
            
            if t > learning_starts and t % train_freq == 0:
                group_counts = np.sum(match_onehot,1)
                print(str(min_values_of_groups[min_values_of_groups < 1000]))
#                print(str(min_values_of_groups2[min_values_of_groups2 < 1000]))
                print(str(group_counts[group_counts > 0]))
                
                # display one of most valuable deictic patches
                min_values_of_groups_trunc = min_values_of_groups[min_values_of_groups < 1000]
                most_valuable_patches_idx = np.argmax(min_values_of_groups_trunc)
                most_valuable_patches = obses_t_deic[fingerprintMatch == most_valuable_patches_idx]
                print(str(np.reshape(most_valuable_patches[0],deicticShape[0:2])))
                print("value of most valuable patch: " + str(min_values_of_groups_trunc[most_valuable_patches_idx]))
                print("sum group counts: " + str(np.sum(group_counts)))

    num2avg = 20
    rListAvg = np.convolve(episode_rewards,np.ones(num2avg))/num2avg
    plt.plot(rListAvg)
#    plt.plot(episode_rewards)
    plt.show()

    sess


if __name__ == '__main__':
    main()


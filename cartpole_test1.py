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


def main():

    def make_obs_ph(name):
        return U.BatchInput(env.observation_space.shape, name=name)
#        return U.BatchInput(env.observation_space.shape, name)

#    env = gym.make("CartPole-v0")
#    env = gym.make("CartPole-v1")
#    env = gym.make("Acrobot-v1")
    env = gym.make("MountainCar-v0")
#    model = models.mlp([32])
    model = models.mlp([64])
#    model = models.mlp([16, 16])

    # parameters
    q_func=model
    lr=1e-3
    max_timesteps=100000
#    max_timesteps=10000
    buffer_size=50000
    exploration_fraction=0.1
#    exploration_fraction=0.3
    exploration_final_eps=0.02
    train_freq=1
    batch_size=32
    print_freq=10
    checkpoint_freq=10000
    learning_starts=1000
    gamma=1.0
    target_network_update_freq=500
#    prioritized_replay=False
    prioritized_replay=True
    prioritized_replay_alpha=0.6
    prioritized_replay_beta0=0.4
    prioritized_replay_beta_iters=None
    prioritized_replay_eps=1e-6
    num_cpu=16

#    # try mountaincar w/ different input dimensions
#    inputDims = [50,2]
    
    sess = U.make_session(num_cpu)
    sess.__enter__()

    act, train, update_target, debug = build_graph.build_train(
        make_obs_ph=make_obs_ph,
        q_func=q_func,
        num_actions=env.action_space.n,
        optimizer=tf.train.AdamOptimizer(learning_rate=lr),
        gamma=gamma,
        grad_norm_clipping=10
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

        # Take action and update exploration to the newest value
        action = act(np.array(obs)[None], update_eps=exploration.value(t))[0]
        new_obs, rew, done, _ = env.step(action)
        
        # Store transition in the replay buffer.
        replay_buffer.add(obs, action, rew, new_obs, float(done))
        obs = new_obs
        
        episode_rewards[-1] += rew
        if done:
            obs = env.reset()
            episode_rewards.append(0.0)

        if t > learning_starts and t % train_freq == 0:
            
            # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
            if prioritized_replay:
                experience = replay_buffer.sample(batch_size, beta=beta_schedule.value(t))
                (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
            else:
                obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(batch_size)
                weights, batch_idxes = np.ones_like(rewards), None
            
            td_errors = train(obses_t, actions, rewards, obses_tp1, dones, weights)

            if prioritized_replay:
                new_priorities = np.abs(td_errors) + prioritized_replay_eps
                replay_buffer.update_priorities(batch_idxes, new_priorities)

        if t > learning_starts and t % target_network_update_freq == 0:
            
            # Update target network periodically.
            update_target()

        mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
        num_episodes = len(episode_rewards)
        
        if done and print_freq is not None and len(episode_rewards) % print_freq == 0:
#        if done:
            print("steps: " + str(t) + ", episodes: " + str(num_episodes) + ", mean 100 episode reward: " + str(mean_100ep_reward) + ", % time spent exploring: " + str(int(100 * exploration.value(t))))
#            if done and print_freq is not None and len(episode_rewards) % print_freq == 0:
#                logger.record_tabular("steps", t)
#                logger.record_tabular("episodes", num_episodes)
#                logger.record_tabular("mean 100 episode reward", mean_100ep_reward)
#                logger.record_tabular("% time spent exploring", int(100 * exploration.value(t)))
#                logger.dump_tabular()
#        sess
            
    plt.plot(episode_rewards)
    plt.show()
       
    sess


if __name__ == '__main__':
    main()


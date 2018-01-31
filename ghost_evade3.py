#
# Ghost_evade w/ standard DQN at work.
#
#
import gym
import numpy as np
import time as time
import tensorflow as tf
import tf_util_rob as U
import models as models
import build_graph_rob3 as build_graph
from replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from schedules import LinearSchedule
import matplotlib.pyplot as plt

# I had problems w/ the gym environment, so here I made my own standalone class
#import envs.frozen_lake
#import envs.ballcatch1_standalone as envstandalone
import envs.testrob3_standalone as envstandalone

def main():

#    env = envstandalone.BallCatch()
    env = envstandalone.TestRob3Env()
    
    max_timesteps=40000
    learning_starts=1000
    buffer_size=50000
#    buffer_size=1000
    exploration_fraction=0.2
    exploration_final_eps=0.02
    print_freq=10
    gamma=.98
    target_network_update_freq=500
    learning_alpha = 0.2
    
    batch_size=32
    train_freq=1

    obsShape = (8,8,1)
    deicticShape = (3,3,1)
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
                deicticObs.append(obs[i:i+windowLen,j:j+windowLen,:])
        return np.array(deicticObs)

    # conv model parameters: (num_outputs, kernel_size, stride)
    model = models.cnn_to_mlp(
#        convs=[(16,3,1)],
        convs=[(16,2,1)],
#        convs=[(32,3,1)],
        hiddens=[16],
#        hiddens=[64],
#        dueling=True
        dueling=False
    )

    q_func=model
#    lr=1e-3
    lr=0.001
    
    def make_obs_ph(name):
#        return U.BatchInput(deicticShape, name=name)
        return U.BatchInput(obsShape, name=name)

    def make_target_ph(name):
        return U.BatchInput([num_actions], name=name)

    sess = U.make_session(num_cpu)
    sess.__enter__()

    getq, targetTrain = build_graph.build_train_nodouble(
        make_obs_ph=make_obs_ph,
        make_target_ph=make_target_ph,
        q_func=q_func,
        num_actions=env.action_space.n,
        optimizer=tf.train.AdamOptimizer(learning_rate=lr),
        grad_norm_clipping=10,
        double_q=False
    )

    # Initialize the parameters and copy them to the target network.
    U.initialize()

    replay_buffer = ReplayBuffer(buffer_size)
    obs = env.reset()

    timerStart = time.time()
    for t in range(max_timesteps):

        # Get current q-values: neural network version        
        qCurr = getq(np.array([obs]))
        
        # select action
        qCurrNoise = qCurr + np.random.random(np.shape(qCurr))*0.01 # add small amount of noise to break ties randomly
        action = np.argmax(qCurrNoise,1)
        if np.random.rand() < exploration.value(t):
            action = np.random.randint(env.action_space.n)

        # take action
        new_obs, rew, done, _ = env.step(action)
        replay_buffer.add(obs, action, rew, new_obs, float(done))
        
#        # debug
#        if t > 5000:
#            print("obs:\n" + str(np.squeeze(obs)))
#            print("qCurr:\n" + str(qCurr))
#            print("action: " + str(action) + ", patch: " + str(selPatch))
#            print("close:\n" + str(obsDeictic[selPatch,:,:,0] + obsDeictic[selPatch,:,:,1]))
#            print("far:\n" + str(obsDeictic[selPatch,:,:,2] + obsDeictic[selPatch,:,:,3]))
#            action
            
        # sample from replay buffer and train
        if t > learning_starts and t % train_freq == 0:

            # Sample from replay buffer
            obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(batch_size)
            actions = np.int32(np.reshape(actions,[batch_size,]))
            
            # Get curr, next values: neural network version
            qNext = getq(obses_tp1)
            qCurr = getq(obses_t)

            # Get targets
            qNextmax = np.max(qNext,1)
            targets = rewards + (1-dones) * gamma * qNextmax

            qCurrTargets = np.zeros(np.shape(qCurr))
            for i in range(num_actions):
                myActions = actions == i
                qCurrTargets[:,i] = myActions * targets + (1 - myActions) * qCurr[:,i]
            
            # Update values: neural network version
            td_error_out, obses_out, targets_out = targetTrain(
                    obses_t,
                    qCurrTargets
                    )

            td_error_pre = qCurr[range(batch_size),actions] - targets
            
#            print("td error pre-update: " + str(np.linalg.norm(td_error_pre)))

            # neural network version
            qCurr = getq(obses_t)
            
            td_error_post = qCurr[range(batch_size),actions] - targets
#            print("td error post-update: " + str(np.linalg.norm(td_error_post)))

                
        # bookkeeping for storing episode rewards
        episode_rewards[-1] += rew
        if done:
            new_obs = env.reset()
            episode_rewards.append(0.0)
        mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
        num_episodes = len(episode_rewards)
        if done and print_freq is not None and len(episode_rewards) % print_freq == 0:
#            print("steps: " + str(t) + ", episodes: " + str(num_episodes) + ", mean 100 episode reward: " + str(mean_100ep_reward) + ", % time spent exploring: " + str(int(100 * exploration.value(t))) + ", max q at curr state: " + str(np.max(qCurr)))
            timerFinal = time.time()
            print("steps: " + str(t) + ", episodes: " + str(num_episodes) + ", mean 100 episode reward: " + str(mean_100ep_reward) + ", % time spent exploring: " + str(int(100 * exploration.value(t))) + ", time elapsed: " + str(timerFinal - timerStart))
            timerStart = timerFinal
        
        obs = new_obs
        


        

if __name__ == '__main__':
    main()


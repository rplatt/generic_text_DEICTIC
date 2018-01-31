#
# This version solves TestRob3Env ghost gridworld problem using a cnn representation.
# The agent gets r=+1 on each time step it survives. Starts in a random position.
#
# This is an attempt at a streamlined version of cartpole_test9.py. 
# We're still not explicitly estimating the min values -- we're
# just setting the targets to the min value for that deictic path (instead of the 
# avg value) in the batch.
#
# 
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
    
    
    # input: batch x nxnx1 tensor of observations
    def convertState(observations):
        shape = np.shape(observations)
        observations_small = np.squeeze(observations)
        agent_pos = np.nonzero(observations_small == 10)
        ghost_pos = np.nonzero(observations_small == 20)
        state_numeric = 3*np.ones((4,shape[0]))
        state_numeric[0,agent_pos[0]] = agent_pos[1]
        state_numeric[1,agent_pos[0]] = agent_pos[2]
        state_numeric[2,ghost_pos[0]] = ghost_pos[1]
        state_numeric[3,ghost_pos[0]] = ghost_pos[2]
        return np.int32(state_numeric)


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

    # Get deictic patch and action groupings
    # input: obses_deic, actions_deic -> Nx.. a bunch of deictic patches and actions
    # output: groups -> assignment of each row in obses_deic, actions_deic to a group
#    def getDeicticGroups(obses_deic, actions_deic, max_num_groups):
    def getDeicticGroups(obses_deic, max_num_groups):
        
        # create groups of equal obs/actions
        shape = np.shape(obses_deic)
        obses_deic_flat = np.reshape(obses_deic,[shape[0], shape[1]*shape[2]])
        _, group_matching, group_counts = np.unique(obses_deic_flat,axis=0,return_inverse=True,return_counts=True)
        
#        obses_actions_deic_flat = np.c_[obses_deic_flat,actions_deic]
#        _, group_matching, group_counts = np.unique(obses_actions_deic_flat,axis=0,return_inverse=True,return_counts=True)

#        # take max_num_groups of most frequent groups
#        group_indices = np.float32(np.r_[np.array([group_counts]),np.array([range(np.shape(group_counts)[0])])])
#        group_indices[0] = group_indices[0] + np.random.random(np.shape(group_indices)[1])*0.1 # add small random values to randomize sort order for equal numbers
#        group_indices_sorted = group_indices[:,group_indices[0,:].argsort()]
#        group_indices_to_keep = np.int32(group_indices_sorted[1,-max_num_groups:])        
#
#        # Replace group numbers with new numbers in 0:max_num_groups
#        # All elts with group=max_num_groups have no group.
#        new_group_matching = np.ones(np.shape(group_matching)[0])*max_num_groups
#        for i in range(np.shape(group_indices_to_keep)[0]):
#            idx = np.nonzero(group_matching == group_indices_to_keep[i])
#            new_group_matching[idx] = i
#
#        # Get final list of groups. Get observations, actions corresponding to each group
#        groups,idx = np.unique(new_group_matching,return_index=True)
#        groups_idx = np.r_[np.array([groups]),np.array([idx])]
#        groups_idx_sorted = groups_idx[:,groups_idx[0].argsort()]
#        groups = groups_idx_sorted[0]
#        idx = np.int32(groups_idx_sorted[1,:-1])
#        group_obs = obses_deic_flat[idx]
#        group_actions = actions_deic[idx]
#        
#        # reshape output observations
#        obsshape = np.shape(group_obs)
#        group_obs = np.reshape(group_obs,(obsshape[0],np.int32(np.sqrt(obsshape[1])),np.int32(np.sqrt(obsshape[1])),1))
        
        # Get final list of groups. Get observations, actions corresponding to each group
        groups,idx = np.unique(group_matching,return_index=True)
        group_obs = obses_deic_flat[idx]
        
        # reshape output observations
        obsshape = np.shape(group_obs)
        group_obs = np.reshape(group_obs,(obsshape[0],shape[1],shape[2],shape[3]))


#        return new_group_matching, group_obs, group_actions
#        return group_matching, group_obs
        return group_matching

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
    batch_size=64
    train_freq=2
#    batch_size=128
#    train_freq=4
#    batch_size=256
#    train_freq=4
#    batch_size=512
#    train_freq=8
#    batch_size=1024
#    train_freq=8
#    batch_size=2048
#    train_freq=8
#    batch_size=4096
#    train_freq=8

    max_num_groups = 600

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

    num_actions = 4
    tabularQ = 100*np.ones([deicticShape[0]+1,deicticShape[1]+1,deicticShape[0]+1,deicticShape[1]+1, num_actions])
    OHEnc = np.identity(max_num_groups)


    def make_obs_ph(name):
#        return U.BatchInput(env.observation_space.shape, name=name)
        return U.BatchInput(deicticShape, name=name)

    matchShape = (batch_size*25,)
    def make_match_ph(name):
        return U.BatchInput(matchShape, name=name)

    def parallelUpdate(obses_t_deic, actions_deic, rewards, obses_tp1_deic, group_matching, dones, q_tp1, batch_size, num_deictic_patches, max_num_groups):
        q_tp1_target = rewards + gamma*np.max(np.reshape(np.max(q_tp1,1),[batch_size,num_deictic_patches]),1)
        q_tp1_target = (1-dones) * q_tp1_target
        
        group_matching_onehot = OHEnc[group_matching]
        desc_2_state = np.max(np.reshape(group_matching_onehot,[batch_size, num_deictic_patches, max_num_groups]),1)

        max_target = np.max(q_tp1_target)
        target_min_per_D = np.min(desc_2_state * np.tile(np.reshape(q_tp1_target,[batch_size,1]),[1,max_num_groups]) + (1-desc_2_state) * max_target,0)
        
        # I noticed that the line below produces unpredictable behavior. The dotprod does not seem to produce consistent results for some reason. Use the line below that instead.
#        targets1 = np.dot(group_matching_onehot,target_min_per_D)
        targets = np.sum(group_matching_onehot * np.tile(np.reshape(target_min_per_D,[1,max_num_groups]),[batch_size*num_deictic_patches,1]),1)
                   
        D_2_DI = group_matching_onehot
        
        return q_tp1_target, desc_2_state, target_min_per_D, D_2_DI, targets
    
    sess = U.make_session(num_cpu)
    sess.__enter__()

#    getq, train, trainWOUpdate, update_target, debug = build_graph.build_train_deictic_min(
#    getq, train, trainWOUpdate, update_target, debug = build_graph.build_train_deictic_min_streamlined(
#    getq, trainWOUpdate = build_graph.build_train_deictic_min_streamlined(
    getq, train, trainWOUpdate, update_target = build_graph.build_train_deictic_min_streamlined(
        make_obs_ph=make_obs_ph,
        q_func=q_func,
        num_actions=env.action_space.n,
        batch_size=batch_size,
        num_deictic_patches=num_deictic_patches,
        max_num_groups=max_num_groups,
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
#    update_target()


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
#        qvalues = getq(np.array(deicticObs))
        stateCurr = convertState(deicticObs)
        qvalues = tabularQ[stateCurr[0], stateCurr[1], stateCurr[2], stateCurr[3],:]
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
            group_matching = getDeicticGroups(obses_t_deic, max_num_groups)
            
            stateCurr = convertState(obses_t_deic)
            stateNext = convertState(obses_tp1_deic)
            q_tp1 = tabularQ[stateNext[0], stateNext[1], stateNext[2], stateNext[3],:]
#            q_tp1_target_parallel, desc_2_state_parallel, target_min_per_D_parallel, D_2_DI_parallel, targets_parallel = trainWOUpdate(obses_t_deic, actions_deic, rewards, obses_tp1_deic, group_matching, dones, q_tp1)
            q_tp1_target, desc_2_state, target_min_per_D, D_2_DI, targets = parallelUpdate(obses_t_deic, actions_deic, rewards, obses_tp1_deic, group_matching, dones, q_tp1, batch_size, num_deictic_patches, max_num_groups)
            targets_simple = np.reshape(np.tile(np.reshape(q_tp1_target, [batch_size,1]),[1,num_deictic_patches]), batch_size*num_deictic_patches)
            
            tabularQ[stateCurr[0], stateCurr[1], stateCurr[2], stateCurr[3],actions_deic] = np.minimum(targets_simple,tabularQ[stateCurr[0], stateCurr[1], stateCurr[2], stateCurr[3],actions_deic])
            
#            print("Num unique descriptors in batch: " + str(np.shape(np.unique(group_matching))[0]))
            
            # 

#            for i in range(np.shape(obses_t_deic_small)[0]):
#                if i in agent_pos[0]:
#                    
#                    ax = agent_pos[np.nonzero(agent_pos[0] == i)[0][0]]
#                    ax
            
#            if prioritized_replay:
#                new_priorities = np.abs(td_errors) + prioritized_replay_eps
#                replay_buffer.update_priorities(batch_idxes, new_priorities)

        if t > learning_starts and t % target_network_update_freq == 0:
            
            # Update target network periodically.
            update_target()

        mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
        num_episodes = len(episode_rewards)
        
        if done and print_freq is not None and len(episode_rewards) % print_freq == 0:
            print("steps: " + str(t) + ", episodes: " + str(num_episodes) + ", mean 100 episode reward: " + str(mean_100ep_reward) + ", % time spent exploring: " + str(int(100 * exploration.value(t))))
            print("best patch:\n" + str(np.squeeze(deicticObs[np.argmax(np.max(qvalues,1))])))
            print("worst patch:\n" + str(np.squeeze(deicticObs[np.argmin(np.max(qvalues,1))])))
#            if t > learning_starts:
#                print("max td_error: " + str(np.sort(td_error)[-10:]))
            
                
    num2avg = 20
    rListAvg = np.convolve(episode_rewards,np.ones(num2avg))/num2avg
    plt.plot(rListAvg)
#    plt.plot(episode_rewards)
    plt.show()

    sess


if __name__ == '__main__':
    main()


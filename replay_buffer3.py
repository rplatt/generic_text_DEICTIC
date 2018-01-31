# 
# The _encode_sample function in this version of replay_buffer is adapted to allow 
# a discrete action as well as a patch action.
#
# Adapted from replay_buffer2.py 
#
import numpy as np
import random

#from baselines.common.segment_tree import SumSegmentTree, MinSegmentTree
from segment_tree import SumSegmentTree, MinSegmentTree


class ReplayBuffer(object):
    def __init__(self, size):
        """Create Prioritized Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

#    def add(self, obs_t, action, reward, obs_tp1, done):
    def add(self, obs_t, action_discrete, action_patch, reward, obs_tp1, done):
#        data = (obs_t, action, reward, obs_tp1, done)
        data = (obs_t, action_discrete, action_patch, reward, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
#        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        states_t, actions_discrete, actions_patch, rewards, images_tp1, states_tp1, dones = [], [], [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
#            state_t, action, reward, obses_tp1, done = data
            state_t, action_discrete, action_patch, reward, obses_tp1, done = data
            states_t.append(np.array(state_t, copy=False))
#            actions.append(np.array(action, copy=False))
            actions_discrete.append(np.array(action_discrete, copy=False))
            actions_patch.append(np.array(action_patch, copy=False))
            rewards.append(reward)
            images_tp1.append(np.array(obses_tp1[0], copy=False))
            states_tp1.append(np.array(obses_tp1[1], copy=False))
            dones.append(done)
#        return np.array(states_t), np.array(actions), np.array(rewards), np.array(images_tp1), np.array(states_tp1), np.array(dones)
        return np.array(states_t), np.array(actions_discrete), np.array(actions_patch), np.array(rewards), np.array(images_tp1), np.array(states_tp1), np.array(dones)

    def sample(self, batch_size):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)

    # Reassign the rewards of all steps in the last episode in the buffer to 
    # reflect the true experience monte carlo reward.
    def update_montecarlo(self, gamma):
        
        idx = self._next_idx-1
        if idx < 0:
            idx = min(self._maxsize,len(self._storage)) - 1
            
        obs_t, action, reward, obs_tp1, done = self._storage[idx]
        if not done:
            print("replay_buffer.update_montecarlo ERROR! Last entry into buffer must have a positive done flag!")                    
            
        accReward = reward
        for i in range(idx-1,-self._maxsize,-1):
            ii = i
            if i < 0:
                ii = i+min(self._maxsize,len(self._storage))
            obs_t, action, reward, obs_tp1, done = self._storage[ii]
            if done:
                return
            accReward = gamma*accReward + reward
            self._storage[ii] = (obs_t, action, accReward, obs_tp1, done)
        return
    

class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, size, alpha):
        """Create Prioritized Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        alpha: float
            how much prioritization is used
            (0 - no prioritization, 1 - full prioritization)

        See Also
        --------
        ReplayBuffer.__init__
        """
        super(PrioritizedReplayBuffer, self).__init__(size)
        assert alpha > 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def add(self, *args, **kwargs):
        """See ReplayBuffer.store_effect"""
        idx = self._next_idx
#        super().add(*args, **kwargs)
        super(PrioritizedReplayBuffer,self).add(*args, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        for _ in range(batch_size):
            # TODO(szymon): should we ensure no repeats?
            mass = random.random() * self._it_sum.sum(0, len(self._storage) - 1)
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size, beta):
        """Sample a batch of experiences.

        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.


        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        beta: float
            To what degree to use importance weights
            (0 - no corrections, 1 - full correction)

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        weights: np.array
            Array of shape (batch_size,) and dtype np.float32
            denoting importance weight of each sampled transition
        idxes: np.array
            Array of shape (batch_size,) and dtype np.int32
            idexes in buffer of sampled experiences
        """
        assert beta > 0

        idxes = self._sample_proportional(batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)
        encoded_sample = self._encode_sample(idxes)
        return tuple(list(encoded_sample) + [weights, idxes])

    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions.

        sets priority of transition at index idxes[i] in buffer
        to priorities[i].

        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)


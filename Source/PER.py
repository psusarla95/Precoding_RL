import torch
import numpy as np
from collections import deque
import random
from collections import namedtuple, deque
import operator

class SegmentTree(object):
    def __init__(self, capacity, operation, neutral_element):
        """Build a Segment Tree data structure.
        https://en.wikipedia.org/wiki/Segment_tree
        Can be used as regular array, but with two
        important differences:
            a) setting item's value is slightly slower.
               It is O(lg capacity) instead of O(1).
            b) user has access to an efficient ( O(log segment size) )
               `reduce` operation which reduces `operation` over
               a contiguous subsequence of items in the array.
        Paramters
        ---------
        capacity: int
            Total size of the array - must be a power of two.
        operation: lambda obj, obj -> obj
            and operation for combining elements (eg. sum, max)
            must form a mathematical group together with the set of
            possible values for array elements (i.e. be associative)
        neutral_element: obj
            neutral element for the operation above. eg. float('-inf')
            for max and 0 for sum.
        """
        assert capacity > 0 and capacity & (capacity - 1) == 0, "capacity must be positive and a power of 2."
        self._capacity = capacity
        self._value = [neutral_element for _ in range(2 * capacity)]
        self._operation = operation

    def _reduce_helper(self, start, end, node, node_start, node_end):
        if start == node_start and end == node_end:
            return self._value[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._reduce_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._reduce_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self._operation(
                    self._reduce_helper(start, mid, 2 * node, node_start, mid),
                    self._reduce_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end)
                )

    def reduce(self, start=0, end=None):
        """Returns result of applying `self.operation`
        to a contiguous subsequence of the array.
            self.operation(arr[start], operation(arr[start+1], operation(... arr[end])))
        Parameters
        ----------
        start: int
            beginning of the subsequence
        end: int
            end of the subsequences
        Returns
        -------
        reduced: obj
            result of reducing self.operation over the specified range of array elements.
        """
        if end is None:
            end = self._capacity
        if end < 0:
            end += self._capacity
        end -= 1
        return self._reduce_helper(start, end, 1, 0, self._capacity - 1)

    def __setitem__(self, idx, val):
        # index of the leaf
        idx += self._capacity
        self._value[idx] = val
        idx //= 2
        while idx >= 1:
            self._value[idx] = self._operation(
                self._value[2 * idx],
                self._value[2 * idx + 1]
            )
            idx //= 2

    def __getitem__(self, idx):
        assert 0 <= idx < self._capacity
        return self._value[self._capacity + idx]


class SumSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(SumSegmentTree, self).__init__(
            capacity=capacity,
            operation=operator.add,
            neutral_element=0.0
        )

    def sum(self, start=0, end=None):
        """Returns arr[start] + ... + arr[end]"""
        return super(SumSegmentTree, self).reduce(start, end)

    def find_prefixsum_idx(self, prefixsum):
        """Find the highest index `i` in the array such that
            sum(arr[0] + arr[1] + ... + arr[i - i]) <= prefixsum
        if array values are probabilities, this function
        allows to sample indexes according to the discrete
        probability efficiently.
        Parameters
        ----------
        perfixsum: float
            upperbound on the sum of array prefix
        Returns
        -------
        idx: int
            highest index satisfying the prefixsum constraint
        """
        assert 0 <= prefixsum <= self.sum() + 1e-5
        idx = 1
        while idx < self._capacity:  # while non-leaf
            if self._value[2 * idx] > prefixsum:
                idx = 2 * idx
            else:
                prefixsum -= self._value[2 * idx]
                idx = 2 * idx + 1
        return idx - self._capacity


class MinSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(MinSegmentTree, self).__init__(
            capacity=capacity,
            operation=min,
            neutral_element=float('inf')
        )

    def min(self, start=0, end=None):
        """Returns min(arr[start], ...,  arr[end])"""

        return super(MinSegmentTree, self).reduce(start, end)



experience = namedtuple("experience", field_names=["state", "action", "reward", "next_state", "done"])

class PrioritizedReplayBuffer(object):
    """
    Fixed-size buffer to store experience tuples

    Note:
        This finite buffer size adds noise to the outputs on function approx.
        larger the buffer size, more is the experience and less the number of episodes needed for training
    """

    def __init__(self, action_size, buffer_size, batch_size, buf_alpha, seed, device):
        """
        Initialize a ReplayBuffer object
        :param action_size (int): dimension of each action
        :param buffer_size (int): maximum size of the buffer
        :param batch_size (int): size of each training batch
        :param seed (int): random seed
        """

        self.action_size = action_size
        self.memory = []#deque(maxlen=buffer_size)
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.next_ndx = 0

        assert buf_alpha >= 0
        self.alpha = buf_alpha
        #self.experience =namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed =random.seed(seed)
        self.device = device

        # Num of nodes of a tree, should be in orders of '2'
        iter_capacity = 1
        while iter_capacity < buffer_size:
            iter_capacity *= 2

        self.iter_sum = SumSegmentTree(iter_capacity)
        self.iter_min = MinSegmentTree(iter_capacity)
        self.max_p = 1.0

    def add(self, state, action, reward, next_state, done):
        #e = self.experience(state, action, reward, next_state, done)
        e = experience(state, action, reward, next_state, done)

        if self.next_ndx >= len(self.memory):
            self.memory.append(e)
        else:
            self.memory[self.next_ndx] = e

        self.next_ndx = (self.next_ndx + 1) % self.buffer_size
        #self.memory.append(e)
        self.iter_sum[self.next_ndx] = self.max_p**self.alpha
        self.iter_min[self.next_ndx] = self.max_p**self.alpha

    def _sample_proportional(self):
        res = []
        p_total = self.iter_sum.sum(0,len(self.memory)-1)
        every_range_len = p_total / self.batch_size
        for i in range(self.batch_size):
            mass = np.random.random()* every_range_len + i*every_range_len
            idx = self.iter_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, beta):
        assert beta > 0
        idxes = self._sample_proportional()

        weights = []
        p_min = self.iter_min.min() / self.iter_sum.sum()
        """
            P(i) = (p_i)^(alpha) /(Sum_k (p_k)^(alpha))
            w_i = ((1/N)*(1/P(i)))^(beta)
        """
        max_weight = (p_min*len(self.memory))**(-beta)

        """Sample a batch of experiences from memory based on selected indices"""
        experiences = []
        for ndx in idxes:
            p_sample = self.iter_sum[ndx] / self.iter_sum.sum()
            weight = (p_sample*len(self.memory))**(-beta)
            weights.append(weight/max_weight)
            experiences.append(self.memory[ndx])


        #experiences = random.sample(self.memory, k=self.batch_size)
        #experiences = [e for e in experiences if e is not None]
        #Convert batch of experiences to experiences of batches
        #batch = self.experience(*zip(*experiences))
        batch = experience(*zip(*experiences))

        batch_weights = [torch.tensor([x]).to(self.device) for x in weights]
        batch_idxes = [torch.tensor([x]).to(self.device) for x in idxes]
        weights = torch.cat(batch_weights).to(self.device)
        idxes = torch.cat(batch_idxes).to(self.device)

        #print(batch.reward)
        states = torch.cat(batch.state).to(self.device)
        actions = torch.cat(batch.action).to(self.device)
        rewards = torch.cat(batch.reward).to(self.device)
        next_states = torch.cat(batch.next_state).to(self.device)
        dones = torch.cat(batch.done).to(self.device)
        #states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        #actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        #rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        #next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        #dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, dones, weights, idxes)

    def update_priorities(self, idxes, priorities):
        """
        Update priorities of sampled transitions

        Sets priority of transition at index idxes[i] in buffer to priorities[i]
        :param idxes [int]: list of sampled transitions
        :param priorities [float]: list of updated priorities corresponding
               to transitions at the sampled idxes denoted by variable 'idxes'
        """
        assert len(idxes) == len(priorities)
        for ndx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= ndx < len(self.memory)
            self.iter_sum[ndx] = priority ** self.alpha
            self.iter_min[ndx] = priority ** self.alpha

            self.max_p = max(self.max_p, priority)

    def can_provide_sample(self):
        return len(self.memory) >= self.batch_size

    def __len__(self):
        "Return the current size of replay memory"
        return len(self.memory)

    def save(self, filename):
        with open(filename, 'wb') as f:
            torch.save(self.memory, f)

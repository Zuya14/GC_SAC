import torch
import numpy as np

class ReplayBuffer:

    def __init__(self, buffer_size, state_size, velocity_size, observation_size, action_size, goal_size, device):
        # 次にデータを挿入するインデックス．
        self._p = 0
        # データ数．
        self._n = 0
        # リプレイバッファのサイズ．
        self.buffer_size = buffer_size

        # GPU上に保存するデータ．
        self.states = torch.empty((buffer_size, *state_size), dtype=torch.float, device=device)
        self.velocitys = torch.empty((buffer_size, *velocity_size), dtype=torch.float, device=device)
        self.observations = torch.empty((buffer_size, *observation_size), dtype=torch.float, device=device)
        self.actions = torch.empty((buffer_size, *action_size), dtype=torch.float, device=device)
        self.rewards = torch.empty((buffer_size, 1), dtype=torch.float, device=device)
        self.dones = torch.empty((buffer_size, 1), dtype=torch.float, device=device)
        self.next_states = torch.empty((buffer_size, *state_size), dtype=torch.float, device=device)
        self.next_velocitys = torch.empty((buffer_size, *velocity_size), dtype=torch.float, device=device)
        self.next_observations = torch.empty((buffer_size, *observation_size), dtype=torch.float, device=device)
        self.goals = torch.empty((buffer_size, *goal_size), dtype=torch.float, device=device)

    def append(self, state, velocity, observation, action, reward, done, next_state, next_velocity, next_observation, goal):
        self.states[self._p].copy_(torch.from_numpy(state))
        self.velocitys[self._p].copy_(torch.from_numpy(velocity))
        self.observations[self._p].copy_(torch.from_numpy(observation))
        self.actions[self._p].copy_(torch.from_numpy(action))
        self.rewards[self._p] = float(reward)
        self.dones[self._p] = float(done)
        self.next_states[self._p].copy_(torch.from_numpy(next_state))
        self.next_velocitys[self._p].copy_(torch.from_numpy(next_velocity))
        self.next_observations[self._p].copy_(torch.from_numpy(next_observation))
        self.goals[self._p].copy_(torch.from_numpy(goal))

        self._p = (self._p + 1) % self.buffer_size
        self._n = min(self._n + 1, self.buffer_size)

    def sample(self, batch_size):
        idxes = np.random.randint(low=0, high=self._n, size=batch_size)
        return (
            self.states[idxes],
            self.velocitys[idxes],
            self.observations[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.dones[idxes],
            self.next_states[idxes],
            self.next_velocitys[idxes],
            self.next_observations[idxes],
            self.goals[idxes]
        )
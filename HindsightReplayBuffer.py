import torch
import numpy as np
import random

class episode:
    def __init__(self):
        self.states = []
        self.velocitys = []
        self.observation = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.next_states = []
        self.next_velocitys = []
        self.next_observations = []
        self.goals = []
        self.collisions = []

    def append(self, state, velocity, observation, action, reward, done, next_state, next_velocity, next_observation, goal, collision):
        self.states.append(state)
        self.velocitys.append(velocity)
        self.observation.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.next_states.append(next_state)
        self.next_velocitys.append(next_velocity)
        self.next_observations.append(observation)
        self.goals.append(goal)
        self.collisions.append(collision)

    def size(self):
        return len(self.states)

    def __call__(self):
        return [
        self.states,
        self.velocitys,
        self.observation,
        self.actions,
        self.rewards,
        self.dones,
        self.next_states,
        self.next_velocitys,
        self.next_observations,
        self.goals,
        self.collisions
        ]


class HindsightReplayBuffer():

    def __init__(self, buffer_size, state_size, velocity_size, observation_size, action_size, goal_size, device, num_subgoals=4):
        self.device=device
        self.episode = episode()
        self.num_subgoals = num_subgoals

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
        self.collisions = torch.empty((buffer_size, 1), dtype=torch.float, device=device)

    def append(self, state, velocity, observation, action, reward, done, next_state, next_velocity, next_observation, goal, collision, save_episode=True):
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
        self.collisions[self._p] = float(collision)

        if save_episode:
            self.episode.append(state, velocity, observation, action, reward, done, next_state, next_velocity, next_observation, goal, collision)

        self._p = (self._p + 1) % self.buffer_size
        self._n = min(self._n + 1, self.buffer_size)

    def sample(self, batch_size):
        indices = np.random.randint(low=0, high=self._n, size=batch_size)
        return (
            self.states[indices],
            self.velocitys[indices],
            self.observations[indices],
            self.actions[indices],
            self.rewards[indices],
            self.dones[indices],
            self.next_states[indices],
            self.next_velocitys[indices],
            self.next_observations[indices],
            self.goals[indices]
        )

    def resample_goals(self, env):
        states, velocitys, observation, actions, rewards, dones, next_states, next_velocitys, next_observations, goals, collisions = self.episode()
        episode_len = self.episode.size()
        
        for i in range(episode_len-1):
            indices = np.random.randint(low=i, high=episode_len-1, size=self.num_subgoals)

            for index in indices:
                new_goal = next_states[index]
                if not collisions[i]:
                    self.append(states[i], velocitys[i], observation[i], actions[i], env.calc_reward(collisions[i], next_states[i], new_goal), dones[i] or env.sim.isArrive(new_goal, next_states[i]), next_states[i], next_velocitys[i], next_observations[i], new_goal, collisions[i], False)

        self.episode = episode()

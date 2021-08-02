# from DDPG import DDPG, CriticNetwork
from ReplayBuffer import ReplayBuffer
import torch
import torch.nn as nn
import numpy as np
import random
import math

def calculate_log_pi(log_stds, noises, actions):
    """ 確率論的な行動の確率密度を返す． """
    # ガウス分布 `N(0, stds * I)` における `noises * stds` の確率密度の対数(= \log \pi(u|a))を計算する．
    # (torch.distributions.Normalを使うと無駄な計算が生じるので，下記では直接計算しています．)
    gaussian_log_probs = \
        (-0.5 * noises.pow(2) - log_stds).sum(dim=-1, keepdim=True) - 0.5 * math.log(2 * math.pi) * log_stds.size(-1)

    # tanh による確率密度の変化を修正する．
    log_pis = gaussian_log_probs - torch.log(1 - actions.pow(2) + 1e-6).sum(dim=-1, keepdim=True)

    return log_pis


def reparameterize(means, log_stds):
    """ Reparameterization Trickを用いて，確率論的な行動とその確率密度を返す． """
    # 標準偏差．
    stds = log_stds.exp()
    # 標準ガウス分布から，ノイズをサンプリングする．
    noises = torch.randn_like(means)
    # Reparameterization Trickを用いて，N(means, stds)からのサンプルを計算する．
    us = means + noises * stds
    # tanh　を適用し，確率論的な行動を計算する．
    actions = torch.tanh(us)

    # 確率論的な行動の確率密度の対数を計算する．
    log_pis = calculate_log_pi(log_stds, noises, actions)

    return actions, log_pis


def atanh(x):
    """ tanh の逆関数． """
    return 0.5 * (torch.log(1 + x + 1e-6) - torch.log(1 - x + 1e-6))


class ActorNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, 2*action_size),
        )

    def forward(self, states):
        return torch.tanh(self.net(states).chunk(2, dim=-1)[0])

    def sample(self, states):
        means, log_stds = self.net(states).chunk(2, dim=-1)
        return reparameterize(means, log_stds.clamp(-20, 2))

class CriticNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super().__init__()

        self.net1 = nn.Sequential(
            nn.Linear(state_size + action_size, hidden_size),
            nn.ELU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(inplace=True),
            nn.Linear(hidden_size, 1),
        )
        self.net2 = nn.Sequential(
            nn.Linear(state_size + action_size, hidden_size),
            nn.ELU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(inplace=True),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=-1)
        return self.net1(x), self.net2(x)

class GC_SAC():

    def __init__(self, state_size, velocity_size, observation_size, action_size, goal_size, hidden_size=64, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 batch_size=256, gamma=0.99, lr=3e-4,
                 replay_size=10**6, start_steps=10**4, tau=5e-3, alpha=0.2, reward_scale=1.0, epsilon_decay = 50000,
                 automatic_entropy_tuning=True):

        self.name = 'GC_SAC'

        self.buffer = ReplayBuffer(
            buffer_size=replay_size,
            state_size=state_size,
            velocity_size=velocity_size,
            observation_size=observation_size,
            action_size=action_size,
            goal_size=goal_size,
            device=device
        )
        
        # Actor-Criticのネットワークを構築する．
        self.actor = ActorNetwork(
            state_size=state_size[0]+velocity_size[0]+observation_size[0]+goal_size[0],
            action_size=action_size[0],
            hidden_size=hidden_size
        ).to(device)
        self.critic = CriticNetwork(
            state_size=state_size[0]+velocity_size[0]+observation_size[0]+goal_size[0],
            action_size=action_size[0],
            hidden_size=hidden_size
        ).to(device)
        self.critic_target = CriticNetwork(
            state_size=state_size[0]+velocity_size[0]+observation_size[0]+goal_size[0],
            action_size=action_size[0],
            hidden_size=hidden_size
        ).to(device).eval()

        # ターゲットネットワークの重みを初期化し，勾配計算を無効にする．
        self.critic_target.load_state_dict(self.critic.state_dict())
        for param in self.critic_target.parameters():
            param.requires_grad = False

        # オプティマイザ．
        self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.optim_critic = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.device = device

        # その他パラメータ．
        self.action_size = action_size
        self.learning_steps = 0
        self.batch_size = batch_size
        self.device = device
        self.gamma = gamma
        self.start_steps = start_steps
        self.tau = tau
        self.alpha = alpha
        self.reward_scale = reward_scale

        self.automatic_entropy_tuning = automatic_entropy_tuning

        if self.automatic_entropy_tuning:
                self.target_entropy = -torch.prod(torch.Tensor(action_size[0]).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=lr)
    
    def is_update(self, steps):
        # 学習初期の一定期間(start_steps)は学習しない．
        return steps >= max(self.start_steps, self.batch_size)

    def explore(self, obs_all, goal, state):
        """ 確率論的な行動と，その行動の確率密度の対数 \log(\pi(a|s)) を返す． """
        _obs_all = np.concatenate(obs_all)

        # state_input = torch.tensor(np.concatenate([obs_all, goal]), dtype=torch.float, device=self.device).unsqueeze_(0)
        state_input = torch.tensor(np.concatenate([_obs_all, goal-state]), dtype=torch.float, device=self.device).unsqueeze_(0)
        with torch.no_grad():
            action, log_pi = self.actor.sample(state_input)
        return action.cpu().numpy()[0], log_pi.item()

    def exploit(self, obs_all, goal, state):
        """ 決定論的な行動を返す． """
        _obs_all = np.concatenate(obs_all)

        # state_input = torch.tensor(np.concatenate([obs_all, goal]), dtype=torch.float, device=self.device).unsqueeze_(0)
        state_input = torch.tensor(np.concatenate([_obs_all, goal-state]), dtype=torch.float, device=self.device).unsqueeze_(0)
        with torch.no_grad():
            action = self.actor(state_input)
        return action.cpu().numpy()[0]

    def step(self, env, obs_all, goal, t, steps):
        t += 1

        state, velocity, observe = obs_all

        # 学習初期の一定期間(start_steps)は，ランダムに行動して多様なデータの収集を促進する．
        if steps <= self.start_steps:
            action = env.action_space.sample()
        else:
            action, _ = self.explore(obs_all, goal, state)

        next_obs_all, reward, done, _ = env.step(action)

        # ゲームオーバーではなく，最大ステップ数に到達したことでエピソードが終了した場合は，
        # 本来であればその先も試行が継続するはず．よって，終了シグナルをFalseにする．
        # NOTE: ゲームオーバーによってエピソード終了した場合には， done_masked=True が適切．
        # しかし，以下の実装では，"たまたま"最大ステップ数でゲームオーバーとなった場合には，
        # done_masked=False になってしまう．
        # その場合は稀で，多くの実装ではその誤差を無視しているので，今回も無視する．
        if t == env._max_episode_steps:
            done_masked = False
        else:
            done_masked = done

        next_state, next_velocity, next_observe = next_obs_all

        # リプレイバッファにデータを追加する．
        self.buffer.append(
            state, velocity, observe,
            action, reward, done_masked, 
            next_state, next_velocity, next_observe, 
            goal
            )

        # エピソードが終了した場合には，環境をリセットする．
        if done:
            t = 0
            next_obs_all = env.reset()

        return next_obs_all, t

    def update(self):
        self.learning_steps += 1
        
        states, velocitys, observations, actions, rewards, dones, next_states, next_velocitys, next_observations, goals = self.buffer.sample(self.batch_size)
        # states, actions, rewards, dones, collisions, next_states, goals = self.buffer.sample(self.batch_size)

        obs_alls = torch.cat([states, velocitys, observations], dim=-1)
        next_obs_alls = torch.cat([next_states, next_velocitys, next_observations], dim=-1)

        states2 = torch.cat([obs_alls, goals-states], dim=-1)
        next_states2 = torch.cat([next_obs_alls, goals-next_states], dim=-1)

        self.update_critic(states2, next_states2, obs_alls, actions, rewards, dones, next_obs_alls, goals, states, next_states)
        self.update_actor(states2, obs_alls, goals, states)
        self.update_target()

    def update_critic(self, states2, next_states2, obs_alls, actions, rewards, dones, next_obs_alls, goals, states, next_states):
        # states2 = torch.cat([states, goals], dim=-1)
        # states2 = torch.cat([obs_alls, goals-states], dim=-1)
        curr_qs1, curr_qs2 = self.critic(states2, actions)

        with torch.no_grad():
            # next_states2 = torch.cat([next_states, goals], dim=-1)
            # next_states2 = torch.cat([next_obs_alls, goals-next_states], dim=-1)
            next_actions, log_pis = self.actor.sample(next_states2)
            next_qs1, next_qs2 = self.critic_target(next_states2, next_actions)
            next_qs = torch.min(next_qs1, next_qs2) - self.alpha * log_pis
        target_qs = rewards * self.reward_scale + (1.0 - dones) * self.gamma * next_qs

        loss_critic1 = (curr_qs1 - target_qs).pow_(2).mean()
        loss_critic2 = (curr_qs2 - target_qs).pow_(2).mean()

        self.optim_critic.zero_grad()
        (loss_critic1 + loss_critic2).backward(retain_graph=False)
        self.optim_critic.step()

    def update_actor(self, states2, obs_alls, goals, states):
        # states2 = torch.cat([states, goals], dim=-1)
        # states2 = torch.cat([obs_alls, goals-states], dim=-1)
        actions, log_pis = self.actor.sample(states2)
        qs1, qs2 = self.critic(states2, actions)
        loss_actor = (self.alpha * log_pis - torch.min(qs1, qs2)).mean()

        self.optim_actor.zero_grad()
        loss_actor.backward(retain_graph=False)
        self.optim_actor.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pis + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()

    def update_target(self):
        for t, s in zip(self.critic_target.parameters(), self.critic.parameters()):
            t.data.mul_(1.0 - self.tau)
            t.data.add_(self.tau * s.data)

    def save(self, path="./"):
        torch.save(self.actor.to('cpu').state_dict(), path+"GC_SAC_HER_actor.pth")
        self.actor.to(self.device)

        torch.save(self.critic.to('cpu').state_dict(), path+"GC_SAC_HER_critic.pth")
        self.critic.to(self.device)

    def load(self, path="./"):
        self.actor.load_state_dict(torch.load(path+"GC_SAC_HER_actor.pth"))
        self.critic.load_state_dict(torch.load(path+"GC_SAC_HER_critic.pth"))

'''
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

state_t と　state_t+1 のそれぞれに対して goal を concat するのが下位層だと面倒なので上位層（Bufferくらい）でconcatしたい

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
'''
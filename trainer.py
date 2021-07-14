import os
import glob
from time import time
from datetime import timedelta
from base64 import b64encode
import numpy as np
import gym
import matplotlib.pyplot as plt
# from IPython.display import HTML
import cv2

def wrap_monitor(env):
    """ Gymの環境をmp4に保存するために，環境をラップする関数． """
    pass
    # return gym.wrappers.Monitor(env, '/tmp/monitor', video_callable=lambda x: True, force=True)

def play_mp4():
    pass
    # """ 保存したmp4をHTMLに埋め込み再生する関数． """
    # path = glob.glob(os.path.join('/tmp/monitor', '*.mp4'))[0]
    # mp4 = open(path, 'rb').read()
    # url = "data:video/mp4;base64," + b64encode(mp4).decode()
    # return HTML("""<video width=400 controls><source src="%s" type="video/mp4"></video>""" % url)

class Trainer:

    def __init__(self, env, env_test, algo, seed=0, num_steps=10**6, eval_interval=10**4, num_eval_episodes=3, is_GC=False):

        self.env = env
        self.env_test = env_test
        self.algo = algo

        # 環境の乱数シードを設定する．
        self.env.seed(seed)
        self.env_test.seed(2**31-seed)

        # 平均収益を保存するための辞書．
        self.returns = {'step': [], 'return': []}

        # データ収集を行うステップ数．
        self.num_steps = num_steps
        # 評価の間のステップ数(インターバル)．
        self.eval_interval = eval_interval
        # 評価を行うエピソード数．
        self.num_eval_episodes = num_eval_episodes

        self.is_GC = is_GC

    def train(self):
        """ num_stepsステップの間，データ収集・学習・評価を繰り返す． """

        # 学習開始の時間
        self.start_time = time()
        # エピソードのステップ数．
        t = 0

        # 環境を初期化する．
        obs_all = self.env.reset()

        for steps in range(1, self.num_steps + 1):
            # 環境(self.env)，現在の状態(state)，現在のエピソードのステップ数(t)，今までのトータルのステップ数(steps)を
            # アルゴリズムに渡し，状態・エピソードのステップ数を更新する．

            if self.is_GC:
                goal = self.env.sim.tgt_pos
                obs_all, t = self.algo.step(self.env, obs_all, goal, t, steps)
            else:
                obs_all, t = self.algo.step(self.env, obs_all, t, steps)

            # アルゴリズムが準備できていれば，1回学習を行う．
            if self.algo.is_update(steps):
                self.algo.update()

            # 一定のインターバルで評価する．
            if steps % self.eval_interval == 0:
                self.evaluate(steps)

    def evaluate(self, steps):
        """ 複数エピソード環境を動かし，平均収益を記録する． """

        returns = []
        for _ in range(self.num_eval_episodes):
            # obs_all = self.env_test.reset()
            obs_all = self.env_test.test_reset()
            goal = self.env_test.sim.tgt_pos
            done = False
            episode_return = 0.0

            state, velocity obs = obs_all

            if self.algo.name == 'TDM':
                for _ in range(self.env_test._max_episode_steps):
                    action = self.algo.exploit(obs_all, goal, self.env_test.get_left_steps())
                    state, reward, done, _ = self.env_test.step(action)
                    episode_return += reward
                    # episode_return += sum(reward)
                    if done:
                        break
            elif self.is_GC:
                for _ in range(self.env_test._max_episode_steps):
                    action = self.algo.exploit(obs_all, goal)
                    state, reward, done, _ = self.env_test.step(action)
                    episode_return += reward
                    if done:
                        break
            else:
                for _ in range(self.env_test._max_episode_steps):
                    action = self.algo.exploit(obs_all)
                    state, reward, done, _ = self.env_test.step(action)
                    episode_return += reward
                    if done:
                        break

            returns.append(episode_return)

        mean_return = np.mean(returns)
        self.returns['step'].append(steps)
        self.returns['return'].append(mean_return)

        print(f'Num steps: {steps:<6}   '
              f'Return: {mean_return:<5.1f}   '
              f'Final state: {state}   '
              f'Time: {self.time}')

    def visualize(self):
        """ 1エピソード環境を動かし，mp4を再生する． """
        env = wrap_monitor(gym.make(self.env.unwrapped.spec.id))
        obs_all = env.reset()
        done = False

        if self.is_GC:
            goal = self.env.sim.tgt_pos
            while (not done):
                action = self.algo.exploit(obs_all, goal)
                obs_all, _, done, _ = env.step(action)
        else:
            while (not done):
                action = self.algo.exploit(obs_all)
                obs_all, _, done, _ = env.step(action)

        del env
        return play_mp4()

    def plot(self, s=""):
        """ 平均収益のグラフを描画する． """
        fig = plt.figure(figsize=(8, 6))
        plt.plot(self.returns['step'], self.returns['return'])
        plt.xlabel('Steps', fontsize=24)
        plt.ylabel('Return', fontsize=24)
        plt.tick_params(labelsize=18)
        # plt.title(f'{self.env.unwrapped.spec.id}', fontsize=24)
        plt.title('return', fontsize=24)
        plt.tight_layout()
        fig.savefig("log/"+self.algo.name+s+".png")

    @property
    def time(self):
        """ 学習開始からの経過時間． """
        return str(timedelta(seconds=int(time() - self.start_time)))

    def saveVideo(self, path="./", s=""):   
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  
        video = cv2.VideoWriter(path + self.algo.name + s + ".mp4", fourcc, 10, (800,800))  # 動画の仕様（ファイル名、fourcc, FPS, サイズ）

        done = False
        obs_all = self.env.test_reset()
        goal = self.env.sim.tgt_pos

        video.write(self.env.render())

        for _ in range(self.env._max_episode_steps):
            if self.algo.name == 'TDM':
                action = self.algo.exploit(obs_all, goal, self.env_test.get_left_steps())
            elif self.is_GC:
                action = self.algo.exploit(obs_all, goal)
            else:
                action = self.algo.exploit(obs_all)
            obs_all, _, done, _ = self.env.step(action)
            video.write(self.env.render())
            if done:
                break

    def saveVideo_subgoals(self, subgoals, path="./", s=""):   
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  
        video = cv2.VideoWriter(path + self.algo.name + s + ".mp4", fourcc, 10, (800,800))  # 動画の仕様（ファイル名、fourcc, FPS, サイズ）

        done = False
        obs_all = self.env.test_reset()
        # goal = self.env.sim.tgt_pos

        # video.write(self.env.render())
        video.write(self.env.sim.render(subgoals[0]))

        i = 0
        goal = subgoals[i]

        for _ in range(self.env._max_episode_steps):

            if self.algo.name == 'TDM':
                action = self.algo.exploit(obs_all, goal, self.env_test.get_left_steps())
            elif self.is_GC:
                action = self.algo.exploit(obs_all, goal)
            else:
                action = self.algo.exploit(obs_all)
            obs_all, _, done, _ = self.env.step(action)
            # video.write(self.env.render())
            video.write(self.env.sim.render(goal))
            
            if i < len(subgoals):
                if self.env.sim.isArrive(goal, obs_all):
                    i += 1
                    if len(subgoals) == i:
                        goal = self.env.sim.tgt_pos
                    else:
                        goal = subgoals[i]

            if done:
                break

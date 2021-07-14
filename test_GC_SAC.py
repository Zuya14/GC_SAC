import gym
import pybullet_envs
# from PPO import PPO
from GC_SAC import GC_SAC
from trainer import Trainer

# from mazeEnv import mazeEnv 
# from crossEnv import crossEnv 
from square3Env import square3Env 
# from maze3Env import maze3Env 

# ENV_ID = 'InvertedPendulumBulletEnv-v0'
SEED = 0
NUM_STEPS = 5 * 10 ** 4
# NUM_STEPS = 10 * 10 ** 4
# NUM_STEPS = 2 * 10 ** 5
# NUM_STEPS = 25 * 10 ** 4
EVAL_INTERVAL = 10 ** 3

# env = gym.make(ENV_ID)
# env_test = gym.make(ENV_ID)

env_id = 0

if env_id == 0:
    env = square3Env()
    env_test = square3Env()
elif env_id ==1:
    env = maze3Env()
    env_test = maze3Env()

env.setting()
env_test.setting()

algo = GC_SAC(
    state_size=env.state_space.shape,
    action_size=env.action_space.shape,
    velocity_size=env.velocity_space.shape,
    observation_size=env.observation_space.shape,
    goal_size = env.state_space.shape,
    epsilon_decay = NUM_STEPS,
    start_steps=1000
)


trainer = Trainer(
    env=env,
    env_test=env_test,
    algo=algo,
    num_steps=NUM_STEPS,
    eval_interval=EVAL_INTERVAL,
    is_GC=True
)

algo.load()

trainer.saveVideo()

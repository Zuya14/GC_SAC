import sys
import os
import glob
import gym
import pybullet_envs
# from PPO import PPO
from GC_SAC import GC_SAC
from trainer import Trainer

from square3Env import square3Env 
from maze3Env import maze3Env 


if __name__ == '__main__':

    assert len(sys.argv) == 3

    arg = sys.argv[2]

    if os.path.isdir(arg):
        if 'Env' in os.path.basename(os.path.dirname(arg)):
            dirs = sorted(["{:s}/{:s}/".format(arg, d) for d in os.listdir(arg)])
        else:
            dirs = [arg]
    else:
        print(arg + " is not dir!")
        exit()

    # print(dirs)

    # NUM_STEPS = 5 * 10 ** 4
    NUM_STEPS = 1 * 10 ** 5
    EVAL_INTERVAL = 10 ** 3

    env_id = int(sys.argv[1])
    # path = sys.argv[2]

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

    for d in dirs:
        if glob.glob(d + '/*.mp4'):
            # print("tested :" + d)
            pass
        else:
            print("test   :" + d)

            algo.load(path=d)

            trainer.saveVideo(path=d)

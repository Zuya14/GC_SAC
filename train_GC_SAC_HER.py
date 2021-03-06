import sys
import os
import datetime
import gym
import pybullet_envs
# from PPO import PPO
from GC_SAC_HER import GC_SAC_HER
from trainer import Trainer

from square3Env import square3Env 
from maze3Env import maze3Env 


if __name__ == '__main__':
    
    if not ( len(sys.argv) == 2 or len(sys.argv) == 3 ):
        print()
        print("!!!")
        print("\tinput args <env_id> <weight_path(option)>")
        print("!!!")
        exit()

    start_time = datetime.datetime.now()

    # NUM_STEPS = 5 * 10 ** 4
    NUM_STEPS = 1 * 10 ** 5
    EVAL_INTERVAL = 10 ** 3

    env_id = int(sys.argv[1])

    if env_id == 0:
        env = square3Env()
        env_test = square3Env()
    elif env_id ==1:
        env = maze3Env()
        env_test = maze3Env()

    env.setting()
    env_test.setting()

    algo = GC_SAC_HER(
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

    start_time = datetime.datetime.now()
    now = '{0:%Y%m%d}-{0:%H%M}'.format(start_time)
    path = "log/{:s}/{:s}/{:s}/".format(algo.name, env.name, now)

    if not os.path.exists(path):
        os.makedirs(path)
    print("dirs:" + path)

    with open(path + '/log.txt', 'w') as f:

        if len(sys.argv) == 3:
            load_path = sys.argv[2]
            algo.load(path=load_path)
            f.write('[LOAD PATH]\n')
            f.write('{:s}\n'.format(load_path))

        
        f.write('[STEPS]\n')
        f.write('NUM_STEPS:{:f}\n'.format(NUM_STEPS))
        f.write('EVAL_INTERVAL:{:f}\n'.format(EVAL_INTERVAL))

        trainer.train()

        trainer.plot(path=path)

        algo.save(path=path)

        end_time = datetime.datetime.now()

        f.write('[PARAMS]\n')
        params = env.getParams()
        for k, v in params.items():
            f.write('{:s}:{:f}\n'.format(k, v))

        f.write('[Env PARAMS]\n')
        params = env.getEnvParams()
        for k, v in params.items():
            f.write('{:s}:{:f}\n'.format(k, v))

        f.write('[TIME]\n')
        f.write('START:{:s}\n'.format(str(start_time)))
        f.write('END:{:s}\n'.format(str(end_time)))
        f.write('TOTAL:{:s}\n'.format(str(end_time - start_time)))

    # trainer.visualize()

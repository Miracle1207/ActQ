from arguments import get_args
from ppo_agent import ppo_agent
from b_agent import b_agent
from actq_agent import actq_agent
from rl_utils.env_wrapper.create_env import create_multiple_envs, create_single_env
from rl_utils.env_wrapper.mujoco_wrapper import discrete_mujoco_wrapper
from rl_utils.seeds.seeds import set_seeds
import gym
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
if __name__ == '__main__':
    # set signle thread
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    # get arguments
    args = get_args()
    set_seeds(args)
    # start to create the environment
    if args.env_type == 'atari':
        envs = create_multiple_envs(args)
    elif args.env_type == 'mujoco':
        envs = create_single_env(args)
        # envs = discrete_mujoco_wrapper(envs, args.k, args.act_type)
    else:
        raise NotImplementedError
    # create trainer
    if args.dist == "actQ":
        trainer = b_agent(envs, args)
    elif args.dist == "actq":
        trainer = actq_agent(envs, args)
    else:
        trainer = ppo_agent(envs, args)
    # start to learn
    trainer.learn()
    # close the environment
    envs.close()


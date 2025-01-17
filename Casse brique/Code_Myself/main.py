from __future__ import print_function
import os
import torch
import torch.multiprocessing as mp
from envs import create_atari_env
from model import ActorCritic
from train import train
from test_file import test
import my_optim

# Gathering all the parameters (that we can modify to explore)
class Params():
    def __init__(self):
        self.lr = 0.0001
        self.gamma = 0.98
        self.tau = 1.
        self.seed = 1
        self.entropy_coef = 0.01
        self.num_processes = 16
        self.num_steps = 100
        self.max_episode_length = 10000
        self.env_name = 'ALE/Breakout-v5'

# Main run function inside the '__main__' block
if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    os.environ['OMP_NUM_THREADS'] = '8'
    params = Params()
    torch.manual_seed(params.seed)
    
    # Create the environment and model
    print("\033[33mCreating env ...\033[0m")
    env = create_atari_env(params.env_name)
    print("\033[33mEnv created\033[0m")
    
    shared_model = ActorCritic(env.observation_space.shape[0], env.action_space)
    shared_model.share_memory()
    
    # Setup the optimizer
    optimizer = my_optim.SharedAdam(shared_model.parameters(), lr=params.lr)
    optimizer.share_memory()

    processes = []

    # Start the test process
    print("\033[33mStart the test process ...\033[0m")
    p = mp.Process(target=test, args=(params.num_processes, params, shared_model))
    p.start()
    print("\033[33mTest process started ...\033[0m")
    processes.append(p)
    

    # Start the training processes
    print("\033[33mStart the training processes : \033[0m")
    for rank in range(0, params.num_processes):
        p = mp.Process(target=train, args=(rank, params, shared_model, optimizer))
        p.start()
        processes.append(p)

    # Wait for all processes to finish
    print("\033[33mWait for all processes to finish : \033[0m")
    for p in processes:
        p.join()
        
from simulator import Manufacturing_Simulator
from utils import PPO
import argparse

hyperparameters = {
        'timesteps_per_batch': 2048,
        'max_timesteps_per_episode': 200,
        'gamma': 0.99,
        'n_updates_per_iteration': 10,
        'lr': 3e-4,
        'clip': 0.2
    }

    
if __name__ == "__main__" :
  parser = argparse.ArgumentParser()
  parser.add_argument("-a", "--agents", help="Enter your title",default= 4)
  parser.add_argument("-d", "--decay_factor" , help="Enter your title",default= 0.05)
  parser.add_argument("-t", "--total_timesteps" , help="Enter your title",default= 2048)
  args = parser.parse_args()
  ppo = PPO(agents = int(args.agents), hyperparameters = hyperparameters)
  env = Manufacturing_Simulator(agents = int(args.agents),PPO = ppo, decay_factor = float(args.decay_factor))
  env.learn(total_timesteps = int(args.total_timesteps)) 

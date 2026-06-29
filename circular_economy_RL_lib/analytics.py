import logging
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os

def save_results(args,logger_dict):
  n_agents  = len(logger_dict[()]['avg_actor_losses'])
  try:
    os.makedirs(args.save_path)
  except:
    pass
  new_path = os.path.join(args.save_path,args.parameter)
  try:
    os.makedirs(new_path)
  except:
    pass
  for i in range(1,n_agents+1):
    print(flush=True)
    print(f"-------------------- Agent #{i} --------------------", flush=True)
    print(f"For the agent{i}, plot of " + args.parameter + " vs number of iterations is given as : ",flush  = True)
    print(flush=True)
    fig = plt.figure()
    ax = fig.gca()
    plt.plot(logger_dict[()][args.parameter][f'agent{i}'])
    plt.grid()
    agent_path = os.path.join(new_path,f"agent{i}_"+args.parameter+".png")
    plt.savefig(agent_path, bbox_inches='tight')


if __name__ == "__main__" :
  parser = argparse.ArgumentParser()
  parser.add_argument("-i", "--iterations", help="Enter your title",default= 1)
  parser.add_argument("-p","--parameter", help="Select one of these keys : 'avg_batch_rews', 'avg_actor_losses' ",default= "avg_actor_losses")
  parser.add_argument("-s","--save_path", help="Select one of these keys : 'avg_batch_rews', 'avg_actor_losses' ",default= "/content/save_results/")
  args = parser.parse_args()
  logger_dict = np.load(f"/content/drive/MyDrive/manufacturing_optimization_results/ppo_results_{args.iterations}.npy",allow_pickle=True)
  save_results(args,logger_dict)

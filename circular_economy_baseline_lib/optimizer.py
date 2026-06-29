import torch
import os
import numpy as np
from simulator import Manufacturing_Simulator
from config import config, SELLER, BUYER, TRANSFORM, stages

from torch.utils.tensorboard import SummaryWriter
from utils import get_baseline_result_folder, seedEverything
from logging import getLogger

from opt_solver import XpressSolver

class Optimizer:

    def __init__(self):
        for key, value in config.items():
            # print(key, value)
            setattr(self, key, value)

        if self.seed != None:
            # Check if our seed is valid first
            assert(type(self.seed) == int)
            # Set the seed
            seedEverything(self.seed)
            print(f"Successfully set seed to {self.seed}")

        self.env = Manufacturing_Simulator()
        self.opt = XpressSolver(self.num_agents, self.num_commodities)

        self.logger = getLogger(name='optimizer')
        log_folder = get_baseline_result_folder(self.env.phi) + '/log'
        global debug_folder
        debug_folder = get_baseline_result_folder(self.env.phi) + '/debug'
        os.makedirs(log_folder, exist_ok=True)
        os.makedirs(debug_folder, exist_ok=True)
        self.writer = SummaryWriter(log_folder)

    def rollout(self):

        batch_obs = [[] for _ in range(len(stages))]            # batch observations. 
        batch_acts = [[] for _ in range(len(stages))]           # batch actions
        batch_rews = [[] for _ in range(len(stages))]           # batch rewards
        batch_rtgs = [[] for _ in range(len(stages))]           # batch rewards-to-go
        batch_objs = [[] for _ in range(len(stages))]           # batch objectives
        batch_lens = [] # episodic lengths in batch

        t = 0 # Keeps track of how many timesteps we've run so far this batch

        while t < self.num_steps: # self.num_steps # 10000
            ep_rews = [[] for _ in range(len(stages))]
            ep_objs = [[] for _ in range(len(stages))]

            obs_s = self.env.reset()
            # Shape: seller observations - (n_agents, seller_state_size)
            done = False
            for ep_t in range(self.episode_length): # 500
                t += 1
                #==================Collect seller data==================
                # Collect seller observation
                # Append the seller_obs to the proper slot
                batch_obs[SELLER].append(obs_s)

                # Get seller action
                seller_params = self.env.get_seller_params()
                obj_s, action_s = self.opt.step1(seller_params)
                obs_b = self.env.step_sell(obs_s, action_s)

                ep_objs[SELLER].append(obj_s)
                batch_acts[SELLER].append(action_s)

                #==================Collect buyer data==================
                # Collect buyer observation
                batch_obs[BUYER].append(obs_b)

                # Get buyer action
                buyer_params = self.env.get_buyer_param()
                obj_b, action_b = self.opt.step2(buyer_params)
                # Shape: (num_buyers, buyer_action_size)
                # print("5", action_b)

                # Send buyer action and get transformation observation
                obs_t, rew_b, rew_s = self.env.step_buy(obs_b, action_b)

                # Collect buyer reward, action, and log prob
                ep_rews[SELLER].append(rew_s)
                ep_rews[BUYER].append(rew_b)
                ep_objs[BUYER].append(obj_b)
                batch_acts[BUYER].append(action_b)

                #==================Collect transform data==================
                # Collect transformtion observation
                batch_obs[TRANSFORM].append(obs_t)
                trans_params = self.env.get_trans_param()
                # Get transformation action
                obj_t, action_t= self.opt.step3(trans_params)
                # Shape: (num_transformers, transformer_action_size)
                # print("6", action_t)

                # Send transformation action and get seller observation
                obs_s, rew_t, done_t = self.env.step_trans(obs_t, action_t)

                # Collect transform reward, action, and log prob
                ep_rews[TRANSFORM].append(rew_t)
                ep_objs[TRANSFORM].append(obj_t)
                batch_acts[TRANSFORM].append(action_t)

                # Collect episodic length and rewards
            batch_lens.append(ep_t + 1)  # plus 1 because timestep starts at 0
            for stage in stages:
                batch_rews[stage].append(ep_rews[stage])
                batch_objs[stage].append(ep_objs[stage])

        # ALG STEP #4
        batch_rtgs, batch_rets = self.compute_rtgs(batch_rews)

        # Return the batch data
        return batch_obs, batch_acts, batch_rtgs, batch_rets, batch_objs, batch_lens

    def compute_rtgs(self,batch_rews):
        """
        Calculate the Reward-To-Go of each timestep in a batch given the rewards
        """
        batch_rtgs = [[] for _ in stages] 
        batch_rets = [[] for _ in stages]
        for stage in stages:
            # print(np.array(batch_rews[stage]).shape) # (20, 500, 3)
            batch_rtgs[stage], batch_rets[stage] = self._compute_rtgs(batch_rews[stage])

        return batch_rtgs, batch_rets

    def _compute_rtgs(self, batch_rews):

        batch_rtgs = []
        cumu_rews = []

        for ep_rews in reversed(batch_rews):
            s  = []
            for i in range(self.num_agents):
                s.append(0.0)
            discounted_reward = np.array(s).reshape(batch_rews[0][0].shape) # The discounted reward so far
            ep_rtgs = []
            
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                ep_rtgs.insert(0, discounted_reward)
            batch_rtgs.append(ep_rtgs)
            # print(ep_rtgs[0])
            cumu_rews.append(np.array(ep_rtgs[0]))

        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float).reshape(-1, self.num_agents)
        return batch_rtgs, cumu_rews

    def run(self):

        total_timesteps = self.num_steps * self.num_epochs
        t_so_far = 0  # Timesteps simulated so far
        i_so_far = 0  # Batches simulated so far

        while t_so_far < total_timesteps:
            self.logger.info('=================================================================')

            batch_obs, batch_acts, batch_rtgs, batch_rets, batch_objs, batch_lens = self.rollout()

            t_so_far += np.sum(batch_lens)

            i_so_far += 1

            curr_epoch_results_dict = {}

            for ag in range(self.num_agents):
                for stage in stages:
                    for i in range(self.num_steps // self.episode_length):
                        self.writer.add_scalar('return_stage_{}_agent_{}'.format(stage, ag), batch_rets[stage][i][ag],
                                               i)
                        print('return_stage_{}_agent_{}_iter_{}:'.format(stage, ag, i), batch_rets[stage][i][ag])

            self.logger.info("Epoch {:3d}/{:3d}]".format(i_so_far, self.num_epochs))

            curr_epoch_results_dict['actual_d'] = self.env.actual_d
            curr_epoch_results_dict['spot_q'] = self.env.spot_q
            curr_epoch_results_dict['price'] = self.env.price
            curr_epoch_results_dict['spot_price'] = self.env.spot_price
            curr_epoch_results_dict['inv'] = self.env.inv
            curr_epoch_results_dict['rewards'] = np.sum(np.array(batch_rets), axis = (0,1))
            curr_epoch_results_dict['u_eco'] = self.env.eco_u
            curr_epoch_results_dict['u_tx'] = self.env.tx_u
            curr_epoch_results_dict['wastewater'] = self.env.wastewater


            np.save(str(debug_folder) + "/epoch={}_results.npy".format(i_so_far), curr_epoch_results_dict)

            #import pdb; pdb.set_trace()

        self.logger.info(" *** Done *** ")

if __name__ == "__main__":
    opt = Optimizer()
    opt.run()
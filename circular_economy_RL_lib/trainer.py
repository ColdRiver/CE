import torch
import time, os
import numpy as np
from simulator import Manufacturing_Simulator
from agent import AgentPool
from config import config, SELLER, BUYER, TRANSFORM, stages
from logging import getLogger
from utils import AverageMeter

from torch.utils.tensorboard import SummaryWriter
from utils import get_result_folder

class Trainer:
    """
    The trainer class for the manufacturing problem

    Attributes:
    num_agents - the number of agents (institutions) in the problem
    num_commodities - number of commodities in the problem
    episode_length - the maximum timesteps for every episode

    Functoins:
    rollout - collect trajectories for training
    learn - train the agents
    """
    def __init__(self):
        for key, value in config.items():
            # print(key, value)
            setattr(self, key, value)

        self.env = Manufacturing_Simulator()
        self.agent_pool = AgentPool(self.num_agents, self.num_commodities, self.history_length)
        if self.seed != None:
            # Check if our seed is valid first
            assert(type(self.seed) == int)
            # Set the seed
            torch.manual_seed(self.seed)
            print(f"Successfully set seed to {self.seed}")

        self.logger = getLogger(name='trainer')
        log_folder = get_result_folder() + '/log'
        global debug_folder
        debug_folder = get_result_folder() + '/debug'
        os.makedirs(log_folder, exist_ok=True)
        os.makedirs(debug_folder, exist_ok=True)
        self.writer = SummaryWriter(log_folder)
        # self.logger = {
        #     'delta_t': time.time_ns(),
        #     't_so_far': 0,   # timesteps so far
        #     'i_so_far': 0,   # iterations so far
        #     'batch_lens': [],# episodic lengths in batch
        #     'batch_rews': [],# episodic returns in batch
        #     'actor_losses': [[] for i in range(self.num_agents)],     # losses of actor network in current iteration
        #     "avg_batch_rews": [[] for i in range(self.num_agents)] ,    # avg episodic returns in batch
        #     "avg_actor_losses": [[] for i in range(self.num_agents)]    # avg losses of actor network in current iteration
        # }

    ## The three types agents make decisions in the simulator together and get their own observations
    ## Need to restore the information separately
    def rollout(self):
        # Let the following information be a list of three elements and each element can be the tensors for
        # seller, buyer, and transformation
        batch_obs = [[] for _ in range(len(stages))]            # batch observations. 
        batch_log_probs = [[] for _ in range(len(stages))]     # log probs of each action
        batch_acts = [[] for _ in range(len(stages))]           # batch actions
        batch_rews = [[] for _ in range(len(stages))]           # batch rewards
        batch_rtgs = [[] for _ in range(len(stages))]           # batch rewards-to-go
        batch_lens = []           # episodic lengths in batch

        t = 0 # Keeps track of how many timesteps we've run so far this batch+

        while t < self.num_steps: # self.num_steps # 10000
            # Episodic data. Keeps track of rewards per episode, will get cleared
            # upon each new episode
            ep_rews = [[] for _ in range(len(stages))]
            obs_s = self.env.reset()
            # Shape: seller observations - (n_agents, seller_state_size)
            done = False
            # print(self.episode_length)
            for ep_t in range(self.episode_length): # 500
                t += 1
                #==================Collect seller data==================
                # Collect seller observation
                # Append the seller_obs to the proper slot
                batch_obs[SELLER].append(obs_s)

                # Get seller action
                action_s, log_prob_s = self.agent_pool.get_actions(obs_s, SELLER)
                # Shape: (num_sellers, seller_action_size)
                # print("4", action_s)

                # Send seller action and get buyer observation
                # obs_b, rew_s = self.env.step_sell(obs_s, action_s)
                obs_b = self.env.step_sell(obs_s, action_s)
                # Collect seller reward, action, and log prob
                # ep_rews[SELLER].append(rew_s)
                batch_acts[SELLER].append(action_s)
                batch_log_probs[SELLER].append(log_prob_s)

                #==================Collect buyer data==================
                # Collect buyer observation
                batch_obs[BUYER].append(obs_b)

                # Get buyer action
                action_b, log_prob_b = self.agent_pool.get_actions(obs_b, BUYER)
                # Shape: (num_buyers, buyer_action_size)
                # print("5", action_b)

                # Send buyer action and get transformation observation
                obs_t, rew_b, rew_s = self.env.step_buy(obs_b, action_b)

                # Collect buyer reward, action, and log prob
                ep_rews[SELLER].append(rew_s)
                ep_rews[BUYER].append(rew_b)
                batch_acts[BUYER].append(action_b)
                batch_log_probs[BUYER].append(log_prob_b)

                #==================Collect transform data==================
                # Collect transformtion observation
                batch_obs[TRANSFORM].append(obs_t)

                # Get transformation action
                action_t, log_prob_t = self.agent_pool.get_actions(obs_t, TRANSFORM)
                #import pdb; pdb.set_trace()
                # Shape: (num_transformers, transformer_action_size)
                # print("6", action_t)
                if np.isnan(action_t[0][0]):
                    raise NotImplementedError 

                # Send transformation action and get seller observation
                obs_s, rew_t, done_t = self.env.step_trans(obs_t, action_t)

                # Collect transform reward, action, and log prob
                ep_rews[TRANSFORM].append(rew_t)
                batch_acts[TRANSFORM].append(action_t)
                batch_log_probs[TRANSFORM].append(log_prob_t)

                ## TODO-Check early termination condition
                #if done:
                #    break

                # Collect episodic length and rewards
            batch_lens.append(ep_t + 1) # plus 1 because timestep starts at 0
            for stage in stages:
                batch_rews[stage].append(ep_rews[stage])


        # Reshape data as tensors in the shape specified before returning
        # batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        # batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        # batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
        
        batch_obs = [torch.tensor(obs, dtype=torch.float) for obs in batch_obs]
        batch_acts = [torch.tensor(act, dtype=torch.float) for act in batch_acts]
        batch_log_probs = [torch.tensor(log_p, dtype=torch.float) for log_p in batch_log_probs]

        # print(batch_obs[0].shape, batch_acts[0].shape, batch_log_probs[0].shape) # torch.Size([500, 3, 1414]) torch.Size([500, 3, 28]) torch.Size([500, 3])
        # raise NotImplementedError

        # ALG STEP #4
        batch_rtgs, batch_rets = self.compute_rtgs(batch_rews)
        # self.logger['batch_rews'] = batch_rews
        # self.logger['batch_lens'] = batch_lens

        # Return the batch data
        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_rets, batch_lens

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

    def _compute_rtgs(self,batch_rews):

        batch_rtgs = []

        batch_shape = len(batch_rews[0])*len(batch_rews) # len(ep_rew)*num_episodes
        # Iterate through each episode backwards to maintain same order in batch_rtgs


        for ep_rews in reversed(batch_rews):
            s  = []
            for i in range(self.num_agents):
                s.append(0.0)
            discounted_reward = np.array(s).reshape(batch_rews[0][0].shape) # The discounted reward so far
            ep_rtgs = []
            cumu_ret = 0
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward *self.gamma
                ep_rtgs.insert(0, discounted_reward)
            batch_rtgs.append(ep_rtgs)
            # print(ep_rtgs[0])
            cumu_ret += np.array(ep_rtgs[0])

        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float).reshape(batch_shape,self.num_agents)
        return batch_rtgs, cumu_ret / float(len(batch_rews))

    def learn(self):

        #print(f"Learning... Running {self.episode_length} timesteps per episode, ", end='')
        #print(f"{self.num_steps} timesteps per batch for a total of {total_timesteps} timesteps")
        total_timesteps = self.num_steps*self.num_epochs
        t_so_far = 0 # Timesteps simulated so far
        i_so_far = 0 # Batches simulated so far

        while t_so_far < total_timesteps:
            self.logger.info('=================================================================')
            score_AM = AverageMeter()
            loss_AM = AverageMeter()

            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_rets, batch_lens = self.rollout()

            t_so_far += np.sum(batch_lens)

            i_so_far += 1

            curr_epoch_results_dict = {}

            #self.logger['t_so_far'] = t_so_far
            #self.logger['i_so_far'] = i_so_far

            for ag in range(self.num_agents):
                for stage in stages:
                    a_loss, c_loss = self.agent_pool.learn(batch_obs, batch_acts, batch_log_probs, batch_rtgs,\
                            stage, ag, self.n_updates_per_iteration)
                    # print(a_loss, c_loss)
                    self.writer.add_scalar('actor_loss_stage_{}_agent_{}'.format(stage, ag), a_loss, t_so_far)
                    self.writer.add_scalar('critic_loss_stage_{}_agent_{}'.format(stage, ag), c_loss, t_so_far)
                    self.writer.add_scalar('return_stage_{}_agent_{}'.format(stage, ag), batch_rets[stage][ag], t_so_far)
                if i_so_far % self.save_freq == 0:
                    self.logger.info("Saving trained_model")
                    self.agent_pool.save_model(stage, ag, f'ppo_actor_agent{ag+1}_{i_so_far}.pth')

            self.logger.info("Epoch {:3d}/{:3d}]".format(i_so_far, self.num_epochs))

            #dumping result nd arrays
            curr_epoch_results_dict['actual_d'] = self.env.actual_d
            curr_epoch_results_dict['spot_q'] = self.env.spot_q
            curr_epoch_results_dict['price'] = self.env.price
            curr_epoch_results_dict['spot_price'] = self.env.spot_price
            curr_epoch_results_dict['inv'] = self.env.inv
            curr_epoch_results_dict['rewards'] = np.sum(np.array(batch_rets), axis=(0))
            curr_epoch_results_dict['u_eco'] = self.env.eco_u
            curr_epoch_results_dict['u_tx'] = self.env.tx_u
            curr_epoch_results_dict['wastewater'] = self.env.wastewater

            #import pdb; pdb.set_trace()

            np.save(str(debug_folder) + "/epoch={}_results.npy".format(i_so_far), curr_epoch_results_dict)

            #import pdb; pdb.set_trace()


            #import pdb; pdb.set_trace()





        self.logger.info(" *** Training Done *** ")

    ## TODO-Revise
    # def _log_summary(self,ag):

    #     delta_t = self.logger['delta_t']
    #     self.logger['delta_t'] = time.time_ns()
    #     delta_t = (self.logger['delta_t'] - delta_t) / 1e9
    #     delta_t = str(round(delta_t, 2))

    #     t_so_far = self.logger['t_so_far']
    #     i_so_far = self.logger['i_so_far']
    #     avg_ep_lens = np.mean(self.logger['batch_lens'])
    #     ## TODO-Revise
    #     #avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in np.array(self.logger['batch_rews'])[:,:,ag,:]])
    #     #avg_actor_loss = np.mean([losses.float().mean() for losses in self.logger['actor_losses'][f'agent{ag+1}']])
    #     #self.logger['avg_batch_rews'][f'agent{ag+1}'].append(avg_ep_rews)
    #     #self.logger['avg_actor_losses'][f'agent{ag+1}'].append(avg_actor_loss)
    #     avg_ep_lens = str(round(avg_ep_lens, 2))
    #     #avg_ep_rews = str(round(avg_ep_rews, 2))
    #     #avg_actor_loss = str(round(avg_actor_loss, 5))

    #     print(flush=True)
    #     print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
    #     print(f"Displaying the stats for the agent: {ag+1}", flush = True)
    #     print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
    #     # print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
    #     # print(f"Average Loss: {avg_actor_loss}", flush=True)
    #     print(f"Timesteps So Far: {t_so_far}", flush=True)
    #     print(f"Iteration took: {delta_t} secs", flush=True)
    #     print(f"------------------------------------------------------", flush=True)
    #     print(flush=True)
    #     ## TODO-Revise
    #     #self.logger['actor_losses'][f'agent{ag+1}'] = []

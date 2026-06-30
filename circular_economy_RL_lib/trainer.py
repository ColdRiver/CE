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
    The trainer class for the manufacturing problem (Flat MARL Baseline)
    """
    def __init__(self):
        for key, value in config.items():
            setattr(self, key, value)

        self.env = Manufacturing_Simulator()
        self.agent_pool = AgentPool(self.num_agents, self.num_commodities, self.history_length)
        if self.seed != None:
            assert(type(self.seed) == int)
            torch.manual_seed(self.seed)
            print(f"Successfully set seed to {self.seed}")

        self.logger = getLogger(name='trainer')
        self.result_folder = './result/flat_ppo'
        log_folder = self.result_folder + '/log'
        global debug_folder
        debug_folder = self.result_folder + '/debug'
        os.makedirs(log_folder, exist_ok=True)
        os.makedirs(debug_folder, exist_ok=True)
        self.writer = SummaryWriter(log_folder)

    def rollout(self):
        batch_obs = [[] for _ in range(len(stages))]            
        batch_log_probs = [[] for _ in range(len(stages))]     
        batch_acts = [[] for _ in range(len(stages))]           
        batch_rews = [[] for _ in range(len(stages))]           
        batch_rtgs = [[] for _ in range(len(stages))]           
        batch_leader_returns = []  # Tracks unpenalized leader returns offline
        batch_lens = []

        t = 0 
        while t < self.num_steps:
            ep_rews = [[] for _ in range(len(stages))]
            ep_leader_rews = []
            obs_s = self.env.reset()
            done = False
            for ep_t in range(self.episode_length): 
                t += 1
                #==================Collect seller data==================
                batch_obs[SELLER].append(obs_s)
                action_s, log_prob_s = self.agent_pool.get_actions(obs_s, SELLER)
                obs_b = self.env.step_sell(obs_s, action_s)
                batch_acts[SELLER].append(action_s)
                batch_log_probs[SELLER].append(log_prob_s)

                #==================Collect buyer data==================
                batch_obs[BUYER].append(obs_b)
                action_b, log_prob_b = self.agent_pool.get_actions(obs_b, BUYER)
                obs_t, rew_b, rew_s = self.env.step_buy(obs_b, action_b)

                ep_rews[SELLER].append(rew_s)
                ep_rews[BUYER].append(rew_b)
                batch_acts[BUYER].append(action_b)
                batch_log_probs[BUYER].append(log_prob_b)

                #==================Collect transform data==================
                batch_obs[TRANSFORM].append(obs_t)
                action_t, log_prob_t = self.agent_pool.get_actions(obs_t, TRANSFORM)
                if np.isnan(action_t[0][0]):
                    raise NotImplementedError 

                # Send transformation action and unpack 5 elements from the corrected simulator
                s_leader_next, s_follower_next, rew_t, rew_l, done_t = self.env.step_trans(obs_t, action_t)
                obs_s = s_follower_next

                ep_rews[TRANSFORM].append(rew_t)
                ep_leader_rews.append(rew_l)  # Track leader reward offline
                batch_acts[TRANSFORM].append(action_t)
                batch_log_probs[TRANSFORM].append(log_prob_t)

            batch_lens.append(ep_t + 1)
            for stage in stages:
                batch_rews[stage].append(ep_rews[stage])
                
            # Compute unpenalized leader return for this episode
            discounted_reward = 0.0
            for rew in reversed(ep_leader_rews):
                discounted_reward = rew + discounted_reward * self.gamma
            batch_leader_returns.append(discounted_reward)

        batch_obs = [torch.tensor(obs, dtype=torch.float) for obs in batch_obs]
        batch_acts = [torch.tensor(act, dtype=torch.float) for act in batch_acts]
        batch_log_probs = [torch.tensor(log_p, dtype=torch.float) for log_p in batch_log_probs]

        batch_rtgs, batch_rets = self.compute_rtgs(batch_rews)
        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_rets, batch_lens, batch_leader_returns

    def compute_rtgs(self,batch_rews):
        batch_rtgs = [[] for _ in stages] 
        batch_rets = [[] for _ in stages]
        for stage in stages:
            batch_rtgs[stage], batch_rets[stage] = self._compute_rtgs(batch_rews[stage])
        return batch_rtgs, batch_rets

    def _compute_rtgs(self,batch_rews):
        batch_rtgs = []
        batch_shape = len(batch_rews[0])*len(batch_rews)
        for ep_rews in reversed(batch_rews):
            s  = []
            for i in range(self.num_agents):
                s.append(0.0)
            discounted_reward = np.array(s).reshape(batch_rews[0][0].shape) 
            ep_rtgs = []
            cumu_ret = 0
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward *self.gamma
                ep_rtgs.insert(0, discounted_reward)
            batch_rtgs.append(ep_rtgs)
            cumu_ret += np.array(ep_rtgs[0])

        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float).reshape(batch_shape,self.num_agents)
        return batch_rtgs, cumu_ret / float(len(batch_rews))

    def learn(self):
        t_so_far = 0 
        i_so_far = 0 

        # Strict epoch-based outer loop constraint guaranteeing exactly 100 epochs
        while i_so_far < self.num_epochs:
            self.logger.info('=================================================================')
            score_AM = AverageMeter()
            loss_AM = AverageMeter()

            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_rets, batch_lens, batch_leader_returns = self.rollout()
            t_so_far += 1000  # Budgeted step count
            i_so_far += 1

            curr_epoch_results_dict = {}

            for ag in range(self.num_agents):
                for stage in stages:
                    a_loss, c_loss = self.agent_pool.learn(batch_obs, batch_acts, batch_log_probs, batch_rtgs,\
                            stage, ag, self.n_updates_per_iteration)
                    self.writer.add_scalar('actor_loss_stage_{}_agent_{}'.format(stage, ag), a_loss, t_so_far)
                    self.writer.add_scalar('critic_loss_stage_{}_agent_{}'.format(stage, ag), c_loss, t_so_far)
                    self.writer.add_scalar('return_stage_{}_agent_{}'.format(stage, ag), batch_rets[stage][ag], t_so_far)
                if i_so_far % self.save_freq == 0:
                    self.logger.info("Saving trained_model")
                    self.agent_pool.save_model(stage, ag, f'ppo_actor_agent{ag+1}_{i_so_far}.pth')

            self.logger.info("Epoch {:3d}/{:3d}]".format(i_so_far, self.num_epochs))

            # Save full episodic physical flows for paper-level plotting
            curr_epoch_results_dict['actual_d'] = self.env.actual_d
            curr_epoch_results_dict['waste_actual_d'] = self.env.waste_actual_d
            curr_epoch_results_dict['spot_q'] = self.env.spot_q
            curr_epoch_results_dict['price'] = self.env.price
            curr_epoch_results_dict['spot_price'] = self.env.spot_price
            curr_epoch_results_dict['inv'] = self.env.inv
            curr_epoch_results_dict['waste_inv'] = self.env.waste_inv
            curr_epoch_results_dict['rewards'] = np.sum(np.array(batch_rets), axis=(0))
            curr_epoch_results_dict['u_eco'] = self.env.eco_u
            curr_epoch_results_dict['u_tx'] = self.env.tx_u
            curr_epoch_results_dict['wastewater'] = self.env.wastewater
            curr_epoch_results_dict['raw_leader_return'] = np.mean(batch_leader_returns)

            np.save(str(debug_folder) + "/epoch={}_results.npy".format(i_so_far), curr_epoch_results_dict)

        self.logger.info(" *** Training Done *** ")

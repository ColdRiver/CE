import numpy as np
from config import config, init_historical_data
from surrogate_models import SurrogateModel

class Manufacturing_Simulator:
    """
    Environment class for the manufacturing problem (Flat MARL Baseline)
    """
    def __init__(self):
        for key, value in config.items():
            setattr(self, key, value)
        self.surrogate_model = SurrogateModel()
        
    def reset(self):
        self.t = self.history_length
        data_length = self.history_length+self.episode_length+1
        general_shape = (self.num_commodities, data_length)
        individual_shape = (self.num_agents, self.num_commodities, data_length)
        pair_shape = (self.num_agents, self.num_agents, self.num_commodities, data_length)

        self.spot_price = np.zeros(shape=general_shape)
        self.uc_p = np.zeros(shape=general_shape)
        self.tx_p = np.zeros(shape=general_shape)

        self.price = np.zeros(shape=individual_shape)
        self.waste_price = np.zeros(shape=individual_shape)

        self.q = np.zeros(shape=pair_shape)
        self.waste_q = np.zeros(shape=pair_shape)
        self.spot_q = np.zeros(shape=individual_shape)

        self.actual_d = np.zeros(shape=pair_shape)
        self.waste_actual_d = np.zeros(shape=pair_shape)

        self.inv = np.zeros(shape=individual_shape) 
        self.waste_inv = np.zeros(shape=individual_shape) 
        self.inv_sell = np.zeros(shape=individual_shape) 
        self.waste_inv_sell = np.zeros(shape=individual_shape) 
        self.inv_buy = np.zeros(shape=individual_shape) 
        self.waste_inv_buy = np.zeros(shape=individual_shape) 

        self.eco_u = np.zeros(shape=individual_shape)
        self.tx_u = np.zeros(shape=individual_shape)
        self.wastewater = np.zeros(shape=(1, 1, data_length))

        historical_data = init_historical_data()
        self.spot_price = np.repeat(historical_data['spot_price'], repeats=data_length, axis=1)
        self.uc_p = self.UC * self.spot_price
        self.tx_p = self.TX_P * self.spot_price
        self.inv[:, :, self.t] = self.inv[:, :, self.t] + self.INIT_INV
        self.waste_inv = self.inv.copy()

        return self.get_seller_state()
    
    def get_leader_state(self):
        """
        Upper-level state reflecting macro-environmental performance (14 dimensions)
        """
        avg_spot = np.mean(self.spot_price[:, self.t-self.history_length:self.t], axis=1)
        total_waste_landfilled = np.sum(self.spot_q[:, :, self.t-1]) if self.t > self.history_length else 0.
        total_freshwater = np.sum(self.inv[:, 0, self.t])
        return np.concatenate((avg_spot, [total_waste_landfilled, total_freshwater]))

    def get_seller_state(self):
        seller_states = []
        start_time = self.t-self.history_length
        for n in range(self.num_agents):
            p = self.spot_price[..., start_time:self.t].flatten()
            e = self.price[..., start_time:self.t].flatten()
            ew = self.waste_price[..., start_time:self.t].flatten()

            q = self.q[:, n, :, start_time:self.t].flatten()
            qw = self.waste_q[:, n, :, start_time:self.t].flatten()

            q_ = self.q[n, :, :, start_time:self.t].flatten()
            qw_ = self.waste_q[n, :, :, start_time:self.t].flatten()

            qs = self.spot_q[n, :, start_time:self.t].flatten()

            d = self.actual_d[n, :, :, start_time:self.t].flatten()
            dw = self.waste_actual_d[n, :, :, start_time:self.t].flatten()

            I = self.inv[n,:,start_time:self.t+1].flatten()
            Iw = self.waste_inv[n,:,start_time:self.t+1].flatten()

            u_eco = self.eco_u[n,:,start_time:self.t].flatten()
            u_tx = self.tx_u[n,:,start_time:self.t].flatten()   

            state_flat = np.concatenate((p, e, ew, q, qw, q_, qw_, qs, d, dw, I, Iw, u_eco, u_tx))
            seller_states.append(state_flat)

        return np.array(seller_states)

    def action_conversion(self, keys, actions):
        conv_actions = {k: np.zeros((self.num_agents, length), dtype=actions.dtype) for k, length in keys.items()}
        for i in range(self.num_agents):
            start = 0
            for key, length in keys.items():
                conv_actions[key][i] = actions[i, start:start + length]
                start += length
        return conv_actions

    def step_sell(self, seller_states, orig_seller_actions):
        keys = ['price', 'waste_price']
        key_len_dict = {k: self.num_commodities for k in keys}
        seller_actions = self.action_conversion(key_len_dict, orig_seller_actions)

        for key, value in seller_actions.items():
            getattr(self, key)[..., self.t] = value
        buyer_states = self.get_buyer_state(keys, seller_states, seller_actions)
        return buyer_states
    
    def get_seller_reward(self):
        actual_d = self.actual_d[:, :, :, self.t].sum(axis=0)
        waste_actual_d = self.waste_actual_d[:, :, :, self.t].sum(axis=0)

        reward = (self.price[:,:,self.t]*actual_d).sum(axis=1)
        reward += (self.waste_price[:,:,self.t]*waste_actual_d).sum(axis=1)
        return reward * self.RWD_SCALE
    
    def get_buyer_state(self, keys, seller_states, seller_actions):
        buyer_states = []
        for n in range(self.num_agents):
            state_flat = seller_states[n]
            state_flat = np.concatenate((state_flat, self.spot_price[:, self.t]))
            for key in keys:
                state_flat = np.concatenate((state_flat, seller_actions[key].flatten()))
        return np.array(buyer_states)

    def step_buy(self, buyer_states, orig_buyer_actions):
        keys = ['q', 'waste_q', 'spot_q']
        nc = (self.num_agents-1) * self.num_commodities
        lengths = [nc, nc, self.num_commodities]
        key_len_dict = {k: v for k, v in zip(keys, lengths)}
        buyer_actions = self.action_conversion(key_len_dict, orig_buyer_actions)
        for k, arr in buyer_actions.items():
            if k=='spot_q':
                continue
            new_actions = np.zeros((self.num_agents, self.num_agents, self.num_commodities))
            arr = arr.reshape(self.num_agents, self.num_agents - 1, self.num_commodities)
            for i in range(self.num_agents):
                i_list = list(range(self.num_agents))
                i_list.remove(i)
                new_actions[i, i_list] = arr[i]
            buyer_actions[k] = new_actions
            
        for key, value in buyer_actions.items():
            getattr(self, key)[..., self.t] = value

        actual_d = self.calc_actual_sold(self.q[:,:,:,self.t], self.inv[:,:,self.t])
        actual_dw = self.calc_actual_sold(self.waste_q[:,:,:,self.t], self.waste_inv[:,:,self.t])
        inv_buy =  self.calc_inv_buy(self.inv[:,:,self.t], actual_d, 0)
        waste_inv_buy = self.calc_inv_buy(self.waste_inv[:,:,self.t], actual_dw, 1)

        inv_buy = inv_buy + self.spot_q[:, :, self.t]

        self.actual_d[..., self.t] = actual_d
        self.waste_actual_d[..., self.t] = actual_dw
        self.inv_buy[..., self.t] = inv_buy
        self.waste_inv_buy[..., self.t] = waste_inv_buy
        
        buyer_rewards = []
        for n in range(self.num_agents):
            e_reshape = self.price[:,:,self.t].reshape(1, self.num_agents, self.num_commodities)
            ew_reshape = self.waste_price[:,:,self.t].reshape(1, self.num_agents, self.num_commodities)
            p_reshape = self.spot_price[:,self.t].reshape(1, self.num_commodities)

            reward = -np.sum(self.actual_d[n,:,:,self.t]*e_reshape)
            reward -= np.sum(self.waste_actual_d[n,:,:,self.t]*ew_reshape)
            reward -= np.sum(self.spot_q[n,:,self.t]*p_reshape)
            buyer_rewards.append(reward * self.RWD_SCALE)

        seller_rewards = self.get_seller_reward()
        for n in range(self.num_agents):
            buyer_rewards[n] += seller_rewards[n]

        trans_states = self.get_trans_state(keys, buyer_states, buyer_actions)
        return trans_states, np.array(buyer_rewards), seller_rewards
    
    def get_buyer_reward(self):
        e_reshape = self.price[:,:,self.t].reshape(1, self.num_agents, self.num_commodities)
        ew_reshape = self.waste_price[:,:,self.t].reshape(1, self.num_agents, self.num_commodities)
        p_reshape = self.spot_price[:,self.t].reshape(1, self.num_commodities)

        reward = -np.sum(self.actual_d[:,:,:,self.t]*e_reshape, axis=(1,2))
        reward -= np.sum(self.waste_actual_d[:,:,:,self.t]*ew_reshape, axis=(1,2))
        reward -= np.sum(self.spot_q[:,:,self.t]*p_reshape, axis=1)
        reward += self.LAMBDA*np.sum(self.actual_d[:,:,:,self.t]-self.q[:,:,:,self.t], axis=(1,2))
        reward += self.LAMBDA*np.sum(self.waste_actual_d[:,:,:,self.t]-self.waste_q[:,:,:,self.t], axis=(1,2))
        return reward * self.RWD_SCALE
    
    def get_trans_state(self, keys, buyer_states, buyer_actions):
        trans_states = []
        for n in range(self.num_agents):
            state_flat = buyer_states[n]
            for key in keys:
                state_flat = np.concatenate([state_flat, buyer_actions[key][n].flatten()])
            state_flat = np.concatenate([state_flat, self.actual_d[n,:,:,self.t].flatten(), self.waste_actual_d[n,:,:,self.t].flatten()])
            state_flat = np.concatenate([state_flat, self.inv_buy[n,:,self.t].flatten(), self.waste_inv_buy[n,:,self.t].flatten()])
            trans_states.append(state_flat)
        return np.array(trans_states)
    
    def h2_surrogate_input_conversion(self, water, acetic_acid):
        if water >= 19.*acetic_acid :
            return 20.* acetic_acid
        return 20.* water/19.

    def apply_agent_surrogate(self, tx_u):
        agent0 = tx_u[0]
        agent1 = tx_u[1]
        agent2 = tx_u[2]

        agents_final_output_vec = np.zeros_like(tx_u)
        agent0_surrogate_input_vec = np.array([agent0[5], agent0[7], agent0[2], agent0[0], agent0[1]])
        agent0_surrogate_output_vec = self.surrogate_model.get_apap_model_outputs(agent0_surrogate_input_vec.reshape(1,-1))
        agents_final_output_vec[0, [2, 6, 0]] = np.array(agent0_surrogate_output_vec)[0, [0, 1, 3]]
        self.wastewater[:, :, self.t] = np.array(agent0_surrogate_output_vec)[0, [3]]

        agent1_surrogate_input_vec = np.array([agent1[3], agent1[4], agent1[10], agent1[9]])
        agent1_surrogate_output_vec = self.surrogate_model.get_pap_model_outputs(agent1_surrogate_input_vec.reshape(1,-1))
        agents_final_output_vec[1, [11, 5]] = np.array(agent1_surrogate_output_vec)[0, [0, 1]]

        water_conv = agent2[0] if agent2[0] >= 19.*agent2[2] else 20.*agent2[0]/19.
        agent2_surrogate_input_vec = np.array([water_conv])
        agent2_surrogate_output_vec = self.surrogate_model.get_hyd_model_outputs(agent2_surrogate_input_vec.reshape(1,-1))
        agents_final_output_vec[2, [2, 8, 3, 0]] = np.array(agent2_surrogate_output_vec)[0, [0, 1, 3, 4]]

        agents_waste_final_output_vec = np.zeros_like(tx_u)
        agents_waste_final_output_vec[0, 0] = np.array(agent0_surrogate_output_vec)[0, 3]

        return agents_final_output_vec, agents_waste_final_output_vec

    def step_trans(self, trans_states, orig_trans_actions):
        keys = ['tx_u', 'eco_u']
        key_len_dict = {k: self.num_commodities for k in keys}
        trans_actions = self.action_conversion(key_len_dict, orig_trans_actions)
        trans_actions['tx_u'] = np.minimum(trans_actions['tx_u'], 0.5*self.inv_buy[...,self.t])
        trans_actions['eco_u'] = np.minimum(trans_actions['eco_u'], 0.5 * self.inv_buy[..., self.t])
        for key, value in trans_actions.items():
            getattr(self, key)[..., self.t] = value

        u_bot, w_bot = self.apply_agent_surrogate(trans_actions['tx_u'])
        self.inv[:,:,self.t+1] = np.maximum(self.inv_buy[:,:,self.t]-trans_actions['tx_u'][:,:]-\
            trans_actions['eco_u'][:,:]+u_bot, 0.)
        self.waste_inv[:,:,self.t+1] = (1-self.delta)*(self.waste_inv_buy[:,:,self.t]+w_bot)
        
        trans_rewards = self.get_trans_reward(trans_actions)
        
        recycled_volume = np.sum(self.waste_actual_d[..., self.t])
        landfilled_volume = np.sum(self.spot_q[..., self.t])
        freshwater_consumption = np.sum(self.spot_q[:, 0, self.t])
        leader_reward = recycled_volume - 1.0 * landfilled_volume - 0.1 * freshwater_consumption
        leader_reward_scaled = leader_reward * self.RWD_SCALE

        self.t += 1
        done = False
        if self.t==self.episode_length:
            done = True

        return self.get_leader_state(), self.get_seller_state(), np.array(trans_rewards), leader_reward_scaled, done
    
    def get_trans_reward(self, trans_actions):
        uc_p = self.uc_p[:, self.t].reshape(1, self.num_commodities)
        tx_p = self.tx_p[:, self.t].reshape(1, self.num_commodities)
        reward = np.sum(self.eco_u[:, :, self.t] * uc_p, axis=1)  
        reward -= np.sum(self.tx_u[:,:,self.t] * tx_p, axis=1)
        return reward * self.RWD_SCALE

    def calc_actual_sold(self, q, I):
        d = np.zeros_like(q)
        for c in range(self.num_commodities):
            for n in range(self.num_agents):
                buys = []
                for m in range(self.num_agents):
                    buys.append(q[m, n, c])
                buys = np.array(buys)
                sorted_indices = np.argsort(-buys)
                cum_sum = 0  
                for i in range(self.num_agents):
                    agent_i = sorted_indices[i]
                    if i == 0:
                        d[agent_i, n, c] = min(I[n, c], buys[agent_i])
                    else:
                        available_I = I[n, c] - cum_sum
                        if available_I <= 0:
                            break
                        d[agent_i, n, c] = min(available_I, buys[agent_i])
                    cum_sum += d[agent_i, n, c]
        return d
    
    def calc_inv_buy(self, I_bar, d, flag):
        I_bar  = I_bar + np.sum(d, axis=1)
        I_bar = I_bar - np.sum(d, axis=0)
        return I_bar

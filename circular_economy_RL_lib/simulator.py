import numpy as np
from config import config, init_historical_data
from surrogate_models import SurrogateModel

class Manufacturing_Simulator:
    """
    Environment class for the manufacturing problem
    Main functions:
    -- reset: return states for the seller agents
    -- step_sell: return states for the buyer agents and rewards for the seller agents
    -- step_buy: return states for the transformation agents and rewards for the buyer agents
    -- step_trans: return states for the seller agents and rewards for the transformation agents
    The rollout stops when the step_trans function returns done=True
    """
    def __init__(self):
        """
        Initialize the environment with the configuration
        """
        for key, value in config.items():
            setattr(self, key, value)

        self.surrogate_model = SurrogateModel()
        
    def reset(self):
        """
        Reset the environment to the initial state
        """
        # Set the starting time so the first self.history_length can  be used as historical data
        self.t = self.history_length

        ## Initilize the system parameters with the time step equal to self.history_length+self.episode_length
        ## Add additional time steps to avoid overflow
        data_length = self.history_length+self.episode_length+1
        # Shapes of the data for an episode
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

        self.inv = np.zeros(shape=individual_shape) # 
        self.waste_inv = np.zeros(shape=individual_shape) # 
        self.inv_sell = np.zeros(shape=individual_shape) # not being used
        self.waste_inv_sell = np.zeros(shape=individual_shape) # not being used
        self.inv_buy = np.zeros(shape=individual_shape) # 
        self.waste_inv_buy = np.zeros(shape=individual_shape) # 

        self.eco_u = np.zeros(shape=individual_shape)
        self.tx_u = np.zeros(shape=individual_shape)

        self.wastewater = np.zeros(shape=(1, 1, data_length))

        historical_data = init_historical_data()
        self.spot_price = np.repeat(historical_data['spot_price'], repeats=data_length, axis=1)
        self.uc_p = self.UC * self.spot_price
        self.tx_p = self.TX_P * self.spot_price
        self.inv[:, :, self.t] = self.inv[:, :, self.t] + self.INIT_INV
        self.waste_inv = self.inv.copy()
        # print(self.spot_price.shape, self.uc_p, self.tx_p, self.inv[:, :, self.t], self.waste_inv)

        # Make the first self.history_length time steps the same as the historical data
        # for key, value in historical_data.items():
        #     getattr(self, key)[..., :self.history_length] = value
        ## Initialize the seller state values
        return self.get_seller_state()
    
    def get_seller_state(self):
        """
        Get the output of the step 3 function to the states for the seller agents.
        """
        seller_states = []
        start_time = self.t-self.history_length
        for n in range(self.num_agents):
            # Flatten the specific slices
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
            # I_bar1 = self.inv_sell[n,:,start_time:self.t].flatten()
            # Iw_bar1 = self.waste_inv_sell[n,:,start_time:self.t].flatten()
            # I_bar = self.inv_buy[n,:,start_time:self.t].flatten()
            # Iw_bar = self.waste_inv_buy[n,:,start_time:self.t].flatten()

            u_eco = self.eco_u[n,:,start_time:self.t].flatten()
            u_tx = self.tx_u[n,:,start_time:self.t].flatten()   

            # Concatenate flattened arrays
            # danger                                        
            state_flat = np.concatenate((p, e, ew, q, qw, q_, qw_, qs, d, dw, I, Iw, u_eco, u_tx))
            # print(p.shape, e.shape, ew.shape, q.shape, qw.shape, q_.shape, qw_.shape,
            #       qs.shape, d.shape, dw.shape, I.shape, Iw.shape, u_eco.shape, u_tx.shape)
            # (35,) (105,) (105,) (105,) (105,) (105,) (105,) (35,) (105,) (105,) (42,) (42,) (35,) (35,)
            
            # Append the flattened state to the list of agent states
            seller_states.append(state_flat)

        return np.array(seller_states)

    def action_conversion(self, keys, actions):
        """
        The actions input is the direct output of the RL agent.
        It has to be converted to be meaningful system values.
        conv_actions = {'price': array arr_e of shape (num_agents, num_commodities), 'waste_price': arr_ew},
        where arr_e[k] is the price of seller k for all commodities

        The function only arranges the actions input to be a dictionary now.
        The output of the RL agent are values in [0,1]. Need further conversion to make them price, etc.
        """
        # TODO - Convert the value to be meaningful system values
        conv_actions = {k: np.zeros((self.num_agents, length), dtype=actions.dtype) for k, length in keys.items()}
        for i in range(self.num_agents):
            start = 0
            for key, length in keys.items():
                conv_actions[key][i] = actions[i, start:start + length]
                start += length
        return conv_actions

    def step_sell(self, seller_states, orig_seller_actions):
        """
        Step function for the selling step
        Get the seller actions and return the states for the buyer agents
        """
        # Seller action conversion
        keys = ['price', 'waste_price']
        key_len_dict = {k: self.num_commodities for k in keys}
        seller_actions = self.action_conversion(key_len_dict, orig_seller_actions)


        #import pdb; pdb.set_trace()
        # Update the seller states with the seller actions
        for key, value in seller_actions.items():
            getattr(self, key)[..., self.t] = value
        # Get the buyer states and seller rewards
        # print(seller_actions)
        # raise NotImplementedError {'price': array(3, 7), 'waste_price': array(3, 7)}
        buyer_states = self.get_buyer_state(keys, seller_states, seller_actions)
        # seller_rewards = self.get_seller_reward()
        return buyer_states
    
    # def get_seller_reward(self):
    #     """
    #     Get the rewards for the seller agents
    #     """
    #     reward = (self.price[:,:,self.t]*(self.inv[:,:,self.t]-self.inv_sell[:,:,self.t])).sum(axis=1)
    #     return reward

    def get_seller_reward(self):
        actual_d = self.actual_d[:, :, :, self.t].sum(axis=0)
        waste_actual_d = self.waste_actual_d[:, :, :, self.t].sum(axis=0)

        reward = (self.price[:,:,self.t]*actual_d).sum(axis=1)
        reward += (self.waste_price[:,:,self.t]*waste_actual_d).sum(axis=1)

        # print(actual_d.shape, waste_actual_d.shape, reward.shape)
        # (3, 7) (3, 7) (3,)
        # print("1", reward)
        return reward * self.RWD_SCALE
    
    def get_buyer_state(self, keys, seller_states, seller_actions):
        """
        Get the output of the step 1 function to the states for the buyer agents.
        """
        buyer_states = []

        for n in range(self.num_agents):
            state_flat = seller_states[n]
            # Add the new information to the seller state of the n-th agent
            state_flat = np.concatenate((state_flat, self.spot_price[:, self.t]))
            for key in keys:
                # if 'price' in key:
                state_flat = np.concatenate((state_flat, seller_actions[key].flatten()))
            buyer_states.append(state_flat)

        # # Update the system parameters # TODO: redundant
        # for key, value in seller_actions.items():
        #     getattr(self, key)[..., self.t] = value
        
        return np.array(buyer_states)

    def step_buy(self, buyer_states, orig_buyer_actions):
        """
        Step function for the buying step
        Get the buyer actions and return the states for the trans agents
        """
        # Buyer action conversion
        keys = ['q', 'waste_q', 'spot_q']
        nc = (self.num_agents-1) * self.num_commodities
        lengths = [nc, nc, self.num_commodities]
        key_len_dict = {k: v for k, v in zip(keys, lengths)}
        buyer_actions = self.action_conversion(key_len_dict, orig_buyer_actions)
        for k, arr in buyer_actions.items():
            if k=='spot_q':
                continue
            # print(k)
            new_actions = np.zeros((self.num_agents, self.num_agents, self.num_commodities))
            arr = arr.reshape(self.num_agents, self.num_agents - 1, self.num_commodities)
            for i in range(self.num_agents):
                i_list = list(range(self.num_agents))
                i_list.remove(i)
                # print(i_list, list(range(self.num_agents)))
                new_actions[i, i_list] = arr[i]
            buyer_actions[k] = new_actions
        #TODO - Action conversion for buyer
        # Update the buyer states with the buyer actions
        for key, value in buyer_actions.items():
            getattr(self, key)[..., self.t] = value
        # Get trans states and buyer rewards
        # print(buyer_actions['q'].shape) # (3, 3, 7)

        trans_states = self.get_trans_state(keys, buyer_states, buyer_actions)
        buyer_rewards = self.get_buyer_reward()
        seller_rewards = self.get_seller_reward()

        return trans_states, buyer_rewards, seller_rewards
    
    def get_buyer_reward(self):
        """
        Get the rewards for the buyer agents
        """
        
        e_reshape = self.price[:,:,self.t].reshape(1, self.num_agents, self.num_commodities)
        ew_reshape = self.waste_price[:,:,self.t].reshape(1, self.num_agents, self.num_commodities)
        p_reshape = self.spot_price[:,self.t].reshape(1, self.num_commodities)

        # print(self.actual_d[:,:,:,self.t].shape, (self.actual_d[:,:,:,self.t] * e_reshape).shape, self.spot_price[:,self.t].shape)

        reward = -np.sum(self.actual_d[:,:,:,self.t]*e_reshape, axis=(1,2))
        reward -= np.sum(self.waste_actual_d[:,:,:,self.t]*ew_reshape, axis=(1,2))
        reward -= np.sum(self.spot_q[:,:,self.t]*p_reshape, axis=1)
        reward += self.LAMBDA*np.sum(self.actual_d[:,:,:,self.t]-self.q[:,:,:,self.t], axis=(1,2))
        reward += self.LAMBDA*np.sum(self.waste_actual_d[:,:,:,self.t]-self.waste_q[:,:,:,self.t], axis=(1,2))
        # print("2", reward)
        return reward * self.RWD_SCALE
    
    def get_trans_state(self, keys, buyer_states, buyer_actions):
        """
        Get the output of the step 2 function to the states for the transformation agents.
        """
        actual_d = self.calc_actual_sold(self.q[:,:,:,self.t], self.inv[:,:,self.t])
        actual_dw = self.calc_actual_sold(self.waste_q[:,:,:,self.t], self.waste_inv[:,:,self.t])
        inv_buy =  self.calc_inv_buy(self.inv[:,:,self.t], actual_d, 0)
        waste_inv_buy = self.calc_inv_buy(self.waste_inv[:,:,self.t], actual_dw, 1)

        # from spot
        inv_buy = inv_buy + self.spot_q[:, :, self.t]

        trans_states = []
        for n in range(self.num_agents):
            state_flat = buyer_states[n]
            for key in keys:
                state_flat = np.concatenate([state_flat, buyer_actions[key][n].flatten()])
            # for key, value in buyer_actions.items():
            #     state_flat = np.concatenate(state_flat, value[n])
            # print(inv_buy.shape)
            state_flat = np.concatenate([state_flat, actual_d[n,:,:].flatten(), actual_dw[n,:,:].flatten()])
            state_flat = np.concatenate([state_flat, inv_buy[n,:].flatten(), waste_inv_buy[n,:].flatten()])
            trans_states.append(state_flat)
        
        # Update the system parameters
        # for key, value in buyer_actions.items():
        #     getattr(self, key)[..., self.t] = value
        self.actual_d[..., self.t] = actual_d
        self.waste_actual_d[..., self.t] = actual_dw
        self.inv_buy[..., self.t] = inv_buy
        self.waste_inv_buy[..., self.t] = waste_inv_buy
        
        return np.array(trans_states)
    

    def h2_surrogate_input_conversion(self, water, acetic_acid):
        if water >= 19.*acetic_acid :
            return 20.* acetic_acid

        return 20.* water/19.

    def apply_agent_surrogate(self, tx_u):
        """
        Agent 0: APAP, Agent 1: PAP, Agent 2: Hyd
        Shape input for agents according to surrogate model,
        Shape surrogate output to give appropriate transformation output
        """
        agent0 = tx_u[0]
        agent1 = tx_u[1]
        agent2 = tx_u[2]

        agents_final_output_vec = np.zeros_like(tx_u)

        agent0_surrogate_input_vec = np.array([agent0[5], agent0[7], agent0[2], agent0[0], agent0[1]])
        agent0_surrogate_output_vec = self.surrogate_model.get_apap_model_outputs(agent0_surrogate_input_vec.reshape(1,-1))
        agents_final_output_vec[0, [2, 6,0]] = np.array(agent0_surrogate_output_vec)[0,[0,1,3]] #for now assumed waste1 is water in the excel sheet

        self.wastewater[:, :, self.t] = np.array(agent0_surrogate_output_vec)[0, [3]]


        agent1_surrogate_input_vec = np.array([agent1[3], agent1[4], agent1[10], agent1[9]])
        agent1_surrogate_output_vec = self.surrogate_model.get_pap_model_outputs(agent1_surrogate_input_vec.reshape(1,-1))
        agents_final_output_vec[1, [11, 5]] = np.array(agent1_surrogate_output_vec)[0,[0,1]]

        agent2_surrogate_input_vec = np.array([self.h2_surrogate_input_conversion(agent2[0], agent2[2])])
        agent2_surrogate_output_vec = self.surrogate_model.get_hyd_model_outputs(agent2_surrogate_input_vec.reshape(1,-1))
        agents_final_output_vec[2, [2, 8, 3, 0]] = np.array(agent2_surrogate_output_vec)[0,[0, 1, 3, 4]]

        agents_waste_final_output_vec = np.zeros_like(tx_u)

        return agents_final_output_vec, agents_waste_final_output_vec


    def step_trans(self, trans_states, orig_trans_actions):
        """
        Step function for the transformation step
        Get the trans actions and return the states for the seller agents for the next time step
        """
        # Trans action conversion
        keys = ['tx_u', 'eco_u']
        key_len_dict = {k: self.num_commodities for k in keys}
        trans_actions = self.action_conversion(key_len_dict, orig_trans_actions)
        trans_actions['tx_u'] = np.minimum(trans_actions['tx_u'], 0.5*self.inv_buy[...,self.t])
        trans_actions['eco_u'] = np.minimum(trans_actions['eco_u'], 0.5 * self.inv_buy[..., self.t])
        # Update the state with the trans actions
        for key, value in trans_actions.items():
            getattr(self, key)[..., self.t] = value

        # Surrogate model implementation
        # TODO-Use the real surrogate model implementation
        #import pdb; pdb.set_trace()

        u_bot, w_bot = self.apply_agent_surrogate(trans_actions['tx_u'])
        #u_bot, w_bot = 0.1*trans_actions['tx_u'], 0.1*trans_actions['tx_u']
        # Calculate the inv and inv_waste for the next time step
        # print(trans_actions['tx_u'].shape, trans_actions['eco_u'].shape, self.inv_buy[:,:,self.t].shape)
        self.inv[:,:,self.t+1] = np.maximum(self.inv_buy[:,:,self.t]-trans_actions['tx_u'][:,:]-\
            trans_actions['eco_u'][:,:]+u_bot, 0.)
        self.waste_inv[:,:,self.t+1] = (1-self.delta)*(self.waste_inv_buy[:,:,self.t]+w_bot)
        # Get the seller states and trans rewards
        trans_rewards = self.get_trans_reward(trans_actions)
        self.t += 1
        seller_states = self.get_seller_state()
        # print(trans_actions.keys())
        done = False
        # Check if the episode is done
        if self.t==self.episode_length:
            done = True

        return seller_states, trans_rewards, done
    
    def get_trans_reward(self, trans_actions):
        """
        Get the rewards for the transformation agents
        """
        uc_p = self.uc_p[:, self.t].reshape(1, self.num_commodities)
        tx_p = self.tx_p[:, self.t].reshape(1, self.num_commodities)
        # print(self.tx_p.shape, self.tx_u.shape, self.eco_u.shape, p_reshape.shape)
        # reward = np.sum(trans_actions['econ_quantity'], axis=0)
        reward = np.sum(self.eco_u[:, :, self.t] * uc_p, axis=1)  # danger
        reward -= np.sum(self.tx_u[:,:,self.t] * tx_p, axis=1)
        # print("3", reward)
        return reward * self.RWD_SCALE

    def calc_actual_sold(self, q, I):
        """
        Calculate the actual sold quantities
        """
        # Initialize the tensor d with the same shape as q
        d = np.zeros_like(q)


        # Generate the tensor d
        for c in range(self.num_commodities):
            for n in range(self.num_agents):
                
                buys = []
                for m in range(self.num_agents):
                    buys.append(q[m, n, c])

                buys = np.array(buys)
                # Step 1: Sort agents based on q[c, :, n] in descending order
                # sorted_indices = np.argsort(-q[n, :, c])
                sorted_indices = np.argsort(-buys)

                # Step 2: Compute d for each agent in the sorted list
                cum_sum = 0  # Initialize cumulative sum
                for i in range(self.num_agents):
                    agent_i = sorted_indices[i]
                    if i == 0:
                        # First agent n(1)
                        # d[n, agent_i, c] = min(I[n, c], q[n, agent_i, c])
                        d[agent_i, n, c] = min(I[n, c], buys[agent_i])
                    else:
                        # Subsequent agents n(i)
                        available_I = I[n, c] - cum_sum
                        if available_I <= 0:
                            break
                        d[agent_i, n, c] = min(available_I, buys[agent_i])
                    
                    # Update cumulative sum
                    cum_sum += d[agent_i, n, c]

        return d
    
    def calc_inv_buy(self, I_bar, d, flag):
        """
        Calculate the inventory bought by the buyer agents
        """
        #print("flag: {}".format(flag))

        # buying
        I_bar  = I_bar + np.sum(d, axis=1)
        # selling
        I_bar = I_bar - np.sum(d, axis=0)

        # if self.t % 100 == 0:
        #     import pdb;pdb.set_trace()

        return I_bar
    
    
       

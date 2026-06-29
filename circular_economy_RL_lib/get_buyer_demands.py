import torch
import time
import copy
import math
import warnings
import numpy as np
import cvxpy as cp
from mip_solver import MIPSolver
warnings.filterwarnings("ignore")

class sys_funcs:
    def __init__(self,coefs):
        self.coefs = coefs
    
    def utility(self,demand):
        return self.coefs[0]*demand + self.coefs[1]

    def waste_utility(self,waste_demand):
        return cp.power(waste_demand,0.50)
    
    def transport_cost(self,demand):
        return self.coefs[2]*demand + self.coefs[3]

    def waste_cost(waste_qty):
        pass
  

def get_buyer_quantities_multi_commodities(exchange_price_action_dict_arr, agent, total_agents, complete_dict, coefs, t):
    '''
    Connection between the main step function and the buyer problem solver

    Let one of the industries be the seller and the rest be the buyers.
    Solve the buyer problem  by the MIP solver to get the buyer quantities 
        for all the commodities for the rest industries.

    # Inputs:
    exchange_price_action_dict_arr: actions of sellers--price of the commodities, shape=(total_agents, num_commodities)
    agent: The id of the agent who will server as a seller in this buyer problem
    total agents: The number of agents (industries) in the system
    complete_dict: System patameters such as the price of spot market for each commodities or the inventory

    # Outputs:
    spot_demand_qty_arr: The spot demand quantity of the buyer for all the commodities, shape=(num_commodities,)
    exchange_demand_qty_arr: The demand quantity for the exchange market for all the commodities, 
        shape=(total_agent, num_commodities)
    waste_exchange_qty_arr: The demand quantity for the waste for all the commodities,
        shape=(total_agent, num_commodities)
    buyer_reward_total_arr: The optimized objective value of the buyer problem for all the commodities,
        shape=(num_commodities,)
    '''
    num_commodities = len(exchange_price_action_dict_arr)
    num_seller = total_agents-1

    spot_demand_qty_arr = np.zeros(num_commodities)
    exchange_demand_qty_arr = np.zeros((num_commodities, total_agents))
    waste_exchange_qty_arr = np.zeros((num_commodities, total_agents))
    buyer_reward_total_arr = np.zeros(num_commodities)
    
    #class MIPSolver:
    #    def __init__(self, num_sellers: int, num_buyers: int, horizon: int =12):
    solver = MIPSolver(total_agents-1, total_agents)

    #TODO-Set up the buyer params from the function inputs
    buyer_params = {}
    buyer_params['pt'] = complete_dict[f'data_dict_agent{agent+1}']['spot_price'][t] # ()
    buyer_params['Kj'] = complete_dict[f'data_dict_agent{agent+1}']['buyer_init_inventory'] # ()
    buyer_params['eit'] = exchange_price_action_dict_arr[f'agent{agent+1}'] # (total_agents-1,)
    buyer_params['weight_tran_cost'] = coefs[2]*np.ones(num_seller) # (total_agents-1,)
    buyer_params['bias_tran_cost'] = coefs[3]*np.ones(num_seller) # (total_agents-1,)
    buyer_params['weight_spot_cost'] = coefs[2] # ()
    buyer_params['bias_spot_cost'] = coefs[3] # ()
    buyer_params['weight_utility'] = coefs[0] # ()
    buyer_params['bias_utility'] = coefs[1] # ()

    for c_id in range(num_commodities):
        # spot_demand_qty: ()
        # exchange_demand_qty: (total_agents-1, )
        # waste_exchange_qty: (total_agents-1, )
        # buyer_reward_total: ()
        spot_demand_qty, exchange_demand_qty, waste_exchange_qty, buyer_reward_total =\
                solver.solve_buyer(buyer_params, agent)

        # Update the arrays
        exchange_demand_qty_arr[c_id, :] = exchange_demand_qty
        waste_exchange_qty_arr[c_id, :] = waste_exchange_qty
        spot_demand_qty_arr[c_id] = spot_demand_qty
        buyer_reward_total_arr[c_id] = buyer_reward_total

    return spot_demand_qty_arr, exchange_demand_qty_arr, waste_exchange_qty_arr, buyer_reward_total_arr

def get_buyer_quantities(exchange_price_action_dict,agent,total_agents,complete_dict,coefs,t):
    '''
    Takes the current seller agent as input.
    Treats all the other agents in the network except the current seller agent as a buyer.
    For each buyer agent => we define spot demand quantitites and exchange demands for each buyer-seller pair.
    Then for each buyer, we find the spot and exchange demand qtys that maximize the buyer reward which takes into
    consideration the exchange prices of all the sellers.

    For eg. For 4-agent environment, if agent 2 is a buyer , agent 1,3 and 4 are considered as sellers and their exchange
    prices are cosidered in the buyer reward of agent 2.
    
    For the given seller, the function returns the spot and exchange demands of all the other buyer agents and also their 
    corresponding buyer rewards.
    '''
    Sys_funcs = sys_funcs(coefs)
    buyer_spot_quantities = {}
    buyer_exchange_quantities = {}
    waste_exchange_quantities = {}
    for j in range(total_agents):
        buyer_spot_quantities[f'agent{j+1}'] = cp.Variable(nonneg = True)
        buyer_exchange_quantities[f'agent{j+1}'] = {}
        waste_exchange_quantities[f'agent{j+1}'] = {}
        for i in range(total_agents):
            buyer_exchange_quantities[f'agent{j+1}'][f'agent{i+1}'] =  cp.Variable(nonneg = True)
            waste_exchange_quantities[f'agent{j+1}'][f'agent{i+1}'] =  cp.Variable(nonneg = True)
    
    buyer_spot_qtys = {}  
    buyer_exchange_qtys = {} 
    waste_exchange_qtys  = {} 
    buyer_reward_total = {}

    # this loop excludes the seller agent and calculates the buyer reward function for all the other buyers.
    for j in range(total_agents):
        if f'agent{j+1}' == agent:
            continue
        total_demand_by_agent = buyer_spot_quantities[f'agent{j+1}'] 
        for i in range(total_agents):
            if f'agent{i+1}' == f'agent{j+1}':
                continue
            total_demand_by_agent += buyer_exchange_quantities[f'agent{j+1}'][f'agent{i+1}']
            total_demand_by_agent += waste_exchange_quantities[f'agent{j+1}'][f'agent{i+1}']

        function = Sys_funcs.utility(total_demand_by_agent)-\
                complete_dict[f'data_dict_agent{j+1}']['spot_price'][t]*buyer_spot_quantities[f'agent{j+1}']-\
                Sys_funcs.transport_cost(buyer_spot_quantities[f'agent{j+1}'])

        for i in range(total_agents):
            if f'agent{i+1}' == f'agent{j+1}':
                continue
            function -= exchange_price_action_dict[f'agent{i+1}']*(buyer_exchange_quantities[f'agent{j+1}'][f'agent{i+1}']\
                    +waste_exchange_quantities[f'agent{j+1}'][f'agent{i+1}'])
            function -= Sys_funcs.transport_cost(buyer_exchange_quantities[f'agent{j+1}'][f'agent{i+1}']+\
                    waste_exchange_quantities[f'agent{j+1}'][f'agent{i+1}'])
        
        objective = cp.Maximize(function)
        constraints = [total_demand_by_agent<=complete_dict[f'data_dict_agent{i+1}']['buyer_init_inventory']]
        problem = cp.Problem(objective, constraints)
        problem.solve()
        buyer_reward_total[f'agent{j+1}'] = problem.value
        buyer_spot_qtys[f'agent{j+1}'] = buyer_spot_quantities[f'agent{j+1}'].value
        buyer_exchange_qtys[f'agent{j+1}'] = buyer_exchange_quantities[f'agent{j+1}'][agent].value
        waste_exchange_qtys[f'agent{j+1}'] = waste_exchange_quantities[f'agent{j+1}'][agent].value
    
    return buyer_spot_qtys,buyer_exchange_qtys,waste_exchange_qtys,buyer_reward_total

import math
import numpy as np
from dataclasses import dataclass


## Settings for the stages
SELLER = 0
BUYER = 1
TRANSFORM = 2
stages = {SELLER, BUYER, TRANSFORM}

## Settings for system data initialization
@dataclass
class DataStats:
    mean: float
    scale: float
    std: float
    shape: tuple

config = {
    # System parameters
    'num_agents': 3,
    'num_commodities': 12,
    # Optimization parameters
    'alpha': 0.5,
    'beta': 1.5,
    'delta': 0.5,
    'LAMBDA': 0.5,
    'UC': 0.5,
    'TX_P': 0.5,
    'INIT_INV': 100,
    'RWD_SCALE': 1e-9,
    # Training parameters
    'gamma': 0.99,
    'num_episodes': 100,
    'num_steps': 1000, # Number of steps per epoch
    'episode_length': 1000,
    'num_epochs': 100,
    'history_length': 5,
    'decay_factor': 0.1,
    'n_updates_per_iteration': 10,
    'lr': 3e-4,
    'clip': 0.2,
    'save_freq': 1,
    'seed': 2024,
    'phi': [100., 100., 100.],
    'price_factor' : 1.,
}

# Data Shapes for initiliazing the data
general_shape = (config['num_commodities'], config['history_length']) 
individual_shape = (config['num_agents'], config['num_commodities'], config['history_length'])
pair_shape = (config['num_agents'], config['num_agents'], config['num_commodities'], config['history_length'])
# data_config = {
#     # Data parameters
#     'spot_price': DataStats(500., 1., 50., general_shape), # mean, scale, std, 0 
#     'price': DataStats(500., 1., 50., individual_shape),
#     'waste_price': DataStats(500., 1., 50., individual_shape),
#     'q': DataStats(500., 1., 50., pair_shape),
#     'waste_q': DataStats(500., 1., 50., pair_shape),
#     'spot_q': DataStats(500., 1., 50., individual_shape),
#     'actual_d': DataStats(500., 1., 50., pair_shape),
#     'waste_actual_d': DataStats(500., 1., 50., pair_shape),
#     'inv': DataStats(500., 1., 50., individual_shape),
#     'waste_inv': DataStats(500., 1., 50., individual_shape),
#     'inv_sell': DataStats(500., 1., 50., individual_shape),
#     'waste_inv_sell': DataStats(500., 1., 50., individual_shape),
#     'inv_buy': DataStats(400., 1., 50., individual_shape),
#     'waste_inv_buy': DataStats(400., 1., 50., individual_shape),
#     'eco_u': DataStats(500., 1., 50., individual_shape),
#     'tx_u': DataStats(500., 1., 50., individual_shape),
#     'tx_p': DataStats(500., 1., 50., individual_shape),
# } # TODO: danger

def init_historical_data():
    historic_data = {}
    historic_data['spot_price'] = np.array([[config['price_factor']*0.5], [0.8], [1.], [3.], [20.], [4.], [8.], [100.], [0.2], [1.2], [0.15], [1.173]])
    # commodities in order: [water-0, costly water-1, acetic acid-2, hyderogen-3, nitrobenzene-4, PAP-5, Acetaminofen-6, acetic anhydride-7, oxygen-8,
    # NH3-9, H2SO4-10, aniline-11]
    # for param, param_config in data_config.items():
    #     historic_data[param] = np.random.normal(param_config.mean, param_config.std, param_config.shape)
    # # historic_data['spot_price'] = np.random.normal(config['spot_price'].mean, config['spot_price'].std, general_shape)
    # # historic_data['price'] = np.random.normal(config['price'].mean, config['price'].std, individual_shape)
    # # historic_data['waste_price'] = np.random.normal(config['waste_price'].mean, config['waste_price'].std, individual_shape)
    # # historic_data['q'] = np.random.normal(config['q'].mean, config['q'].std, pair_shape)
    # # historic_data['waste_q'] = np.random.normal(config['waste_q'].mean, config['waste_q'].std, pair_shape)
    # # historic_data['spot_q'] = np.random.normal(config['spot_q'].mean, config['spot_q'].std, individual_shape)
    # # historic_data['actual_d'] = np.random.normal(config['actual_d'].mean, config['actual_d'].std, pair_shape)
    # # historic_data['waste_actual_d'] = np.random.normal(config['waste_actual_d'].mean, config['waste_actual_d'].std, pair_shape)
    # # historic_data['inv'] = np.random.normal(config['inv'].mean, config['inv'].std, individual_shape)
    # # historic_data['waste_inv'] = np.random.normal(config['waste_inv'].mean, config['waste_inv'].std, individual_shape)
    # # historic_data['inv_sell'] = np.random.normal(config['inv_sell'].mean, config['inv_sell'].std, individual_shape)
    # # historic_data['waste_inv_sell'] = np.random.normal(config['waste_inv_sell'].mean, config['waste_inv_sell'].std, individual_shape)
    # # historic_data['inv_buy'] = np.random.normal(config['inv_buy'].mean, config['inv_buy'].std, individual_shape)
    # # historic_data['waste_inv_buy'] = np.random.normal(config['waste_inv_buy'].mean, config['waste_inv_buy'].std, individual_shape)
    # # historic_data['eco_u'] = np.random.normal(config['eco_u'].mean, config['eco_u'].std, individual_shape)
    # # historic_data['tx_u'] = np.random.normal(config['tx_u'].mean, config['tx_u'].std, individual_shape)
    # # historic_data['tx_p'] = np.random.normal(config['tx_p'].mean, config['tx_p'].std, individual_shape)
    return historic_data

## For the transformation models
## Usage: Call the get model_output function with the equations and the current state and control input
def evaluate_equations(equation, x, u):
    """
    Given the ODE equation, the current state x and control input u, evaluate the equation
    """
    # Convert x and u into dictionaries for easy variable name access
    x_dict = {f'x{i}': val for i, val in enumerate(x)}
    u_dict = {f'u{i}': val for i, val in enumerate(u)}
    variables = {**x_dict, **u_dict, "cos": math.cos, "sin": math.sin}
    
    # Replace carat '^' with '**' for python power operator
    equation = equation.replace('^', '**')

    # Evaluate the equation using eval
    return eval(equation, {"__builtins__": None}, variables)

# Transformation models
def get_model_output(equations, x, u):
    """
    Get the derivative of the state variables x using the given equations
    u are the u^{tx}_{c,n} and x are the combination of u_{c,n} and w_{c,n}
    """
    result = np.zeros(len(equations))
    for i, equation in enumerate(equations):
        result[i] = evaluate_equations(equation, x, u)
    return result

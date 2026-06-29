import numpy as np

exogenous_cost_attr = ['spot_price', 'holding_cost', 'waste_disposal_cost', 'exchange_transport_cost', 'spot_transport_cost']

exogenous_qty_attr = ['recycled_qty', 'waste_qty', 'produced_qty','total_demand_qty']

## TODO-import config in this file
def synthetic_exo_data_generator(total_timesteps = 2048, num_commodities=4, *args, **kwargs):
    
    data_dict_obj = {}
    dims = (num_commodities, total_timesteps)
    #cost attributes sampled for total timesteps from a log-normal distribution
    data_dict_obj['spot_price'] = np.random.normal(500., 50., dims)
    data_dict_obj['holding_cost'] = 0.15*np.random.normal(500., 50., dims)
    data_dict_obj['waste_disposal_cost'] = 0.05*np.random.normal(500., 50., dims)
    #ignoring spot, exchange transport costs

    data_dict_obj['produced_qty'] = np.random.normal(200., 20., dims)
    #data_dict_obj['recycled_qty'] = 0.05*np.random.normal(200., 20., dims)
    data_dict_obj['waste_qty'] = 0.10*np.random.normal(200., 20., dims)
    data_dict_obj['seller_init_inventory'] = 500.*np.ones(num_commodities)
    data_dict_obj['buyer_init_inventory'] = 400.*np.ones(num_commodities)
    return data_dict_obj






    

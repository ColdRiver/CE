import cvxpy as cp
import numpy as np
from mip import Model, xsum, minimize, maximize, CONTINUOUS, OptimizationStatus
from mip_solver import Solver

SELLER_PROBLEM = 0
BUYER_PROBLEM = 1

class MIPSolver(Solver):
    def __init__(self, num_sellers: int, num_buyers: int, horizon: int =12):
        super().__init__(num_sellers, num_buyers, horizon)

    def solve_buyer(self):
        pass

    def solve_seller(self, params, agent):
        m = Model()
        # Parameters
        num_commodities = params['num_commodities']
        avg_spot_price = params['avg_spot_price']
        inv = params['inv']
        waste_inv = params['waste_inv']
        alpha = params['alpha']

        # Variables
        e = [m.add_var(var_type=CONTINUOUS) for _ in range(num_commodities)]
        ew = [m.add_var(var_type=CONTINUOUS) for _ in range(num_commodities)]
        carry_inv = [m.add_var(var_type=CONTINUOUS) for _ in range(num_commodities)]
        carry_waste = [m.add_var(var_type=CONTINUOUS) for _ in range(num_commodities)]

        # Objective
        m.objective = maximize(xsum(e[i]*(inv[i]-carry_inv[i])+ew[i]*(waste_inv[i]-carry_waste[i])+\
                alpha*(e[i]*carry_inv[i]+ew[i]*carry_waste[i]) for i in range(num_commodities)))

        m.emphasis=2
        
        # Constraints
        for i in range(num_commodities):
            m += e[i]<=avg_spot_price
            m += carry_inv[i] <= inv[i]
            m += carry_waste[i] <= waste_inv[i]

        #Optimize
        status = m.optimize()

        if (status != OptimizationStatus.OPTIMAL and status != OptimizationStatus.FEASIBLE):
            return "Infeasible", m.objective_value, None, None, None, None 

        price = np.zeros(num_commodities)
        waste_price = np.zeros(num_commodities)
        inv = np.zeros(num_commodities)
        waste_inv = np.zeros(num_commodities)
        for i in range(num_commodities):
            price[i]=e[i].x
            waste_price[i]=ew[i].x
            inv[i]=carry_inv[i].x
            waste_inv[i]=carry_waste[i].x

        return "Pass", m.objective_value, price, waste_price, inv, waste_inv

class CVXSolver(Solver):
    def __init__(self, num_sellers: int, num_buyers: int, horizon: int =12):
        super().__init__(num_sellers, num_buyers, horizon)

    def solve_buyer(self, params, agent):
        '''
        Solves the optimization problem where the agent servers as the buyer and
        the reset agents server as sellers.
        In this case, self.num_sellers = self.num_buyers-1
        '''
        status, obj, action = self.solve(params, BUYER_PROBLEM)
        buyer_reward_total_arr = obj 
        exchange_demand_qty = np.insert(action['exchange_qty'], agent-1, 0)
        waste_exchange_qty = np.insert(action['waste_exchange_qty'], agent-1, 0)
        spot_demand_qty = action['spot_qty']

        # return spot_demand_qty, exchange_demand_qty, waste_exchange_qty, buyer_reward,total 


    def solve(self, params, problem_type: int):
        if problem_type==SELLER_PROBLEM:
            if self.seller_problem is None:
                self.create_seller_model()
            parameter_dict = self.seller_params
            problem = self.seller_problem
            variables = self.seller_variables
        elif problem_type==BUYER_PROBLEM:
            if self.buyer_problem is None:
                self.create_buyer_model()
            parameter_dict = self.buyer_params
            problem = self.buyer_problem
            variables = self.buyer_variables
        else:
            print(f'Problem type should be either 0 for seller or 1 for buyer')
            raise RuntimeError

        for para in parameter_dict:
            if para not in params:
                print(f'The parameter {para} is not provided')
                raise RuntimeError
            parameter_dict[para].value = params[para]
        problem.solve()
        action = {}
        for key, var in variables.items():
            action[key] = var.value
        return problem.status, problem.value, action

    def create_updated_buyer_model(self):
        pass

    def create_updated_seller_model(self):
        # Variables
        self.updated_seller_variables = {}
        seller_exchange_price = cp.Variable(shape=self.num_commodities, nonneg=True, name='ecnt')
        seller_waste_price = cp.Variable(shape=self.num_commodities, nonneg=True, name='ewcnt')
        seller_carrying_inv = cp.Variable(shape=self.num_commodities, nonneg=True, name='icnt')
        seller_carrying_waste_inv = cp.Variable(shape=self.num_commodities, nonneg=True, name='iwcnt')
        self.updated_seller_variables['exchange_price'] = seller_exchange_price
        self.updated_seller_variables['waste_price'] = seller_waste_price 
        self.updated_seller_variables['carrying_inv'] = seller_carrying_inv 
        self.updated_seller_variables['carrying_waste_inv'] = seller_carrying_waste_inv 
        # Parameters
        self.updated_seller_params = {}
        alpha = cp.Parameter(nonneg=True, name='alpha') 
        avg_spot_price = cp.Parameter(nonneg=True, name='avg_spot_price') 
        inv = cp.Parameter(shape=self.num_commodities, nonneg=True, name='inv')
        waste_inv = cp.Parameter(shape=self.num_commodities, nonneg=True, name='waste_inv')
        self.updated_seller_params['alpha'] = alpha
        self.updated_seller_params['avg_spot_price'] = avg_spot_price 
        self.updated_seller_params['inv'] = inv
        self.updated_seller_params['waste_inv'] = waste_inv 
        ## TODO-add objectives and constraints


    def create_seller_model(self):
        # Variables
        self.seller_variables = {}
        seller_exchange_price = cp.Variable(nonneg=True, name='eit')
        seller_exchange_qty = cp.Variable(shape=self.num_buyers, nonneg=True, name='d_jit')
        seller_waste_qty = cp.Variable(nonneg=True, name='wi')
        seller_recycle_qty = cp.Variable(nonneg=True, name='uit_tilda')
        self.seller_variables['exchange_price'] = seller_exchange_price
        self.seller_variables['exchange_qty'] = seller_exchange_qty
        self.seller_variables['waste_qty'] = seller_waste_qty
        self.seller_variables['recycle_qty'] = seller_recycle_qty 
        # Parameters
        self.seller_params = {}
        pt = cp.Parameter(nonneg=True, name='pt') 
        delta = cp.Parameter(nonneg=True, name='delta') 
        gamma = cp.Parameter(nonneg=True, name='gamma') 
        alpha = cp.Parameter(nonneg=True, name='alpha') 
        prev_uit = cp.Parameter(nonneg=True, name='prev_uit')
        Ki = cp.Parameter(nonneg=True, name='Ki')
        hit = cp.Parameter(nonneg=True, name='hit')
        prev_exchange_qty = cp.Parameter(shape=(self.num_buyers, self.horizon), nonneg=True)
        self.seller_params['pt'] = pt
        self.seller_params['delta'] = delta
        self.seller_params['gamma'] = gamma
        self.seller_params['alpha'] = alpha
        self.seller_params['prev_uit'] = prev_uit
        self.seller_params['Ki'] = Ki
        self.seller_params['hit'] = hit
        self.seller_params['prev_exchange_qty'] = prev_exchange_qty 

        # Functions
        weight_tran_cost = cp.Parameter(self.num_buyers, name='w_tc')
        bias_tran_cost = cp.Parameter(self.num_buyers, name='b_tc')
        tran_cost = cp.multiply(weight_tran_cost, seller_exchange_qty)+bias_tran_cost
        weight_spot_cost = cp.Parameter(self.num_buyers, name='w_sc')
        bias_spot_cost = cp.Parameter(self.num_buyers, name='b_sc')
        spot_cost = cp.multiply(weight_spot_cost, seller_exchange_qty)+bias_spot_cost
        weight_waste_cost = cp.Parameter(name='w_wc')
        bias_waste_cost = cp.Parameter(name='b_wc')
        waste_cost = weight_waste_cost*seller_waste_qty+bias_waste_cost
        self.seller_params['weight_tran_cost'] = weight_tran_cost 
        self.seller_params['bias_tran_cost'] = bias_tran_cost 
        self.seller_params['weight_spot_cost'] = weight_spot_cost 
        self.seller_params['bias_spot_cost'] = bias_spot_cost
        self.seller_params['weight_waste_cost'] = weight_waste_cost
        self.seller_params['bias_waste_cost'] = bias_waste_cost
        # Optimization Problem
        total_supply = cp.sum(seller_exchange_qty)
        new_inv = prev_uit-total_supply+seller_recycle_qty-seller_waste_qty 
        #objective = seller_exchange_price*total_supply+cp.sum(bias_tran_cost)-waste_cost+hit*new_inv
        objective = seller_exchange_price+total_supply+cp.sum(tran_cost)-waste_cost+hit*new_inv
        constraints = []
        constraints.append(seller_exchange_price<=delta*pt) # Eq(5)
        constraints.append(tran_cost<=gamma*spot_cost) # Eq(6)
        constraints.append(total_supply<=prev_uit) # Eq(7) and Eq(8)
        constraints.append(seller_exchange_qty<=alpha/self.horizon*cp.sum(prev_exchange_qty, 1)) # Eq(9)
        constraints.append(new_inv<=Ki) # Eq(10) and Eq(11)
        self.seller_problem = cp.Problem(cp.Maximize(objective), constraints)

    def create_buyer_model(self):
        # Variables
        self.buyer_variables = {}
        buyer_spot_qty = cp.Variable(nonneg=True, name='ds_jt')
        buyer_exchange_qty = cp.Variable(shape=self.num_sellers, nonneg=True, name='d_jit')
        waste_exchange_qty = cp.Variable(shape=self.num_sellers, nonneg=True, name='w_jit')
        self.buyer_variables['spot_qty'] = buyer_spot_qty 
        self.buyer_variables['exchange_qty'] = buyer_exchange_qty 
        self.buyer_variables['waste_exchange_qty'] = waste_exchange_qty 

        total_demand = buyer_spot_qty+cp.sum(buyer_exchange_qty)+cp.sum(waste_exchange_qty)
        # Parameters
        self.buyer_params = {}
        pt = cp.Parameter(nonneg=True, name='pt')
        Kj = cp.Parameter(nonneg=True, name='Kj')
        eit = cp.Parameter(shape=self.num_sellers, nonneg=True, name='eit')
        self.buyer_params['pt'] = pt
        self.buyer_params['Kj'] = Kj
        self.buyer_params['eit'] = eit
        # Functions
        weight_tran_cost = cp.Parameter(self.num_sellers, name='w_tc')
        bias_tran_cost = cp.Parameter(self.num_sellers, name='b_tc')
        tran_cost = cp.multiply(weight_tran_cost, buyer_exchange_qty+waste_exchange_qty)+bias_tran_cost
        weight_spot_cost = cp.Parameter(name='w_sc')
        bias_spot_cost = cp.Parameter(name='b_sc')
        spot_cost = weight_spot_cost*buyer_spot_qty+bias_spot_cost
        weight_utility = cp.Parameter(name='w_u')
        bias_utility = cp.Parameter(name='b_u')
        utility = weight_utility*total_demand+bias_utility
        self.buyer_params['weight_tran_cost'] = weight_tran_cost 
        self.buyer_params['bias_tran_cost'] = bias_tran_cost 
        self.buyer_params['weight_spot_cost'] = weight_spot_cost 
        self.buyer_params['bias_spot_cost'] = bias_spot_cost
        self.buyer_params['weight_utility'] = weight_utility
        self.buyer_params['bias_utility'] = bias_utility
        # Optimization Problem
        objective = utility-pt*buyer_spot_qty-spot_cost-cp.sum(cp.multiply(eit,\
                buyer_exchange_qty+waste_exchange_qty))-cp.sum(tran_cost)
        constraints = []
        constraints.append(total_demand<=Kj)
        self.buyer_problem = cp.Problem(cp.Maximize(objective), constraints)

def get_test_params(num_buyer, num_seller, horizon=12):
    buyer_params = {}
    buyer_params['pt'] = 1 
    buyer_params['Kj'] = 100
    buyer_params['eit'] = np.ones(num_seller)
    buyer_params['weight_tran_cost'] = 2*np.ones(num_seller)
    buyer_params['bias_tran_cost'] = 3*np.ones(num_seller)
    buyer_params['weight_spot_cost'] = 4
    buyer_params['bias_spot_cost'] = 5
    buyer_params['weight_utility'] = 6
    buyer_params['bias_utility'] = 7

    seller_params = {}
    seller_params['pt'] = 1 
    seller_params['delta'] = 2 
    seller_params['gamma'] = 3
    seller_params['alpha'] = 4
    seller_params['prev_uit'] = 20 
    seller_params['Ki'] = 100
    seller_params['hit'] = 10
    seller_params['prev_exchange_qty'] = 5*np.ones((num_buyer,horizon))
    seller_params['weight_tran_cost'] = 2*np.ones(num_buyer) 
    seller_params['bias_tran_cost'] = 3*np.ones(num_buyer) 
    seller_params['weight_spot_cost'] = 4*np.ones(num_buyer) 
    seller_params['bias_spot_cost'] = 5*np.ones(num_buyer)
    seller_params['weight_waste_cost'] = 6
    seller_params['bias_waste_cost'] = 7
    return seller_params, buyer_params

def get_mip_test_params(num_buyer, num_seller):
    buyer_params = {}
    seller_params = {}
    num_commodities = 2
    seller_params['num_commodities'] = num_commodities
    seller_params['avg_spot_price'] = 5*np.ones(num_commodities)
    seller_params['inv'] = 3*np.ones(num_commodities)
    seller_params['waste_inv'] = 2*np.ones(num_commodities)
    seller_params['alpha'] = 0.3
    return seller_params, buyer_params

def test_cvx_solver():
    num_buyer = 2
    num_seller = 2
    seller_params, buyer_params = get_test_params(num_buyer, num_seller)
    solver = CVXSolver(num_seller, num_buyer)
    solver.solve(seller_params, SELLER_PROBLEM)
    solver.solve(buyer_params, BUYER_PROBLEM)

def test_mip_solver():
    num_buyer = 2
    num_seller = 2
    seller_params, buyer_params = get_mip_test_params(num_buyer, num_seller)
    solver = MIPSolver(num_seller, num_buyer)
    solver.solve_seller(seller_params, 0)
    
if __name__=="__main__":
    pass
    #test_cvx_solver()
    #test_mip_solver()
import numpy as np
import xpress as xp

class Solver: 
    def __init__(self, num_agents: int, num_commodities: int,  horizon: int =12):
        self.num_agents = num_agents
        self.num_commodities = num_commodities
        self.horizon = horizon
        self.seller_problem = None
        self.buyer_problem = None

class XpressSolver(Solver):
    def __init__(self, num_agents: int, horizon: int =12):
        super().__init__(num_agents, horizon)

    def surrogate(self, q, c1, c2):
        return 0.5*q, 0.5*q
    
    def recycle_surrogate(self, q, c1, c2):
        return q 
    
    def utility_func(self, u, c):
        return u

    def solve_transformation(self, params, agent):
        m = xp.problem()
        # Parameters
        num_commodities = self.num_commodities 
        delta = params['delta']
        e = params['prices']
        ew = params['waste_prices']
        p = params['cost']
        pr = params['recycle_cost']
        bar_inv = params['balanced_inv']
        bar_winv = params['balanced_winv']
        # Variables
        uec = [xp.var(lb=0, vartype=xp.continuous) for _ in range(num_commodities)]
        utx = [xp.var(lb=0, vartype=xp.continuous) for _ in range(num_commodities)]
        rcy_winv = [xp.var(lb=0, vartype=xp.continuous) for _ in range(num_commodities)]
        uec = np.array(uec, dtype=xp.npvar)
        utx = np.array(utx, dtype=xp.npvar)
        rcy_winv = np.array(rcy_winv, dtype=xp.npvar)
        m.addVariable(uec, utx, rcy_winv)
        # Helper
        trans_u, trans_w = [], []
        recy_u = []
        f_inv = [] 
        f_winv = [] 
        utility = [] 
        for c1 in range(num_commodities):
            row_tu, row_tw, row_rcyu = [], [], []
            for c2 in range(num_commodities):
                u, w = self.surrogate(utx[c1], c1, c2)
                ur = self.recycle_surrogate(rcy_winv[c1], c1, c2)
                row_tu.append(u)
                row_tw.append(w)
                row_rcyu.append(ur)
            trans_u.append(row_tu)
            trans_w.append(row_tw)
            recy_u.append(row_rcyu)
            utility.append(self.utility_func(uec[c1], c1))
        
        for c1 in range(num_commodities):
            f_inv.append(bar_inv[c1]-uec[c1]-utx[c1]+xp.Sum(trans_u[c2][c1]+\
                    recy_u[c2][c1] for c2 in range(num_commodities)))
            f_winv.append(bar_winv[c1]-rcy_winv[c1]+xp.Sum(trans_w[c2][c1] for c2 in range(num_commodities)))
            f_winv[-1]*=(1-delta)

        # Objective
        m.setObjective(xp.Sum(utility[c]-p[c]*utx[c]-pr[c]*rcy_winv[c]+delta*e[c]*f_inv[c]+\
                delta*ew[c]*f_winv[c] for c in range(num_commodities)), sense=xp.maximize)
        # Optimize
        m.solve()
        status = m.getProbStatus()

        eco_utility = np.zeros(num_commodities)
        trans_quan = np.zeros(num_commodities)
        waste_inv = np.zeros(num_commodities)
        for i in range(num_commodities):
            eco_utility[i]=m.getSolution(uec[i])
            trans_quan[i]=m.getSolution(utx[i])
            waste_inv[i]=m.getSolution(rcy_winv[i])

        return m.getObjVal(), eco_utility, trans_quan, waste_inv

    def solve_buyer(self, params, agent):
        m = xp.problem()
        # Parameters
        num_commodities = self.num_commodities 
        num_other_ind = self.num_agents-1
        inv = params['inv']
        invw = params['waste_inv']
        max_inv = params['max_inv']
        max_invw = params['max_waste_inv']
        beta = params['beta']
        e = params['prices']
        ew = params['waste_prices']
        p = params['spot_price']
        # Variables
        q = [[xp.var(lb=0, vartype=xp.continuous) for _ in range(num_commodities)] for _ in range(num_other_ind)]
        qw = [[xp.var(lb=0, vartype=xp.continuous) for _ in range(num_commodities)] for _ in range(num_other_ind)]
        qs = [xp.var(lb=0, vartype=xp.continuous) for _ in range(num_commodities)]
        q =   np.array(q, dtype=xp.npvar)
        qw = np.array(qw, dtype=xp.npvar)
        qs = np.array(qs, dtype=xp.npvar)
        m.addVariable(q, qw, qs)
        # Helper
        qsum = [xp.Sum(q[n][c] for n in range(num_other_ind)) for c in range(num_commodities)]
        qwsum = [xp.Sum(qw[n][c] for n in range(num_other_ind)) for c in range(num_commodities)]
        index_comb = []
        for n in range(num_other_ind):
            for c in range(num_commodities):
                index_comb.append((n,c))
        # Objective
        m.setObjective(xp.Sum(-q[n][c]*e[n][c]-qw[n][c]*ew[n][c] for c in range(num_commodities)\
                for n in range(num_other_ind))+xp.Sum(-qs[c]*p[c] for c in range(num_commodities))\
                +beta*xp.Sum(e[agent][c]*(inv[c]+qsum[c]+qs[c])+ew[agent][c]*(invw[c]+qwsum[c])\
                for c in range(num_commodities)), sense=xp.maximize)
        # Constraints
        for c in range(num_commodities):
            #m.addConstraint(inv[c]+xp.Sum(q[n][c] for n in range(num_other_ind))+qs[c]<=max_inv)
            #m.addConstraint(inv[c]+xp.Sum(q[n][c] for n in range(num_other_ind))+qs[c]>=0.2*max_inv)
            #m.addConstraint(invw[c]+xp.Sum(qw[n][c] for n in range(num_other_ind))<=max_invw)
            #m.addConstraint(invw[c]+xp.Sum(qw[n][c] for n in range(num_other_ind))>=0.2*max_invw)
            #m.addConstraint(qsum[c]+qs[c]<=max_inv-inv[c])
            #m.addConstraint(qsum[c]+qs[c]>=0.2*max_inv-inv[c])
            #m.addConstraint(qwsum[c]<=max_invw-invw[c])
            #m.addConstraint(qwsum[c]>=0.2*max_invw-invw[c])
            m.addConstraint(inv[c]+qsum[c]+qs[c]<=max_inv[c])
            m.addConstraint(inv[c]+qsum[c]+qs[c]>=0.2*max_inv[c])
            m.addConstraint(invw[c]+qwsum[c]<=max_invw[c])
            m.addConstraint(invw[c]+qwsum[c]>=0.2*max_invw[c])
        # Optimize
        m.solve()
        status = m.getProbStatus()
        explanation = m.getProbStatusString()
        # Return
        quantity  = np.zeros((num_other_ind, num_commodities))
        quantityw = np.zeros((num_other_ind, num_commodities))
        quantitys = np.zeros(num_commodities)
        for c in range(num_commodities):
            for n in range(num_other_ind):
                quantity[n][c]=m.getSolution(q[n][c])
                quantityw[n][c]=m.getSolution(qw[n][c])
            quantitys[c]=m.getSolution(qs[c])

        quantity  = np.insert(quantity, agent-1, 0, axis=0)
        quantityw = np.insert(quantityw, agent-1, 0, axis=0)

        return m.getObjVal(), quantity, quantityw, quantitys 

    def solve_seller(self, params, agent):
        m = xp.problem()
        # Parameters
        num_commodities = self.num_commodities 
        avg_spot_price = params['avg_spot_price']
        inv = params['inv']
        waste_inv = params['waste_inv']
        alpha = params['alpha']

        # Variables
        e = [xp.var(lb=0, ub=avg_spot_price[i], vartype=xp.continuous) for i in range(num_commodities)]
        ew = [xp.var(lb=0, ub=2, vartype=xp.continuous) for _ in range(num_commodities)]
        carry_inv = [xp.var(lb=0, ub=inv[i], vartype=xp.continuous) for i in range(num_commodities)]
        carry_waste = [xp.var(lb=0, ub=waste_inv[i], vartype=xp.continuous) for i in range(num_commodities)]

        m.addVariable(e, ew, carry_inv, carry_waste)
        # Objective
        m.setObjective(xp.Sum([e[i]*(inv[i]-carry_inv[i])+ew[i]*(waste_inv[i]-carry_waste[i])+\
                alpha*(e[i]*carry_inv[i]+ew[i]*carry_waste[i]) for i in range(num_commodities)]),\
                sense=xp.maximize)

        #Optimize
        m.solve()
        status = m.getProbStatus()
        explanation = m.getProbStatusString()

        #if (status != OptimizationStatus.OPTIMAL and status != OptimizationStatus.FEASIBLE):
        #    return "Infeasible", m.objective_value, None, None, None, None 

        price = np.zeros(num_commodities)
        waste_price = np.zeros(num_commodities)
        inv = np.zeros(num_commodities)
        waste_inv = np.zeros(num_commodities)
        for i in range(num_commodities):
            price[i]=m.getSolution(e[i])
            waste_price[i]=m.getSolution(ew[i])
            inv[i]=m.getSolution(carry_inv[i])
            waste_inv[i]=m.getSolution(carry_waste[i])

        return m.getObjVal(), price, waste_price, inv, waste_inv

    def step1(self, state):
        obj = np.zeros(self.num_agents)
        price = np.zeros((self.num_agents, self.num_commodities)) 
        waste_price = np.zeros((self.num_agents, self.num_commodities)) 
        inv = np.zeros((self.num_agents, self.num_commodities))
        waste_inv = np.zeros((self.num_agents, self.num_commodities))
        for n in range(self.num_agents):
            o, e, ew, i, iw = self.solve_seller(state[n], n)
            price[n,:] = e
            waste_price[n,:] = ew
            inv[n,:] = i
            waste_inv[n,:] = iw
            obj[n] = o
        return obj, price, waste_price, inv, waste_inv
    
    def step2(self, state):
        obj = np.zeros(self.num_commodities)
        quantity = np.zeros((self.num_agents, self.num_agents, self.num_commodities))
        quantityw = np.zeros((self.num_agents, self.num_agents, self.num_commodities))
        quantitys = np.zeros((self.num_agents, self.num_commodities))
        for n in range(self.num_agents):
            o, q, qw, qs = self.solve_buyer(state[n], n)
            quantity[n, :] = q
            quantityw[n, :] = qw
            quantitys[n, :] = qs
            obj[n] = o
        return obj, quantity, quantityw, quantitys 
    
    def step3(self, state):
        obj = np.zeros(self.num_commodities)
        eco_utility = np.zeros((self.num_agents, self.num_commodities)) 
        trans_quan = np.zeros((self.num_agents, self.num_commodities))
        waste_inv = np.zeros((self.num_agents, self.num_commodities))
        for n in range(self.num_agents):
            o, uec, utx, rcy_winv = self.solve_transformation(state[n], n)
            eco_utility[n, :] = uec
            trans_quan[n,:] = utx
            waste_inv[n,:] = rcy_winv
            obj[n] = o
        return obj, eco_utility, trans_quan, waste_inv

def get_xpress_test_params(num_agents, num_commodities):
    buyer_params = {}
    seller_params = {}
    trans_params = {}
    seller_params['num_commodities'] = num_commodities
    seller_params['avg_spot_price'] = 5*np.ones(num_commodities)
    seller_params['inv'] = 3*np.ones(num_commodities)
    seller_params['waste_inv'] = 2*np.ones(num_commodities)
    seller_params['alpha'] = 0.3
    buyer_params['num_commodities'] = num_commodities
    buyer_params['inv'] = 5*np.ones(num_commodities)
    buyer_params['waste_inv'] = 2*np.ones(num_commodities)
    buyer_params['max_inv'] = 10*np.ones(num_commodities)
    buyer_params['max_waste_inv'] = 6*np.ones(num_commodities)
    buyer_params['beta'] = 0.4
    buyer_params['prices'] = 2*np.ones((num_agents-1, num_commodities))
    buyer_params['waste_prices'] = 0.5*np.ones((num_agents-1, num_commodities))
    buyer_params['spot_price'] = 1*np.ones(num_commodities)
    trans_params['num_commodities'] = num_commodities
    trans_params['delta'] = 0.3
    trans_params['prices'] = np.ones(num_commodities)
    trans_params['waste_prices'] = np.ones(num_commodities)
    trans_params['cost'] = np.ones(num_commodities)
    trans_params['recycle_cost'] = np.ones(num_commodities)
    trans_params['balanced_inv'] = np.ones(num_commodities)
    trans_params['balanced_winv'] = np.ones(num_commodities)
    return seller_params, buyer_params, trans_params

def test_xpress_solver():
    #num_buyer = 2
    #num_seller = 2
    num_agents = 2
    num_commodities = 2
    seller_params, buyer_params, trans_params = get_xpress_test_params(num_agents, num_commodities)
    solver = XpressSolver(num_agents, num_commodities)
    solver.solve_seller(seller_params, 0)
    solver.solve_buyer(buyer_params, 0)
    solver.solve_transformation(trans_params, 0)
    
if __name__=="__main__":
    test_xpress_solver()
